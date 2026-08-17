import importlib.util
import subprocess
from pathlib import Path
from typing import Literal

import pytest

from rvai.model_pipeline.calibration import load_model_pipeline_dependencies
from rvai.model_pipeline.config import load_pipeline_config
from rvai.model_pipeline.dataset import DatasetOverlapError
from rvai.model_pipeline.io import canonical_json_text, load_json, sha256_file
from rvai.model_pipeline.production import (
    MobileNetV2P43BProductionFailureRecord,
    MobileNetV2P43BProductionReport,
    ProductionRunError,
    run_production_pipeline,
    validate_production_evidence,
)
from rvai.model_pipeline.production_package import (
    ProductionPackageError,
    build_production_evidence_package,
    verify_production_evidence_package,
)
from rvai.model_pipeline.schema import (
    MobileNetV2P43BConfiguration,
    MobileNetV2P43BDatasetIdentity,
    MobileNetV2P43BDatasetManifest,
    MobileNetV2P43BDatasetSample,
    MobileNetV2P43BPipelineConfig,
    MobileNetV2P43BSampleCountConfig,
    MobileNetV2P43BSourceModelConfig,
    MobileNetV2P43BSourceModelIdentity,
)

CONFIG_DIR = Path(__file__).parents[1] / "model-pipeline" / "mobilenet-v2"
HAS_PIPELINE_DEPENDENCIES = all(
    importlib.util.find_spec(module) is not None
    for module in ("numpy", "onnx", "onnxruntime", "PIL")
)
pytestmark = pytest.mark.skipif(
    not HAS_PIPELINE_DEPENDENCIES,
    reason="model-pipeline optional dependencies are not installed",
)


def _create_quantizable_model(path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    image = onnx.helper.make_tensor_value_info(
        "image", onnx.TensorProto.FLOAT, [1, 3, 224, 224]
    )
    logits = onnx.helper.make_tensor_value_info(
        "logits", onnx.TensorProto.FLOAT, [1, 1000]
    )
    weights = onnx.helper.make_tensor(
        "weights",
        onnx.TensorProto.FLOAT,
        [3, 1000],
        [((index % 17) - 8) / 100.0 for index in range(3000)],
    )
    bias = onnx.helper.make_tensor(
        "bias",
        onnx.TensorProto.FLOAT,
        [1000],
        [index / 1000.0 for index in range(1000)],
    )
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("GlobalAveragePool", ["image"], ["pooled"]),
            onnx.helper.make_node("Flatten", ["pooled"], ["features"], axis=1),
            onnx.helper.make_node(
                "Gemm", ["features", "weights", "bias"], ["logits"]
            ),
        ],
        "tiny-production-model",
        [image],
        [logits],
        [weights, bias],
    )
    model = onnx.helper.make_model(
        graph,
        producer_name="rvai-tests",
        producer_version="1.0",
        opset_imports=[onnx.helper.make_opsetid("", 13)],
    )
    model.ir_version = 8
    onnx.save_model(model, path)


def _write_ppm(path: Path, rgb: tuple[int, int, int]) -> None:
    path.write_bytes(b"P6\n256 256\n255\n" + bytes(rgb) * (256 * 256))


def _clean_repository(path: Path) -> Path:
    path.mkdir()
    subprocess.run(["git", "init", "--quiet", str(path)], check=True)
    (path / "source.txt").write_text("formal-p43b\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "source.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "-c",
            "user.name=RVAI Tests",
            "-c",
            "user.email=rvai@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "fixture",
        ],
        check=True,
    )
    return path


def _configuration(model_path: Path) -> MobileNetV2P43BConfiguration:
    pipeline = load_pipeline_config(CONFIG_DIR / "pipeline.yaml")
    pipeline = MobileNetV2P43BPipelineConfig.model_validate(
        {
            **pipeline.model_dump(mode="python"),
            "calibration": MobileNetV2P43BSampleCountConfig(
                pilot_samples=1,
                production_samples=2,
            ),
            "evaluation": MobileNetV2P43BSampleCountConfig(
                pilot_samples=1,
                production_samples=3,
            ),
        }
    )
    return MobileNetV2P43BConfiguration(
        pipeline=pipeline,
        source=MobileNetV2P43BSourceModelConfig(
            model=MobileNetV2P43BSourceModelIdentity(
                name="mobilenetv2-12",
                format="onnx",
                precision="fp32",
                filename=model_path.name,
                size_bytes=model_path.stat().st_size,
                sha256=sha256_file(model_path),
            )
        ),
    )


def _manifest(
    root: Path,
    path: Path,
    *,
    purpose: Literal["calibration", "evaluation"],
    colors: tuple[tuple[int, int, int], ...],
) -> MobileNetV2P43BDatasetManifest:
    samples = []
    for index, color in enumerate(colors):
        image = root / f"{purpose}-{index}.ppm"
        _write_ppm(image, color)
        samples.append(
            MobileNetV2P43BDatasetSample(
                id=f"{purpose}-{index}",
                path=image.name,
                label=999 if purpose == "evaluation" else None,
                sha256=sha256_file(image),
            )
        )
    manifest = MobileNetV2P43BDatasetManifest.model_validate(
        {
            "dataset": MobileNetV2P43BDatasetIdentity(
                name=f"tiny-{purpose}",
                version="fixture-v1",
                split=purpose,
                purpose=purpose,
                provenance="Deterministic independently labelled test fixture.",
                license="Generated test fixture; Apache-2.0 repository use.",
            ),
            "preprocessing": "mobilenet-v2-imagenet-v1",
            "sample_order": "manifest",
            "samples": tuple(samples),
        }
    )
    path.write_text(canonical_json_text(manifest), encoding="utf-8")
    return manifest


def _formal_run(tmp_path: Path) -> tuple[Path, Path]:
    model = tmp_path / "mobilenetv2-12.onnx"
    _create_quantizable_model(model)
    calibration_root = tmp_path / "calibration"
    evaluation_root = tmp_path / "evaluation"
    calibration_root.mkdir()
    evaluation_root.mkdir()
    calibration_manifest = tmp_path / "calibration.yaml"
    evaluation_manifest = tmp_path / "evaluation.yaml"
    _manifest(
        calibration_root,
        calibration_manifest,
        purpose="calibration",
        colors=((5, 10, 15), (20, 25, 30)),
    )
    _manifest(
        evaluation_root,
        evaluation_manifest,
        purpose="evaluation",
        colors=((80, 90, 100), (120, 130, 140)),
    )
    repository = _clean_repository(tmp_path / "repository")
    output = tmp_path / "production-run"
    run_production_pipeline(
        model,
        calibration_manifest,
        calibration_root,
        evaluation_manifest,
        evaluation_root,
        output,
        configuration=_configuration(model),
        repository=repository,
        dependencies=load_model_pipeline_dependencies(),
    )
    return output, repository


def test_formal_production_run_and_package_verify_offline(tmp_path: Path) -> None:
    output, repository = _formal_run(tmp_path)
    evidence = validate_production_evidence(output)

    assert evidence.report.label_source == "independent-ground-truth"
    assert evidence.report.evaluation_requested_sample_count == 3
    assert evidence.report.evaluation_actual_sample_count == 2
    assert evidence.report.evaluation_used_complete_available_set is True
    assert evidence.report.production_verified == (
        evidence.report.status == "accepted"
    )
    assert load_json(
        output / "records" / "production-report.json",
        MobileNetV2P43BProductionReport,
    ) == evidence.report

    package = tmp_path / "production-package"
    built = build_production_evidence_package(
        output,
        package,
        repository=repository,
    )
    assert verify_production_evidence_package(package) == built
    assert built.production_verified == evidence.report.production_verified
    assert not (package / "mobilenetv2-12.onnx").exists()
    assert not (package / "dataset").exists()

    (package / "comparison.md").write_text("tampered", encoding="utf-8")
    with pytest.raises(ProductionPackageError, match="identity mismatch"):
        verify_production_evidence_package(package)


def test_formal_run_retains_failure_when_source_tree_is_dirty(tmp_path: Path) -> None:
    model = tmp_path / "mobilenetv2-12.onnx"
    _create_quantizable_model(model)
    root = tmp_path / "data"
    root.mkdir()
    calibration = tmp_path / "calibration.yaml"
    evaluation = tmp_path / "evaluation.yaml"
    _manifest(
        root,
        calibration,
        purpose="calibration",
        colors=((1, 2, 3), (4, 5, 6)),
    )
    _manifest(
        root,
        evaluation,
        purpose="evaluation",
        colors=((7, 8, 9), (10, 11, 12)),
    )
    repository = _clean_repository(tmp_path / "repository")
    (repository / "dirty.txt").write_text("dirty\n", encoding="utf-8")
    output = tmp_path / "failed-run"

    with pytest.raises(ProductionRunError, match="clean Git"):
        run_production_pipeline(
            model,
            calibration,
            root,
            evaluation,
            root,
            output,
            configuration=_configuration(model),
            repository=repository,
        )

    failure = load_json(
        output / "records" / "failure.json",
        MobileNetV2P43BProductionFailureRecord,
    )
    assert failure.failed_stage == "source-revision"
    assert failure.production_verified is False
    assert failure.completed_records == (
        "records/calibration-manifest.json",
        "records/evaluation-manifest.json",
        "records/pipeline-config.json",
        "records/source-model-config.json",
    )


def test_formal_run_retains_completed_records_after_dataset_overlap(
    tmp_path: Path,
) -> None:
    model = tmp_path / "mobilenetv2-12.onnx"
    _create_quantizable_model(model)
    calibration_root = tmp_path / "calibration"
    evaluation_root = tmp_path / "evaluation"
    calibration_root.mkdir()
    evaluation_root.mkdir()
    calibration = tmp_path / "calibration.yaml"
    evaluation = tmp_path / "evaluation.yaml"
    _manifest(
        calibration_root,
        calibration,
        purpose="calibration",
        colors=((1, 2, 3), (4, 5, 6)),
    )
    _manifest(
        evaluation_root,
        evaluation,
        purpose="evaluation",
        colors=((1, 2, 3), (10, 11, 12)),
    )
    repository = _clean_repository(tmp_path / "repository")
    output = tmp_path / "failed-overlap-run"

    with pytest.raises(DatasetOverlapError, match="overlap"):
        run_production_pipeline(
            model,
            calibration,
            calibration_root,
            evaluation,
            evaluation_root,
            output,
            configuration=_configuration(model),
            repository=repository,
            dependencies=load_model_pipeline_dependencies(),
        )

    failure = load_json(
        output / "records" / "failure.json",
        MobileNetV2P43BProductionFailureRecord,
    )
    assert failure.failed_stage == "dataset-overlap"
    assert "records/source-inspection.json" in failure.completed_records
    assert "records/calibration-validation.json" in failure.completed_records
    assert "records/evaluation-validation.json" in failure.completed_records
    assert not tuple((output / "models").iterdir())
