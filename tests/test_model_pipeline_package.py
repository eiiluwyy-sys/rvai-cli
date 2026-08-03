import hashlib
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

import pytest

from rvai.model_pipeline import package as package_module
from rvai.model_pipeline.calibration import load_model_pipeline_dependencies
from rvai.model_pipeline.config import load_pipeline_config
from rvai.model_pipeline.environment import (
    MobileNetV2P43BReproducibilityRecord,
    MobileNetV2P43BSourceRevision,
)
from rvai.model_pipeline.io import canonical_json_bytes, load_json
from rvai.model_pipeline.package import (
    EvidencePackageError,
    MobileNetV2P43BPackageManifest,
    PACKAGE_PAYLOAD_PATHS,
    build_evidence_package,
    verify_evidence_package,
)
from rvai.model_pipeline.pilot import run_synthetic_proxy_pilot
from rvai.model_pipeline.schema import (
    MobileNetV2P43BConfiguration,
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


def clean_revision(commit: str = "a" * 40) -> MobileNetV2P43BSourceRevision:
    return MobileNetV2P43BSourceRevision(
        vcs="git",
        commit=commit,
        working_tree_clean=True,
    )


def create_quantizable_model(path: Path) -> None:
    import onnx

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
        "tiny-package-model",
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


@pytest.fixture(scope="module")
def valid_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("package-run")
    model_path = root / "mobilenetv2-12.onnx"
    create_quantizable_model(model_path)
    contents = model_path.read_bytes()
    configuration = MobileNetV2P43BConfiguration(
        pipeline=load_pipeline_config(CONFIG_DIR / "pipeline.yaml"),
        source=MobileNetV2P43BSourceModelConfig(
            model=MobileNetV2P43BSourceModelIdentity(
                name="mobilenetv2-12",
                format="onnx",
                precision="fp32",
                filename="mobilenetv2-12.onnx",
                size_bytes=len(contents),
                sha256=hashlib.sha256(contents).hexdigest(),
            )
        ),
    )
    import rvai.model_pipeline.pilot as pilot_module

    original_capture = pilot_module.capture_reproducibility_record

    def capture_clean(**kwargs):
        record = original_capture(**kwargs)
        return record.model_copy(update={"source_revision": clean_revision("b" * 40)})

    output = root / "run"
    with patch.object(pilot_module, "capture_reproducibility_record", capture_clean):
        run_synthetic_proxy_pilot(
            model_path,
            output,
            configuration=configuration,
            calibration_sample_count=2,
            evaluation_sample_count=2,
            dependencies=load_model_pipeline_dependencies(),
        )
    return output


def copy_run(valid_run: Path, tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    destination = tmp_path / "run"
    shutil.copytree(valid_run, destination)
    return destination


def build(run: Path, destination: Path):
    return build_evidence_package(
        run,
        destination,
        revision_provider=lambda repository: clean_revision("c" * 40),
    )


def rewrite_reproducibility(run: Path, **updates) -> None:
    path = run / "records" / "reproducibility.json"
    record = load_json(path, MobileNetV2P43BReproducibilityRecord)
    changed = record.model_copy(update=updates)
    path.write_bytes(canonical_json_bytes(changed))


def package_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_repeated_package_construction_is_deterministic_and_complete(
    valid_run: Path,
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    manifest = build(valid_run, first)
    build(valid_run, second)

    assert package_bytes(first) == package_bytes(second)
    assert tuple(entry.path for entry in manifest.entries) == PACKAGE_PAYLOAD_PATHS
    assert list(PACKAGE_PAYLOAD_PATHS) == sorted(PACKAGE_PAYLOAD_PATHS)
    assert "package-manifest.json" not in PACKAGE_PAYLOAD_PATHS
    assert "sha256sums.txt" not in PACKAGE_PAYLOAD_PATHS
    assert len(manifest.entries) == 19
    assert verify_evidence_package(first) == manifest
    assert not any(path.name.endswith(".staging") for path in tmp_path.iterdir())


@pytest.mark.parametrize(
    "relative_path",
    [
        "comparison.md",
        "records/comparison.json",
        "models/mobilenetv2-12-int8.onnx",
    ],
)
def test_verifier_rejects_tampered_payload_byte(
    valid_run: Path,
    tmp_path: Path,
    relative_path: str,
) -> None:
    destination = tmp_path / "package"
    build(valid_run, destination)
    path = destination / relative_path
    data = bytearray(path.read_bytes())
    data[0] ^= 1
    path.write_bytes(data)

    with pytest.raises(EvidencePackageError, match="identity mismatch"):
        verify_evidence_package(destination)


def test_verifier_rejects_missing_and_additional_files(
    valid_run: Path,
    tmp_path: Path,
) -> None:
    missing_package = tmp_path / "missing"
    build(valid_run, missing_package)
    (missing_package / "records" / "comparison.json").unlink()
    with pytest.raises(EvidencePackageError, match="inventory mismatch"):
        verify_evidence_package(missing_package)

    additional_package = tmp_path / "additional"
    build(valid_run, additional_package)
    (additional_package / "dataset-image.ppm").write_bytes(b"forbidden")
    with pytest.raises(EvidencePackageError, match="inventory mismatch"):
        verify_evidence_package(additional_package)


def test_builder_rejects_modified_int8_model_and_invalid_json(
    valid_run: Path,
    tmp_path: Path,
) -> None:
    model_run = copy_run(valid_run, tmp_path / "model-case")
    with (model_run / "models" / "mobilenetv2-12-int8.onnx").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(EvidencePackageError, match="INT8 model identity"):
        build(model_run, tmp_path / "model-package")

    json_run = copy_run(valid_run, tmp_path / "json-case")
    (json_run / "records" / "comparison.json").write_text("{", encoding="utf-8")
    with pytest.raises(EvidencePackageError, match="Invalid evidence record"):
        build(json_run, tmp_path / "json-package")


def test_builder_rejects_invalid_digest_link(valid_run: Path, tmp_path: Path) -> None:
    run = copy_run(valid_run, tmp_path)
    path = run / "records" / "reproducibility.json"
    record = load_json(path, MobileNetV2P43BReproducibilityRecord)
    rewrite_reproducibility(
        run,
        outputs=record.outputs.model_copy(update={"comparison_sha256": "0" * 64}),
    )

    with pytest.raises(EvidencePackageError, match="output digest link"):
        build(run, tmp_path / "package")


def test_verifier_rejects_altered_sha256sums(valid_run: Path, tmp_path: Path) -> None:
    destination = tmp_path / "package"
    build(valid_run, destination)
    with (destination / "sha256sums.txt").open("ab") as handle:
        handle.write(b"0" * 64 + b"  extra\n")

    with pytest.raises(EvidencePackageError, match="sha256sums"):
        verify_evidence_package(destination)


def test_verifier_rejects_symlink(valid_run: Path, tmp_path: Path) -> None:
    destination = tmp_path / "package"
    build(valid_run, destination)
    path = destination / "records" / "comparison.json"
    contents = path.read_bytes()
    external = tmp_path / "external.json"
    external.write_bytes(contents)
    path.unlink()
    path.symlink_to(external)

    with pytest.raises(EvidencePackageError, match="Symlinks"):
        verify_evidence_package(destination)


def test_builder_refuses_existing_destination(valid_run: Path, tmp_path: Path) -> None:
    destination = tmp_path / "existing"
    destination.mkdir()

    with pytest.raises(EvidencePackageError, match="Refusing to overwrite"):
        build(valid_run, destination)


def test_builder_rejects_dirty_run_and_packager_revisions(
    valid_run: Path,
    tmp_path: Path,
) -> None:
    run = copy_run(valid_run, tmp_path)
    rewrite_reproducibility(
        run,
        source_revision=clean_revision("d" * 40).model_copy(
            update={"working_tree_clean": False}
        ),
    )
    with pytest.raises(EvidencePackageError, match="dirty worktree"):
        build(run, tmp_path / "dirty-run-package")

    dirty = clean_revision("e" * 40).model_copy(
        update={"working_tree_clean": False}
    )
    with pytest.raises(EvidencePackageError, match="Packager.*dirty"):
        build_evidence_package(
            valid_run,
            tmp_path / "dirty-packager-package",
            revision_provider=lambda repository: dirty,
        )


def test_builder_cleans_staging_after_failure(
    valid_run: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "package"

    def fail_verification(path):
        raise EvidencePackageError("injected staged verification failure")

    monkeypatch.setattr(package_module, "verify_evidence_package", fail_verification)
    with pytest.raises(EvidencePackageError, match="injected"):
        build(valid_run, destination)

    assert not destination.exists()
    assert not list(tmp_path.glob(".package.*.staging"))


def test_non_public_script_verifies_without_traceback(
    valid_run: Path,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "package"
    build(valid_run, destination)
    script = Path(__file__).parents[1] / "scripts" / "package_p43b_evidence.py"

    completed = subprocess.run(
        (
            sys.executable,
            str(script),
            "verify",
            "--package",
            str(destination),
        ),
        cwd=Path(__file__).parents[1],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0
    assert "verified" in completed.stdout
    assert "Traceback" not in completed.stderr
