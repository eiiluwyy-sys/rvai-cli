from pathlib import Path
import shutil

import pytest

from rvai.model_pipeline import (
    FROZEN_MOBILENET_V2_FP32_IDENTITY,
    PipelineConfigError,
    PipelinePathError,
    load_dataset_manifest,
    load_mobilenet_v2_configuration,
    load_pipeline_config,
    load_source_model_config,
)


CONFIG_DIR = Path(__file__).parents[1] / "model-pipeline" / "mobilenet-v2"


def test_loads_committed_frozen_configuration() -> None:
    configuration = load_mobilenet_v2_configuration(CONFIG_DIR)

    assert configuration.source.model == FROZEN_MOBILENET_V2_FP32_IDENTITY
    assert configuration.source.model.filename == "mobilenetv2-12.onnx"
    assert configuration.source.model.size_bytes == 13_964_571
    assert configuration.source.model.sha256 == (
        "c0c3f76d93fa3fd6580652a45618618a220fced18babf65774ed169de0432ad5"
    )

    pipeline = configuration.pipeline
    assert pipeline.quantization.method == "static"
    assert pipeline.quantization.format == "qdq"
    assert pipeline.quantization.activation_type == "quint8"
    assert pipeline.quantization.weight_type == "qint8"
    assert pipeline.quantization.per_channel is True
    assert pipeline.quantization.calibration_method == "minmax"
    assert pipeline.quantization.execution_provider == "CPUExecutionProvider"
    assert pipeline.quantization.calibration_order == "manifest"
    assert (pipeline.calibration.pilot_samples, pipeline.calibration.production_samples) == (
        200,
        1000,
    )
    assert (pipeline.evaluation.pilot_samples, pipeline.evaluation.production_samples) == (
        200,
        5000,
    )
    assert pipeline.acceptance.max_top1_drop_percentage_points == 1.0
    assert pipeline.acceptance.max_top5_drop_percentage_points == 1.0
    assert pipeline.acceptance.min_model_size_reduction_ratio == 0.50
    assert pipeline.acceptance.min_top1_agreement_ratio == 0.95
    assert pipeline.acceptance.require_zero_inference_failures is True
    assert pipeline.acceptance.require_finite_outputs is True


def test_committed_example_manifests_are_strict_and_purpose_aware() -> None:
    calibration = load_dataset_manifest(
        CONFIG_DIR / "calibration-dataset.example.yaml",
        expected_purpose="calibration",
    )
    evaluation = load_dataset_manifest(
        CONFIG_DIR / "evaluation-dataset.example.yaml",
        expected_purpose="evaluation",
    )

    assert calibration.samples[0].label is None
    assert all(sample.label is not None for sample in evaluation.samples)
    assert {sample.id for sample in calibration.samples}.isdisjoint(
        sample.id for sample in evaluation.samples
    )


def test_dataset_loader_rejects_wrong_expected_purpose() -> None:
    with pytest.raises(PipelineConfigError, match="expected 'evaluation'"):
        load_dataset_manifest(
            CONFIG_DIR / "calibration-dataset.example.yaml",
            expected_purpose="evaluation",
        )


@pytest.mark.parametrize(
    ("old", "new", "field"),
    [
        ("mobilenetv2-12.onnx", "changed.onnx", "filename"),
        ("size_bytes: 13964571", "size_bytes: 13964572", "size_bytes"),
        (
            "c0c3f76d93fa3fd6580652a45618618a220fced18babf65774ed169de0432ad5",
            "0" * 64,
            "sha256",
        ),
    ],
)
def test_source_loader_rejects_frozen_identity_mismatch(
    old: str, new: str, field: str, tmp_path: Path
) -> None:
    source = tmp_path / "source.yaml"
    contents = (CONFIG_DIR / "source-fp32.yaml").read_text(encoding="utf-8")
    source.write_text(contents.replace(old, new), encoding="utf-8")

    with pytest.raises(PipelineConfigError, match=field):
        load_source_model_config(source)


def test_pipeline_loader_rejects_unknown_fields(tmp_path: Path) -> None:
    pipeline = tmp_path / "pipeline.yaml"
    contents = (CONFIG_DIR / "pipeline.yaml").read_text(encoding="utf-8")
    pipeline.write_text(contents + "unknown: true\n", encoding="utf-8")
    with pytest.raises(PipelineConfigError, match="extra_forbidden"):
        load_pipeline_config(pipeline)


def test_source_reference_symlink_cannot_escape_configuration_directory(
    tmp_path: Path,
) -> None:
    configuration_dir = tmp_path / "configuration"
    configuration_dir.mkdir()
    shutil.copy(CONFIG_DIR / "pipeline.yaml", configuration_dir / "pipeline.yaml")
    external_source = tmp_path / "external-source.yaml"
    shutil.copy(CONFIG_DIR / "source-fp32.yaml", external_source)
    (configuration_dir / "source-fp32.yaml").symlink_to(external_source)

    with pytest.raises(PipelinePathError, match="escapes"):
        load_mobilenet_v2_configuration(configuration_dir)


def test_pipeline_source_reference_must_be_canonical_relative(tmp_path: Path) -> None:
    configuration_dir = tmp_path / "configuration"
    shutil.copytree(CONFIG_DIR, configuration_dir)
    pipeline = configuration_dir / "pipeline.yaml"
    contents = pipeline.read_text(encoding="utf-8")
    pipeline.write_text(
        contents.replace("source_model: source-fp32.yaml", "source_model: ../source.yaml"),
        encoding="utf-8",
    )

    with pytest.raises(PipelineConfigError, match="parent components"):
        load_mobilenet_v2_configuration(configuration_dir)
