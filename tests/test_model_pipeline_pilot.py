import hashlib
import importlib.util
from pathlib import Path

import pytest

from rvai.model_pipeline.calibration import load_model_pipeline_dependencies
from rvai.model_pipeline.config import load_pipeline_config
from rvai.model_pipeline.environment import MobileNetV2P43BReproducibilityRecord
from rvai.model_pipeline.io import load_json, sha256_file
from rvai.model_pipeline.pilot import (
    MobileNetV2P43BProxyPilotReport,
    ProxyPilotError,
    run_synthetic_proxy_pilot,
)
from rvai.model_pipeline.schema import (
    MobileNetV2P43BConfiguration,
    MobileNetV2P43BDatasetManifest,
    MobileNetV2P43BPipelineConfig,
    MobileNetV2P43BSourceModelConfig,
    MobileNetV2P43BSourceModelIdentity,
)
from rvai.model_pipeline.synthetic import (
    MobileNetV2P43BPseudoLabelRecord,
    MobileNetV2P43BSyntheticGenerationRecord,
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


def create_quantizable_model(path: Path):
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
        "tiny-proxy-pilot-model",
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


def test_proxy_pilot_requires_one_configuration_source(tmp_path: Path) -> None:
    with pytest.raises(ProxyPilotError, match="exactly one"):
        run_synthetic_proxy_pilot(tmp_path / "missing.onnx", tmp_path / "output")


def test_tiny_proxy_pilot_runs_b1_through_b3_offline(tmp_path: Path) -> None:
    model_path = tmp_path / "mobilenetv2-12.onnx"
    create_quantizable_model(model_path)
    contents = model_path.read_bytes()
    source = MobileNetV2P43BSourceModelConfig(
        model=MobileNetV2P43BSourceModelIdentity(
            name="mobilenetv2-12",
            format="onnx",
            precision="fp32",
            filename="mobilenetv2-12.onnx",
            size_bytes=len(contents),
            sha256=hashlib.sha256(contents).hexdigest(),
        )
    )
    configuration = MobileNetV2P43BConfiguration(
        pipeline=load_pipeline_config(CONFIG_DIR / "pipeline.yaml"),
        source=source,
    )
    output = tmp_path / "pilot"

    result = run_synthetic_proxy_pilot(
        model_path,
        output,
        configuration=configuration,
        calibration_sample_count=2,
        evaluation_sample_count=2,
        dependencies=load_model_pipeline_dependencies(),
    )

    assert result.report.status == "provisional"
    assert result.report.production_verified is False
    assert result.report.label_source == "fp32-top1-pseudo-label"
    assert result.fp32_evaluation.summary.top1_accuracy_ratio == 1.0
    assert result.quantization.structure.quantize_linear_count > 0
    assert result.quantization.structure.dequantize_linear_count > 0
    assert result.overlap.overlap_count == 0
    assert (output / "models" / "mobilenetv2-12-int8.onnx").is_file()
    report_path = output / "records" / "proxy-pilot-report.json"
    assert load_json(report_path, MobileNetV2P43BProxyPilotReport) == result.report
    records = output / "records"
    assert load_json(
        records / "pipeline-config.json",
        MobileNetV2P43BPipelineConfig,
    ) == configuration.pipeline
    assert load_json(
        records / "source-model-config.json",
        MobileNetV2P43BSourceModelConfig,
    ) == configuration.source
    assert load_json(
        records / "calibration-manifest.json",
        MobileNetV2P43BDatasetManifest,
    ).dataset.purpose == "calibration"
    assert load_json(
        records / "evaluation-unlabeled-manifest.json",
        MobileNetV2P43BDatasetManifest,
    ).dataset.split == "evaluation-unlabeled"
    assert load_json(
        records / "evaluation-manifest.json",
        MobileNetV2P43BDatasetManifest,
    ).dataset.purpose == "evaluation"
    assert load_json(
        records / "generation.json",
        MobileNetV2P43BSyntheticGenerationRecord,
    ) == result.generation
    assert load_json(
        records / "pseudo-labels.json",
        MobileNetV2P43BPseudoLabelRecord,
    ) == result.pseudo_labels
    assert load_json(
        records / "reproducibility.json",
        MobileNetV2P43BReproducibilityRecord,
    ) == result.reproducibility
    assert (records / "generation.json").read_bytes() == (
        output / "dataset" / "generation.json"
    ).read_bytes()
    assert (records / "pseudo-labels.json").read_bytes() == (
        output / "dataset" / "pseudo-labels.json"
    ).read_bytes()
    input_files = {
        "pipeline_config_sha256": records / "pipeline-config.json",
        "source_model_config_sha256": records / "source-model-config.json",
        "source_fp32_model_sha256": model_path,
        "calibration_manifest_sha256": records / "calibration-manifest.json",
        "unlabeled_evaluation_manifest_sha256": (
            records / "evaluation-unlabeled-manifest.json"
        ),
        "evaluation_manifest_sha256": records / "evaluation-manifest.json",
    }
    for field, path in input_files.items():
        assert getattr(result.reproducibility.inputs, field) == sha256_file(path)
    output_files = {
        "source_inspection_sha256": records / "source-inspection.json",
        "generation_record_sha256": records / "generation.json",
        "pseudo_label_record_sha256": records / "pseudo-labels.json",
        "calibration_validation_sha256": records / "calibration-validation.json",
        "evaluation_validation_sha256": records / "evaluation-validation.json",
        "overlap_report_sha256": records / "overlap.json",
        "quantization_record_sha256": records / "quantization.json",
        "int8_model_sha256": output / "models" / "mobilenetv2-12-int8.onnx",
        "fp32_evaluation_sha256": records / "fp32-evaluation.json",
        "int8_evaluation_sha256": records / "int8-evaluation.json",
        "comparison_sha256": records / "comparison.json",
        "proxy_pilot_report_sha256": records / "proxy-pilot-report.json",
    }
    for field, path in output_files.items():
        assert getattr(result.reproducibility.outputs, field) == sha256_file(path)
    assert result.reproducibility.status == "provisional"
    assert result.reproducibility.production_verified is False
    assert result.reproducibility.label_source == "fp32-top1-pseudo-label"
    assert sorted(path.name for path in (output / "records").iterdir()) == [
        "calibration-manifest.json",
        "calibration-validation.json",
        "comparison.json",
        "evaluation-manifest.json",
        "evaluation-unlabeled-manifest.json",
        "evaluation-validation.json",
        "fp32-evaluation.json",
        "generation.json",
        "int8-evaluation.json",
        "overlap.json",
        "pipeline-config.json",
        "proxy-pilot-report.json",
        "pseudo-labels.json",
        "quantization.json",
        "reproducibility.json",
        "source-inspection.json",
        "source-model-config.json",
    ]
    with pytest.raises(ProxyPilotError, match="Refusing to overwrite"):
        run_synthetic_proxy_pilot(
            model_path,
            output,
            configuration=configuration,
            calibration_sample_count=2,
            evaluation_sample_count=2,
            dependencies=load_model_pipeline_dependencies(),
        )
