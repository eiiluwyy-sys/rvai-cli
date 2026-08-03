import hashlib
import importlib.util
from pathlib import Path

import pytest

from rvai.model_pipeline.calibration import load_model_pipeline_dependencies
from rvai.model_pipeline.config import load_pipeline_config
from rvai.model_pipeline.io import load_json
from rvai.model_pipeline.pilot import (
    MobileNetV2P43BProxyPilotReport,
    ProxyPilotError,
    run_synthetic_proxy_pilot,
)
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
    assert sorted(path.name for path in (output / "records").iterdir()) == [
        "calibration-validation.json",
        "comparison.json",
        "evaluation-validation.json",
        "fp32-evaluation.json",
        "int8-evaluation.json",
        "overlap.json",
        "proxy-pilot-report.json",
        "quantization.json",
        "source-inspection.json",
    ]
