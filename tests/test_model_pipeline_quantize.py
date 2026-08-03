import hashlib
import importlib.util
from pathlib import Path
import shutil

import pytest

from rvai.model_pipeline.calibration import (
    load_model_pipeline_dependencies,
    select_calibration_samples,
)
from rvai.model_pipeline.config import load_pipeline_config
from rvai.model_pipeline.dataset import validate_dataset
from rvai.model_pipeline.inspect import inspect_source_model
from rvai.model_pipeline.io import canonical_json_bytes, sha256_file
from rvai.model_pipeline.quantize import QuantizationError, quantize_static_qdq
from rvai.model_pipeline.schema import (
    MobileNetV2P43BDatasetManifest,
    MobileNetV2P43BSourceModelIdentity,
)


CONFIG_DIR = Path(__file__).parents[1] / "model-pipeline" / "mobilenet-v2"
COMMITTED_IMAGE = Path(__file__).parent / "fixtures" / "onnx" / "red-image.ppm"
HAS_PIPELINE_DEPENDENCIES = all(
    importlib.util.find_spec(module) is not None
    for module in ("numpy", "onnx", "onnxruntime", "PIL")
)
pytestmark = pytest.mark.skipif(
    not HAS_PIPELINE_DEPENDENCIES,
    reason="model-pipeline optional dependencies are not installed",
)


def create_quantizable_model(path: Path, *, opset_version: int = 13):
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
        [0.01] * 3000,
    )
    bias = onnx.helper.make_tensor(
        "bias",
        onnx.TensorProto.FLOAT,
        [1000],
        [0.0] * 1000,
    )
    nodes = [
        onnx.helper.make_node("GlobalAveragePool", ["image"], ["pooled"]),
        onnx.helper.make_node("Flatten", ["pooled"], ["features"], axis=1),
        onnx.helper.make_node(
            "Gemm", ["features", "weights", "bias"], ["logits"]
        ),
    ]
    graph = onnx.helper.make_graph(
        nodes,
        "tiny-quantizable-mobilenet-contract",
        [image],
        [logits],
        [weights, bias],
    )
    model = onnx.helper.make_model(
        graph,
        producer_name="rvai-tests",
        producer_version="1.0",
        opset_imports=[onnx.helper.make_opsetid("", opset_version)],
    )
    model.ir_version = 8
    onnx.save_model(model, path)
    return onnx


def source_identity(path: Path) -> MobileNetV2P43BSourceModelIdentity:
    contents = path.read_bytes()
    return MobileNetV2P43BSourceModelIdentity(
        name="mobilenetv2-12",
        format="onnx",
        precision="fp32",
        filename="mobilenetv2-12.onnx",
        size_bytes=len(contents),
        sha256=hashlib.sha256(contents).hexdigest(),
    )


def calibration_selection(tmp_path: Path, count: int = 2):
    root = tmp_path / "dataset"
    samples = []
    for index in range(count):
        relative = f"class-{index}/image-{index}.ppm"
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(COMMITTED_IMAGE, destination)
        samples.append({"id": f"sample-{index}", "path": relative})
    manifest = MobileNetV2P43BDatasetManifest.model_validate(
        {
            "schema_version": "1.0",
            "dataset": {
                "name": "calibration-dataset",
                "version": "v1",
                "split": "calibration",
                "purpose": "calibration",
                "provenance": "Tiny committed offline fixture.",
                "license": "Test fixture only.",
            },
            "preprocessing": "mobilenet-v2-imagenet-v1",
            "sample_order": "manifest",
            "samples": samples,
        }
    )
    return select_calibration_samples(validate_dataset(manifest, root), count)


def quantization_inputs(tmp_path: Path):
    source = tmp_path / "mobilenetv2-12.onnx"
    onnx = create_quantizable_model(source)
    inspection = inspect_source_model(
        source,
        source_identity(source),
        onnx_module=onnx,
    )
    pipeline = load_pipeline_config(CONFIG_DIR / "pipeline.yaml")
    selection = calibration_selection(tmp_path)
    dependencies = load_model_pipeline_dependencies()
    return source, inspection, pipeline, selection, dependencies


def test_static_quantization_produces_checked_qdq_artifact(tmp_path: Path) -> None:
    source, inspection, pipeline, selection, dependencies = quantization_inputs(
        tmp_path
    )
    destination = tmp_path / "mobilenetv2-12-int8.onnx"

    record = quantize_static_qdq(
        source,
        destination,
        source_inspection=inspection,
        pipeline=pipeline,
        calibration=selection,
        dependencies=dependencies,
    )

    assert destination.is_file()
    assert record.checker_passed is True
    assert record.contract_matched is True
    assert record.execution_provider == "CPUExecutionProvider"
    assert record.source_opset_version == 13
    assert record.quantization_input_opset_version == 13
    assert record.opset_conversion_applied is False
    assert record.quantization.format == "qdq"
    assert record.structure.quantize_linear_count > 0
    assert record.structure.dequantize_linear_count > 0
    assert record.artifact.sha256 == sha256_file(destination)
    assert record.artifact.size_bytes == destination.stat().st_size
    assert record.calibration_sample_ids == ("sample-0", "sample-1")
    assert canonical_json_bytes(record) == canonical_json_bytes(record)
    assert type(record).model_validate_json(canonical_json_bytes(record)) == record


def test_static_quantization_converts_opset_12_input_without_changing_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "mobilenetv2-12.onnx"
    onnx = create_quantizable_model(source, opset_version=12)
    source_bytes = source.read_bytes()
    inspection = inspect_source_model(
        source,
        source_identity(source),
        onnx_module=onnx,
    )
    destination = tmp_path / "mobilenetv2-12-int8.onnx"

    record = quantize_static_qdq(
        source,
        destination,
        source_inspection=inspection,
        pipeline=load_pipeline_config(CONFIG_DIR / "pipeline.yaml"),
        calibration=calibration_selection(tmp_path),
        dependencies=load_model_pipeline_dependencies(),
    )

    assert source.read_bytes() == source_bytes
    assert record.source_opset_version == 12
    assert record.quantization_input_opset_version == 13
    assert record.opset_conversion_applied is True
    quantized = onnx.load_model(str(destination), load_external_data=False)
    assert [(item.domain, item.version) for item in quantized.opset_import] == [("", 13)]


def test_quantization_refuses_to_overwrite_existing_artifact(tmp_path: Path) -> None:
    source, inspection, pipeline, selection, dependencies = quantization_inputs(
        tmp_path
    )
    destination = tmp_path / "mobilenetv2-12-int8.onnx"
    destination.write_bytes(b"existing")

    with pytest.raises(QuantizationError, match="Refusing to overwrite"):
        quantize_static_qdq(
            source,
            destination,
            source_inspection=inspection,
            pipeline=pipeline,
            calibration=selection,
            dependencies=dependencies,
        )
    assert destination.read_bytes() == b"existing"


def test_quantization_reverifies_source_after_inspection(tmp_path: Path) -> None:
    source, inspection, pipeline, selection, dependencies = quantization_inputs(
        tmp_path
    )
    source.write_bytes(b"tampered")

    with pytest.raises(QuantizationError, match="changed after inspection"):
        quantize_static_qdq(
            source,
            tmp_path / "mobilenetv2-12-int8.onnx",
            source_inspection=inspection,
            pipeline=pipeline,
            calibration=selection,
            dependencies=dependencies,
        )


def test_quantization_failure_leaves_no_artifact_or_temporary_file(
    tmp_path: Path,
) -> None:
    source, inspection, pipeline, selection, dependencies = quantization_inputs(
        tmp_path
    )
    destination = tmp_path / "mobilenetv2-12-int8.onnx"

    def fail_quantization(**kwargs):
        raise RuntimeError("synthetic quantization failure")

    with pytest.raises(QuantizationError, match="synthetic quantization failure"):
        quantize_static_qdq(
            source,
            destination,
            source_inspection=inspection,
            pipeline=pipeline,
            calibration=selection,
            dependencies=dependencies,
            quantize_static_fn=fail_quantization,
        )

    assert not destination.exists()
    assert list(tmp_path.glob(".mobilenetv2-12-int8.onnx.*.onnx")) == []
