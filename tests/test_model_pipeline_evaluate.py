import hashlib
import importlib.util
from pathlib import Path
import shutil

import pytest

from rvai.model_pipeline.config import load_pipeline_config
from rvai.model_pipeline.dataset import validate_dataset
from rvai.model_pipeline.evaluate import (
    EvaluationError,
    MobileNetV2P43BEvaluationArtifact,
    MobileNetV2P43BEvaluationRecord,
    evaluate_model,
    select_evaluation_samples,
)
from rvai.model_pipeline.io import load_json, write_canonical_json
from rvai.model_pipeline.schema import MobileNetV2P43BDatasetManifest


CONFIG_DIR = Path(__file__).parents[1] / "model-pipeline" / "mobilenet-v2"
COMMITTED_IMAGE = Path(__file__).parent / "fixtures" / "onnx" / "red-image.ppm"
HAS_PIPELINE_DEPENDENCIES = all(
    importlib.util.find_spec(module) is not None
    for module in ("numpy", "onnx", "onnxruntime", "PIL")
)


def evaluation_dataset(tmp_path: Path, count: int = 2):
    root = tmp_path / "evaluation"
    samples = []
    for index in range(count):
        relative = f"class-{index}/image-{index}.ppm"
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(COMMITTED_IMAGE, destination)
        samples.append({"id": f"sample-{index}", "path": relative, "label": index})
    manifest = MobileNetV2P43BDatasetManifest.model_validate(
        {
            "schema_version": "1.0",
            "dataset": {
                "name": "evaluation-dataset",
                "version": "v1",
                "split": "validation",
                "purpose": "evaluation",
                "provenance": "Tiny committed offline fixture.",
                "license": "Test fixture only.",
            },
            "preprocessing": "mobilenet-v2-imagenet-v1",
            "sample_order": "manifest",
            "samples": samples,
        }
    )
    return validate_dataset(manifest, root)


def test_evaluation_selection_preserves_manifest_order(tmp_path: Path) -> None:
    dataset = evaluation_dataset(tmp_path, 3)

    selection = select_evaluation_samples(dataset, 2)

    assert selection.record.requested_sample_count == 2
    assert selection.record.actual_sample_count == 2
    assert selection.record.used_complete_available_set is False
    assert selection.record.sample_ids == ("sample-0", "sample-1")


def test_production_selection_records_complete_short_dataset(tmp_path: Path) -> None:
    dataset = evaluation_dataset(tmp_path, 2)

    selection = select_evaluation_samples(
        dataset,
        5,
        allow_complete_dataset_if_fewer=True,
    )

    assert selection.record.requested_sample_count == 5
    assert selection.record.actual_sample_count == 2
    assert selection.record.used_complete_available_set is True


def test_pilot_selection_rejects_too_few_samples(tmp_path: Path) -> None:
    dataset = evaluation_dataset(tmp_path, 1)

    with pytest.raises(EvaluationError, match="requires 2 samples"):
        select_evaluation_samples(dataset, 2)


def test_model_identity_is_verified_before_runtime_loading(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"not the declared model")
    artifact = MobileNetV2P43BEvaluationArtifact(
        role="fp32",
        filename="model.onnx",
        size_bytes=1,
        sha256="0" * 64,
    )

    with pytest.raises(EvaluationError, match="identity mismatch"):
        evaluate_model(
            model,
            artifact=artifact,
            pipeline=load_pipeline_config(CONFIG_DIR / "pipeline.yaml"),
            evaluation=select_evaluation_samples(evaluation_dataset(tmp_path), 2),
        )


@pytest.mark.skipif(
    not HAS_PIPELINE_DEPENDENCIES,
    reason="model-pipeline optional dependencies are not installed",
)
def test_evaluate_tiny_model_with_cpu_provider(tmp_path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    image = onnx.helper.make_tensor_value_info(
        "image", onnx.TensorProto.FLOAT, ["batch_size", 3, 224, 224]
    )
    logits = onnx.helper.make_tensor_value_info(
        "logits", onnx.TensorProto.FLOAT, ["batch_size", 1000]
    )
    weights = onnx.helper.make_tensor(
        "weights", onnx.TensorProto.FLOAT, [3, 1000], [0.0] * 3000
    )
    bias_values = [0.0] * 1000
    bias_values[0] = 10.0
    bias = onnx.helper.make_tensor(
        "bias", onnx.TensorProto.FLOAT, [1000], bias_values
    )
    nodes = [
        onnx.helper.make_node("GlobalAveragePool", ["image"], ["pooled"]),
        onnx.helper.make_node("Flatten", ["pooled"], ["features"], axis=1),
        onnx.helper.make_node("Gemm", ["features", "weights", "bias"], ["logits"]),
    ]
    graph = onnx.helper.make_graph(
        nodes, "tiny-evaluation-model", [image], [logits], [weights, bias]
    )
    model_proto = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 13)]
    )
    model_proto.ir_version = 8
    model = tmp_path / "tiny-fp32.onnx"
    onnx.save_model(model_proto, model)
    contents = model.read_bytes()
    artifact = MobileNetV2P43BEvaluationArtifact(
        role="fp32",
        filename=model.name,
        size_bytes=len(contents),
        sha256=hashlib.sha256(contents).hexdigest(),
    )

    record = evaluate_model(
        model,
        artifact=artifact,
        pipeline=load_pipeline_config(CONFIG_DIR / "pipeline.yaml"),
        evaluation=select_evaluation_samples(evaluation_dataset(tmp_path, 1), 1),
    )

    assert record.execution_provider == "CPUExecutionProvider"
    assert record.summary.sample_count == 1
    assert record.summary.inference_failures == 0
    assert record.summary.all_outputs_finite is True
    assert record.summary.top1_accuracy_ratio == 1.0
    assert record.samples[0].top1_class == 0
    assert "/" not in record.samples[0].sample_id
    record_path = tmp_path / "evaluation.json"
    write_canonical_json(record_path, record)
    assert load_json(record_path, MobileNetV2P43BEvaluationRecord) == record
