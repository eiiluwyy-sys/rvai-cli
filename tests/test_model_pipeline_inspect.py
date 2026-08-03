import hashlib
from pathlib import Path
import shutil

import pytest

from rvai.model_pipeline.inspect import SourceInspectionError, inspect_source_model
from rvai.model_pipeline.io import canonical_json_bytes
from rvai.model_pipeline.schema import MobileNetV2P43BSourceModelIdentity


EXISTING_TINY_MODEL = Path(__file__).parent / "fixtures" / "onnx" / "tiny-classifier.onnx"


def expected_identity(path: Path, *, sha256: str | None = None, size: int | None = None):
    contents = path.read_bytes()
    return MobileNetV2P43BSourceModelIdentity(
        name="mobilenetv2-12",
        format="onnx",
        precision="fp32",
        filename="mobilenetv2-12.onnx",
        size_bytes=len(contents) if size is None else size,
        sha256=sha256 or hashlib.sha256(contents).hexdigest(),
    )


def create_model(
    path: Path,
    *,
    input_shape: tuple[int, ...] = (1, 3, 224, 224),
    output_shape: tuple[int, ...] = (1, 1000),
    valid_graph: bool = True,
):
    onnx = pytest.importorskip("onnx")
    image = onnx.helper.make_tensor_value_info(
        "image", onnx.TensorProto.FLOAT, list(input_shape)
    )
    logits = onnx.helper.make_tensor_value_info(
        "logits", onnx.TensorProto.FLOAT, list(output_shape)
    )
    nodes = []
    if valid_graph:
        values = [0.0] * _shape_size(output_shape)
        constant = onnx.helper.make_tensor(
            "constant_logits",
            onnx.TensorProto.FLOAT,
            list(output_shape),
            values,
        )
        nodes.append(onnx.helper.make_node("Constant", [], ["logits"], value=constant))
    graph = onnx.helper.make_graph(nodes, "tiny-mobilenet-contract", [image], [logits])
    model = onnx.helper.make_model(
        graph,
        producer_name="rvai-tests",
        producer_version="1.0",
        opset_imports=[onnx.helper.make_opsetid("", 13)],
    )
    model.ir_version = 8
    onnx.save_model(model, path)
    return onnx


def _shape_size(shape: tuple[int, ...]) -> int:
    size = 1
    for dimension in shape:
        size *= dimension
    return size


def create_symbolic_batch_model(
    path: Path,
    *,
    input_batch: str = "batch_size",
    output_batch: str = "batch_size",
):
    onnx = pytest.importorskip("onnx")
    image = onnx.helper.make_tensor_value_info(
        "image", onnx.TensorProto.FLOAT, [input_batch, 3, 224, 224]
    )
    logits = onnx.helper.make_tensor_value_info(
        "logits", onnx.TensorProto.FLOAT, [output_batch, 1000]
    )
    weights = onnx.helper.make_tensor(
        "weights", onnx.TensorProto.FLOAT, [3, 1000], [0.01] * 3000
    )
    nodes = [
        onnx.helper.make_node("GlobalAveragePool", ["image"], ["pooled"]),
        onnx.helper.make_node("Flatten", ["pooled"], ["features"], axis=1),
        onnx.helper.make_node("MatMul", ["features", "weights"], ["logits"]),
    ]
    graph = onnx.helper.make_graph(
        nodes,
        "symbolic-batch-mobilenet-contract",
        [image],
        [logits],
        [weights],
    )
    model = onnx.helper.make_model(
        graph,
        opset_imports=[onnx.helper.make_opsetid("", 13)],
    )
    model.ir_version = 8
    onnx.save_model(model, path)
    return onnx


def test_inspection_records_checked_graph_and_frozen_contract(tmp_path: Path) -> None:
    path = tmp_path / "mobilenetv2-12.onnx"
    onnx = create_model(path)

    record = inspect_source_model(path, expected_identity(path), onnx_module=onnx)

    assert record.checker_passed is True
    assert record.contract_matched is True
    assert record.ir_version == 8
    assert [(item.domain, item.version) for item in record.opset_imports] == [
        ("ai.onnx", 13)
    ]
    assert record.producer_name == "rvai-tests"
    assert record.producer_version == "1.0"
    assert record.inputs[0].name == "image"
    assert record.inputs[0].dtype == "float32"
    assert record.inputs[0].shape == (1, 3, 224, 224)
    assert record.outputs[0].shape == (1, 1000)
    assert record.node_count == 1
    assert record.initializer_count == 0
    assert [(item.domain, item.op_type, item.count) for item in record.operators] == [
        ("ai.onnx", "Constant", 1)
    ]
    assert canonical_json_bytes(record) == canonical_json_bytes(record)
    assert type(record).model_validate_json(canonical_json_bytes(record)) == record


def test_inspection_accepts_shared_symbolic_batch_dimension(tmp_path: Path) -> None:
    path = tmp_path / "mobilenetv2-12.onnx"
    onnx = create_symbolic_batch_model(path)

    record = inspect_source_model(path, expected_identity(path), onnx_module=onnx)

    assert record.inputs[0].shape == ("batch_size", 3, 224, 224)
    assert record.outputs[0].shape == ("batch_size", 1000)


def test_inspection_rejects_different_symbolic_batch_dimensions(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mobilenetv2-12.onnx"
    onnx = create_symbolic_batch_model(
        path,
        input_batch="input_batch",
        output_batch="output_batch",
    )

    with pytest.raises(SourceInspectionError, match="batch contract mismatch"):
        inspect_source_model(path, expected_identity(path), onnx_module=onnx)


@pytest.mark.parametrize("mismatch", ["filename", "size", "sha256"])
def test_identity_mismatch_fails_before_onnx_is_opened(
    mismatch: str, tmp_path: Path
) -> None:
    path = tmp_path / "mobilenetv2-12.onnx"
    path.write_bytes(b"not-an-onnx-model")
    expected = expected_identity(path)
    if mismatch == "filename":
        inspected_path = tmp_path / "different.onnx"
        inspected_path.write_bytes(path.read_bytes())
    else:
        inspected_path = path
        values = expected.model_dump()
        if mismatch == "size":
            values["size_bytes"] += 1
        else:
            values["sha256"] = "0" * 64
        expected = MobileNetV2P43BSourceModelIdentity.model_validate(values)

    class UnexpectedOnnx:
        def load_model(self, *args, **kwargs):
            raise AssertionError("ONNX must not be opened after identity mismatch")

    with pytest.raises(SourceInspectionError, match=mismatch.replace("sha256", "SHA-256")):
        inspect_source_model(inspected_path, expected, onnx_module=UnexpectedOnnx())


def test_onnx_checker_failure_stops_inspection(tmp_path: Path) -> None:
    path = tmp_path / "mobilenetv2-12.onnx"
    onnx = create_model(path, valid_graph=False)

    with pytest.raises(SourceInspectionError, match="checker rejected"):
        inspect_source_model(path, expected_identity(path), onnx_module=onnx)


def test_contract_mismatch_stops_inspection(tmp_path: Path) -> None:
    path = tmp_path / "mobilenetv2-12.onnx"
    onnx = create_model(path, input_shape=(1, 3, 2, 2))

    with pytest.raises(SourceInspectionError, match="input contract mismatch"):
        inspect_source_model(path, expected_identity(path), onnx_module=onnx)


def test_existing_committed_tiny_model_is_checked_but_rejected_by_contract(
    tmp_path: Path,
) -> None:
    onnx = pytest.importorskip("onnx")
    path = tmp_path / "mobilenetv2-12.onnx"
    shutil.copy(EXISTING_TINY_MODEL, path)

    with pytest.raises(SourceInspectionError, match="input contract mismatch"):
        inspect_source_model(path, expected_identity(path), onnx_module=onnx)
