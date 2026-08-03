"""Deterministic source-ONNX inspection for MobileNetV2 P4.3B."""

from __future__ import annotations

import importlib
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

from pydantic import NonNegativeInt, PositiveInt, field_validator

from rvai.model_pipeline.errors import ModelPipelineError, PipelineIOError
from rvai.model_pipeline.io import sha256_file
from rvai.model_pipeline.schema import (
    Description,
    Identifier,
    MobileNetV2P43BSourceModelIdentity,
    StrictModel,
)


class SourceInspectionError(ModelPipelineError):
    """Raised before calibration when source identity or ONNX contract fails."""


OptionalMetadata = Description | None
TensorDimension = NonNegativeInt | Description | None


class MobileNetV2P43BOpsetImport(StrictModel):
    """One normalized ONNX opset import."""

    domain: Identifier
    version: NonNegativeInt


class MobileNetV2P43BTensorContract(StrictModel):
    """One deterministic ONNX graph input or output declaration."""

    name: Description
    dtype: Identifier
    shape: tuple[TensorDimension, ...]

    @field_validator("shape", mode="before")
    @classmethod
    def shape_list_to_tuple(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value


class MobileNetV2P43BOperatorSummary(StrictModel):
    """Count of one normalized ONNX operator type and domain."""

    domain: Identifier
    op_type: Identifier
    count: PositiveInt


class MobileNetV2P43BSourceInspectionRecord(StrictModel):
    """Successful source identity, checker, graph, and contract evidence."""

    schema_version: Literal["1.0"] = "1.0"
    model: MobileNetV2P43BSourceModelIdentity
    checker_passed: Literal[True]
    contract: Literal["mobilenet-v2-fp32-classification-v1"]
    contract_matched: Literal[True]
    ir_version: NonNegativeInt
    opset_imports: tuple[MobileNetV2P43BOpsetImport, ...]
    producer_name: OptionalMetadata
    producer_version: OptionalMetadata
    inputs: tuple[MobileNetV2P43BTensorContract, ...]
    outputs: tuple[MobileNetV2P43BTensorContract, ...]
    node_count: NonNegativeInt
    initializer_count: NonNegativeInt
    operators: tuple[MobileNetV2P43BOperatorSummary, ...]

    @field_validator("opset_imports", "inputs", "outputs", "operators", mode="before")
    @classmethod
    def record_lists_to_tuples(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value


_DTYPES = {
    1: "float32",
    2: "uint8",
    3: "int8",
    4: "uint16",
    5: "int16",
    6: "int32",
    7: "int64",
    8: "string",
    9: "bool",
    10: "float16",
    11: "float64",
    12: "uint32",
    13: "uint64",
    14: "complex64",
    15: "complex128",
    16: "bfloat16",
}


def inspect_source_model(
    model_path: Path | str,
    expected: MobileNetV2P43BSourceModelIdentity,
    *,
    onnx_module: ModuleType | Any | None = None,
) -> MobileNetV2P43BSourceInspectionRecord:
    """Verify identity first, then inspect and enforce the frozen ONNX contract."""

    path = Path(model_path)
    _verify_identity(path, expected)
    onnx = onnx_module or _load_onnx()
    try:
        model = onnx.load_model(str(path), load_external_data=False)
    except Exception as exc:
        raise SourceInspectionError(f"Cannot open source ONNX model: {exc}") from exc
    try:
        onnx.checker.check_model(model)
    except Exception as exc:
        raise SourceInspectionError(f"ONNX checker rejected source model: {exc}") from exc

    inputs = _runtime_inputs(model)
    outputs = tuple(_tensor_contract(value) for value in model.graph.output)
    _require_mobilenet_v2_contract(inputs, outputs)

    opsets = tuple(
        sorted(
            (
                MobileNetV2P43BOpsetImport(
                    domain=_domain(item.domain),
                    version=item.version,
                )
                for item in model.opset_import
            ),
            key=lambda item: (item.domain, item.version),
        )
    )
    operator_counts = Counter(
        (_domain(node.domain), node.op_type) for node in model.graph.node
    )
    operators = tuple(
        MobileNetV2P43BOperatorSummary(
            domain=domain,
            op_type=op_type,
            count=count,
        )
        for (domain, op_type), count in sorted(operator_counts.items())
    )
    return MobileNetV2P43BSourceInspectionRecord(
        model=expected,
        checker_passed=True,
        contract="mobilenet-v2-fp32-classification-v1",
        contract_matched=True,
        ir_version=model.ir_version,
        opset_imports=opsets,
        producer_name=_optional_metadata(model.producer_name),
        producer_version=_optional_metadata(model.producer_version),
        inputs=inputs,
        outputs=outputs,
        node_count=len(model.graph.node),
        initializer_count=len(model.graph.initializer),
        operators=operators,
    )


def _verify_identity(
    path: Path,
    expected: MobileNetV2P43BSourceModelIdentity,
) -> None:
    if path.name != expected.filename:
        raise SourceInspectionError(
            f"Source filename mismatch: expected {expected.filename}, got {path.name}"
        )
    try:
        stat_result = path.stat()
    except OSError as exc:
        detail = exc.strerror or str(exc)
        raise SourceInspectionError(f"Cannot stat source model: {detail}") from exc
    if not path.is_file():
        raise SourceInspectionError("Source model must be a regular file")
    if stat_result.st_size != expected.size_bytes:
        raise SourceInspectionError(
            "Source size mismatch: "
            f"expected {expected.size_bytes}, got {stat_result.st_size}"
        )
    try:
        observed_sha256 = sha256_file(path)
    except PipelineIOError as exc:
        raise SourceInspectionError(str(exc)) from exc
    if observed_sha256 != expected.sha256:
        raise SourceInspectionError(
            f"Source SHA-256 mismatch: expected {expected.sha256}, got {observed_sha256}"
        )


def _load_onnx() -> ModuleType:
    try:
        return importlib.import_module("onnx")
    except ImportError as exc:
        raise SourceInspectionError(
            "ONNX inspection requires the model-pipeline development dependencies"
        ) from exc


def _runtime_inputs(model: Any) -> tuple[MobileNetV2P43BTensorContract, ...]:
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    return tuple(
        _tensor_contract(value)
        for value in model.graph.input
        if value.name not in initializer_names
    )


def _tensor_contract(value: Any) -> MobileNetV2P43BTensorContract:
    if not value.type.HasField("tensor_type"):
        raise SourceInspectionError(f"ONNX value {value.name!r} is not a tensor")
    tensor_type = value.type.tensor_type
    dtype = _DTYPES.get(tensor_type.elem_type, f"tensor-type-{tensor_type.elem_type}")
    shape: list[int | str | None] = []
    for dimension in tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            shape.append(dimension.dim_value)
        elif dimension.HasField("dim_param"):
            shape.append(dimension.dim_param)
        else:
            shape.append(None)
    return MobileNetV2P43BTensorContract(
        name=value.name,
        dtype=dtype,
        shape=tuple(shape),
    )


def _require_mobilenet_v2_contract(
    inputs: tuple[MobileNetV2P43BTensorContract, ...],
    outputs: tuple[MobileNetV2P43BTensorContract, ...],
) -> None:
    if len(inputs) != 1:
        raise SourceInspectionError(
            f"MobileNetV2 requires exactly one runtime input, got {len(inputs)}"
        )
    input_shape = inputs[0].shape
    if (
        inputs[0].dtype != "float32"
        or len(input_shape) != 4
        or input_shape[1:] != (3, 224, 224)
    ):
        raise SourceInspectionError(
            "MobileNetV2 input contract mismatch: expected float32 "
            "[batch, 3, 224, 224], "
            f"got {inputs[0].dtype} {list(inputs[0].shape)}"
        )
    if len(outputs) != 1:
        raise SourceInspectionError(
            f"MobileNetV2 requires exactly one classification output, got {len(outputs)}"
        )
    output_shape = outputs[0].shape
    if (
        outputs[0].dtype != "float32"
        or len(output_shape) != 2
        or output_shape[1:] != (1000,)
    ):
        raise SourceInspectionError(
            "MobileNetV2 output contract mismatch: expected float32 [batch, 1000], "
            f"got {outputs[0].dtype} {list(outputs[0].shape)}"
        )
    if not _batch_dimensions_compatible(input_shape[0], output_shape[0]):
        raise SourceInspectionError(
            "MobileNetV2 batch contract mismatch: expected input and output batch "
            "dimensions to both be 1 or the same non-empty symbol, got "
            f"{input_shape[0]!r} and {output_shape[0]!r}"
        )


def _batch_dimensions_compatible(input_batch: Any, output_batch: Any) -> bool:
    """Accept literal batch one or one shared non-empty symbolic batch."""

    if input_batch == 1 and output_batch == 1:
        return True
    return (
        isinstance(input_batch, str)
        and bool(input_batch)
        and input_batch == input_batch.strip()
        and input_batch == output_batch
    )


def _domain(value: str) -> str:
    return value or "ai.onnx"


def _optional_metadata(value: str) -> str | None:
    return value or None
