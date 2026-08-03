"""Static QDQ quantization for the frozen MobileNetV2 P4.3B configuration."""

from __future__ import annotations

import inspect as python_inspect
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import NonNegativeInt, PositiveInt, model_validator

from rvai.model_pipeline.calibration import (
    CalibrationSelection,
    ManifestCalibrationDataReader,
    ModelPipelineDependencies,
    load_model_pipeline_dependencies,
)
from rvai.model_pipeline.errors import ModelPipelineError, PipelineIOError
from rvai.model_pipeline.inspect import (
    MobileNetV2P43BSourceInspectionRecord,
    SourceInspectionError,
    _require_mobilenet_v2_contract,
    _runtime_inputs,
    _tensor_contract,
)
from rvai.model_pipeline.io import sha256_canonical_json, sha256_file
from rvai.model_pipeline.schema import (
    Description,
    Identifier,
    MobileNetV2P43BPipelineConfig,
    MobileNetV2P43BQuantizationConfig,
    PlainFilename,
    Sha256Digest,
    StrictModel,
)


class QuantizationError(ModelPipelineError):
    """Raised when static quantization cannot produce verified QDQ output."""


class MobileNetV2P43BQuantizedArtifact(StrictModel):
    """Portable identity of the generated INT8 ONNX file."""

    filename: PlainFilename
    size_bytes: PositiveInt
    sha256: Sha256Digest


class MobileNetV2P43BQuantizedStructure(StrictModel):
    """Structural evidence that the generated model uses QDQ representation."""

    node_count: PositiveInt
    initializer_count: NonNegativeInt
    quantize_linear_count: PositiveInt
    dequantize_linear_count: PositiveInt


class MobileNetV2P43BQuantizationRecord(StrictModel):
    """Deterministic evidence for one successful frozen-config quantization."""

    schema_version: Literal["1.0"] = "1.0"
    source_model_sha256: Sha256Digest
    source_inspection_sha256: Sha256Digest
    pipeline_config_sha256: Sha256Digest
    calibration_manifest_sha256: Sha256Digest
    calibration_sample_count: PositiveInt
    calibration_sample_ids: tuple[Identifier, ...]
    quantization: MobileNetV2P43BQuantizationConfig
    execution_provider: Literal["CPUExecutionProvider"]
    onnxruntime_version: Description
    checker_passed: Literal[True]
    contract_matched: Literal[True]
    structure: MobileNetV2P43BQuantizedStructure
    artifact: MobileNetV2P43BQuantizedArtifact

    @model_validator(mode="after")
    def count_matches_sample_ids(self) -> "MobileNetV2P43BQuantizationRecord":
        if self.calibration_sample_count != len(self.calibration_sample_ids):
            raise ValueError(
                "calibration_sample_count must equal the number of sample IDs"
            )
        return self


def quantize_static_qdq(
    source_path: Path | str,
    destination_path: Path | str,
    *,
    source_inspection: MobileNetV2P43BSourceInspectionRecord,
    pipeline: MobileNetV2P43BPipelineConfig,
    calibration: CalibrationSelection,
    dependencies: ModelPipelineDependencies | None = None,
    quantize_static_fn: Callable[..., Any] | None = None,
) -> MobileNetV2P43BQuantizationRecord:
    """Quantize into a verified temporary model, then atomically publish it."""

    source = Path(source_path)
    destination = Path(destination_path)
    _verify_source_unchanged(source, source_inspection)
    _verify_calibration_matches(calibration)
    if destination.exists():
        raise QuantizationError(f"Refusing to overwrite existing file: {destination}")
    if not destination.parent.is_dir():
        raise QuantizationError(
            f"Destination directory does not exist: {destination.parent}"
        )

    modules = dependencies or load_model_pipeline_dependencies()
    quantize = quantize_static_fn or modules.quantization.quantize_static
    _require_calibration_provider_support(quantize)
    reader = ManifestCalibrationDataReader(
        calibration,
        input_name=source_inspection.inputs[0].name,
        preprocessing=pipeline.preprocessing,
        dependencies=modules,
    )

    temporary: Path | None = None
    published = False
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".onnx",
            dir=destination.parent,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        quantize(
            model_input=str(source),
            model_output=str(temporary),
            calibration_data_reader=reader,
            quant_format=modules.quantization.QuantFormat.QDQ,
            per_channel=True,
            activation_type=modules.quantization.QuantType.QUInt8,
            weight_type=modules.quantization.QuantType.QInt8,
            calibrate_method=modules.quantization.CalibrationMethod.MinMax,
            calibration_providers=["CPUExecutionProvider"],
            use_external_data_format=False,
        )
        model = modules.onnx.load_model(str(temporary), load_external_data=False)
        modules.onnx.checker.check_model(model)
        inputs = _runtime_inputs(model)
        outputs = tuple(_tensor_contract(value) for value in model.graph.output)
        _require_mobilenet_v2_contract(inputs, outputs)
        operator_counts = _operator_counts(model)
        quantize_count = operator_counts.get("QuantizeLinear", 0)
        dequantize_count = operator_counts.get("DequantizeLinear", 0)
        if quantize_count <= 0 or dequantize_count <= 0:
            raise QuantizationError(
                "Quantized model does not contain required QDQ operators"
            )
        artifact = MobileNetV2P43BQuantizedArtifact(
            filename=destination.name,
            size_bytes=temporary.stat().st_size,
            sha256=sha256_file(temporary),
        )
        record = MobileNetV2P43BQuantizationRecord(
            source_model_sha256=source_inspection.model.sha256,
            source_inspection_sha256=sha256_canonical_json(source_inspection),
            pipeline_config_sha256=sha256_canonical_json(pipeline),
            calibration_manifest_sha256=calibration.record.manifest_sha256,
            calibration_sample_count=calibration.record.sample_count,
            calibration_sample_ids=calibration.record.sample_ids,
            quantization=pipeline.quantization,
            execution_provider="CPUExecutionProvider",
            onnxruntime_version=str(modules.onnxruntime.__version__),
            checker_passed=True,
            contract_matched=True,
            structure=MobileNetV2P43BQuantizedStructure(
                node_count=len(model.graph.node),
                initializer_count=len(model.graph.initializer),
                quantize_linear_count=quantize_count,
                dequantize_linear_count=dequantize_count,
            ),
            artifact=artifact,
        )
        _fsync_file(temporary)
        try:
            os.link(temporary, destination)
            published = True
        except FileExistsError as exc:
            raise QuantizationError(
                f"Refusing to overwrite existing file: {destination}"
            ) from exc
        temporary.unlink()
        temporary = None
        _fsync_directory(destination.parent)
        return record
    except QuantizationError:
        raise
    except (OSError, PipelineIOError, SourceInspectionError) as exc:
        state = "published but not fully synchronized" if published else "not published"
        raise QuantizationError(f"Static QDQ quantization failed ({state}): {exc}") from exc
    except Exception as exc:
        raise QuantizationError(f"Static QDQ quantization failed: {exc}") from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass


def _verify_source_unchanged(
    source: Path,
    inspection: MobileNetV2P43BSourceInspectionRecord,
) -> None:
    if source.name != inspection.model.filename:
        raise QuantizationError("Source filename changed after inspection")
    try:
        size = source.stat().st_size
        digest = sha256_file(source)
    except (OSError, PipelineIOError) as exc:
        raise QuantizationError(f"Cannot reverify inspected source model: {exc}") from exc
    if size != inspection.model.size_bytes or digest != inspection.model.sha256:
        raise QuantizationError("Source model changed after inspection")


def _verify_calibration_matches(calibration: CalibrationSelection) -> None:
    if calibration.dataset.manifest.dataset.purpose != "calibration":
        raise QuantizationError("Quantization requires calibration-purpose data")
    if calibration.record.manifest_sha256 != calibration.dataset.record.manifest_sha256:
        raise QuantizationError("Calibration selection does not match its dataset record")
    selected_ids = tuple(sample.declaration.id for sample in calibration.samples)
    if (
        calibration.record.sample_count != len(calibration.samples)
        or calibration.record.sample_ids != selected_ids
    ):
        raise QuantizationError("Calibration samples do not match the selection record")


def _require_calibration_provider_support(quantize: Callable[..., Any]) -> None:
    parameters = python_inspect.signature(quantize).parameters.values()
    if not any(
        parameter.name == "calibration_providers"
        or parameter.kind is python_inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    ):
        raise QuantizationError(
            "Installed ONNX Runtime cannot explicitly select CPU calibration provider"
        )


def _operator_counts(model: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return counts


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
