"""Deterministic synthetic data and FP32 pseudo-labels for a proxy pilot."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import NonNegativeInt, PositiveInt

from rvai.inference.preprocess import preprocess_image
from rvai.model_pipeline.calibration import (
    ModelPipelineDependencies,
    _image_input_spec,
    load_model_pipeline_dependencies,
)
from rvai.model_pipeline.dataset import ValidatedDataset
from rvai.model_pipeline.errors import ModelPipelineError
from rvai.model_pipeline.evaluate import (
    MobileNetV2P43BEvaluationArtifact,
    _classify_output,
    _require_runtime_contract,
    _verify_artifact,
)
from rvai.model_pipeline.io import (
    sha256_canonical_json,
    sha256_file,
    write_canonical_json,
)
from rvai.model_pipeline.schema import (
    Description,
    MobileNetV2P43BDatasetManifest,
    MobileNetV2P43BPipelineConfig,
    Sha256Digest,
    StrictModel,
)


class SyntheticDataError(ModelPipelineError):
    """Raised when proxy-pilot data cannot be generated reproducibly."""


class MobileNetV2P43BSyntheticGenerationRecord(StrictModel):
    """Portable identity of one deterministic synthetic dataset generation."""

    schema_version: Literal["1.0"] = "1.0"
    algorithm: Literal["rvai-rgb-patterns-v1"]
    seed: NonNegativeInt
    width: Literal[256]
    height: Literal[256]
    image_format: Literal["ppm-p6"]
    calibration_sample_count: PositiveInt
    evaluation_sample_count: PositiveInt
    calibration_manifest_sha256: Sha256Digest
    unlabeled_evaluation_manifest_sha256: Sha256Digest


class MobileNetV2P43BPseudoLabelRecord(StrictModel):
    """Evidence that evaluation labels came from the frozen FP32 artifact."""

    schema_version: Literal["1.0"] = "1.0"
    method: Literal["fp32-top1-pseudo-label"]
    source_model_sha256: Sha256Digest
    unlabeled_manifest_sha256: Sha256Digest
    evaluation_manifest_sha256: Sha256Digest
    sample_count: PositiveInt
    execution_provider: Literal["CPUExecutionProvider"]
    onnxruntime_version: Description


@dataclass(frozen=True)
class SyntheticDataset:
    """Runtime synthetic dataset paths and deterministic declarations."""

    root: Path
    calibration_manifest: MobileNetV2P43BDatasetManifest
    unlabeled_evaluation_manifest: MobileNetV2P43BDatasetManifest
    record: MobileNetV2P43BSyntheticGenerationRecord


@dataclass(frozen=True)
class PseudoLabeledDataset:
    """Runtime evaluation manifest and its pseudo-label evidence."""

    manifest: MobileNetV2P43BDatasetManifest
    record: MobileNetV2P43BPseudoLabelRecord


def generate_synthetic_dataset(
    destination: Path | str,
    *,
    calibration_sample_count: int,
    evaluation_sample_count: int,
    seed: int = 4302,
) -> SyntheticDataset:
    """Generate distinct byte-reproducible PPM images and calibration metadata."""

    _require_non_negative_integer(seed, "seed")
    _require_positive_integer(calibration_sample_count, "calibration sample count")
    _require_positive_integer(evaluation_sample_count, "evaluation sample count")
    root = Path(destination)
    try:
        root.mkdir()
        (root / "calibration").mkdir()
        (root / "evaluation").mkdir()
    except FileExistsError as exc:
        raise SyntheticDataError(
            f"Refusing to overwrite synthetic dataset directory: {root}"
        ) from exc
    except OSError as exc:
        raise SyntheticDataError(
            f"Cannot create synthetic dataset directory: {exc}"
        ) from exc

    calibration_samples = _generate_split(
        root,
        split="calibration",
        count=calibration_sample_count,
        start_index=0,
        seed=seed,
    )
    evaluation_samples = _generate_split(
        root,
        split="evaluation",
        count=evaluation_sample_count,
        start_index=calibration_sample_count,
        seed=seed,
    )
    calibration_manifest = _manifest(
        split="calibration",
        purpose="calibration",
        samples=calibration_samples,
        provenance="Deterministic RVAI synthetic RGB patterns for proxy calibration.",
    )
    unlabeled_evaluation_manifest = _manifest(
        split="evaluation-unlabeled",
        purpose="calibration",
        samples=evaluation_samples,
        provenance=(
            "Deterministic RVAI synthetic RGB patterns awaiting FP32 pseudo-labels."
        ),
    )
    write_canonical_json(root / "calibration-manifest.json", calibration_manifest)
    write_canonical_json(
        root / "evaluation-unlabeled-manifest.json",
        unlabeled_evaluation_manifest,
    )
    record = MobileNetV2P43BSyntheticGenerationRecord(
        algorithm="rvai-rgb-patterns-v1",
        seed=seed,
        width=256,
        height=256,
        image_format="ppm-p6",
        calibration_sample_count=calibration_sample_count,
        evaluation_sample_count=evaluation_sample_count,
        calibration_manifest_sha256=sha256_canonical_json(calibration_manifest),
        unlabeled_evaluation_manifest_sha256=sha256_canonical_json(
            unlabeled_evaluation_manifest
        ),
    )
    write_canonical_json(root / "generation.json", record)
    return SyntheticDataset(
        root=root.resolve(strict=True),
        calibration_manifest=calibration_manifest,
        unlabeled_evaluation_manifest=unlabeled_evaluation_manifest,
        record=record,
    )


def pseudo_label_evaluation_dataset(
    source_model_path: Path | str,
    *,
    artifact: MobileNetV2P43BEvaluationArtifact,
    pipeline: MobileNetV2P43BPipelineConfig,
    unlabeled_evaluation: ValidatedDataset,
    destination_manifest: Path | str,
    dependencies: ModelPipelineDependencies | None = None,
) -> PseudoLabeledDataset:
    """Assign FP32 Top-1 labels and publish the strict evaluation manifest."""

    if unlabeled_evaluation.manifest.dataset.split != "evaluation-unlabeled":
        raise SyntheticDataError(
            "Pseudo-label input must be the synthetic evaluation split"
        )
    if artifact.role != "fp32":
        raise SyntheticDataError("Pseudo-label generation requires the FP32 artifact")
    path = Path(source_model_path)
    _verify_artifact(path, artifact)
    modules = dependencies or load_model_pipeline_dependencies()
    try:
        session = modules.onnxruntime.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
    except Exception as exc:
        raise SyntheticDataError(f"Cannot open FP32 pseudo-label model: {exc}") from exc
    input_name, output_name = _require_runtime_contract(session)
    input_spec = _image_input_spec(pipeline.preprocessing)
    labels: list[int] = []
    for sample in unlabeled_evaluation.samples:
        try:
            tensor, _ = preprocess_image(
                sample.resolved_path,
                input_spec,
                numpy=modules.numpy,
                pillow_image=modules.pillow_image,
            )
            outputs = session.run([output_name], {input_name: tensor})
        except Exception as exc:
            raise SyntheticDataError(
                f"Cannot pseudo-label sample {sample.declaration.id}: {exc}"
            ) from exc
        outcome = _classify_output(sample, 0, outputs, modules.numpy)
        if not outcome.succeeded or outcome.top1_class is None:
            raise SyntheticDataError(
                f"Cannot pseudo-label sample {sample.declaration.id}: {outcome.failure}"
            )
        labels.append(outcome.top1_class)

    source = unlabeled_evaluation.manifest
    manifest = MobileNetV2P43BDatasetManifest(
        dataset=source.dataset.model_copy(
            update={
                "split": "evaluation-proxy",
                "purpose": "evaluation",
                "provenance": (
                    "Deterministic RVAI synthetic RGB patterns with frozen FP32 "
                    "Top-1 pseudo-labels; not ground-truth ImageNet accuracy data."
                ),
            }
        ),
        preprocessing=source.preprocessing,
        sample_order=source.sample_order,
        samples=tuple(
            sample.model_copy(update={"label": label})
            for sample, label in zip(source.samples, labels, strict=True)
        ),
    )
    record = MobileNetV2P43BPseudoLabelRecord(
        method="fp32-top1-pseudo-label",
        source_model_sha256=artifact.sha256,
        unlabeled_manifest_sha256=sha256_canonical_json(source),
        evaluation_manifest_sha256=sha256_canonical_json(manifest),
        sample_count=len(labels),
        execution_provider="CPUExecutionProvider",
        onnxruntime_version=str(modules.onnxruntime.__version__),
    )
    destination = Path(destination_manifest)
    write_canonical_json(destination, manifest)
    write_canonical_json(destination.with_name("pseudo-labels.json"), record)
    return PseudoLabeledDataset(manifest=manifest, record=record)


def _generate_split(
    root: Path,
    *,
    split: str,
    count: int,
    start_index: int,
    seed: int,
) -> tuple[dict[str, Any], ...]:
    declarations: list[dict[str, Any]] = []
    for offset in range(count):
        index = start_index + offset
        identifier = f"synthetic-{split}-{offset:04d}"
        relative_path = f"{split}/{identifier}.ppm"
        image_path = root / relative_path
        _write_ppm(image_path, index=index, seed=seed)
        declaration: dict[str, Any] = {
            "id": identifier,
            "path": relative_path,
            "sha256": sha256_file(image_path),
        }
        declarations.append(declaration)
    return tuple(declarations)


def _write_ppm(path: Path, *, index: int, seed: int) -> None:
    header = b"P6\n256 256\n255\n"
    pixels = bytearray(256 * 256 * 3)
    position = 0
    index_block = index >> 8
    for y in range(256):
        for x in range(256):
            tile = ((x // 16) + (y // 16) + index) & 1
            pixels[position] = (
                3 * x
                + y
                + 17 * index
                + seed
                + 47 * tile
                + 11 * index_block * (1 + x // 32)
            ) & 255
            pixels[position + 1] = (
                x
                + 5 * y
                + 29 * index
                + 31 * (x // 16)
                + seed // 3
                + 23 * index_block * (1 + y // 32)
            ) & 255
            pixels[position + 2] = (
                (x ^ y)
                + 43 * index
                + 19 * (y // 16)
                + seed // 7
                + 37 * index_block * (1 + (x + y) // 64)
            ) & 255
            position += 3
    try:
        with path.open("xb") as handle:
            handle.write(header)
            handle.write(pixels)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise SyntheticDataError(f"Cannot write synthetic image: {exc}") from exc


def _manifest(
    *,
    split: str,
    purpose: Literal["calibration", "evaluation"],
    samples: tuple[dict[str, Any], ...],
    provenance: str,
) -> MobileNetV2P43BDatasetManifest:
    return MobileNetV2P43BDatasetManifest.model_validate(
        {
            "schema_version": "1.0",
            "dataset": {
                "name": "rvai-synthetic-proxy",
                "version": "v1",
                "split": split,
                "purpose": purpose,
                "provenance": provenance,
                "license": "Generated by RVAI; no third-party dataset content.",
            },
            "preprocessing": "mobilenet-v2-imagenet-v1",
            "sample_order": "manifest",
            "samples": samples,
        }
    )


def _require_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SyntheticDataError(f"{name} must be a positive integer")


def _require_non_negative_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SyntheticDataError(f"{name} must be a non-negative integer")
