"""Deterministic reproducibility evidence for MobileNetV2 P4.3B."""

from __future__ import annotations

import os
import platform
import re
import subprocess
from pathlib import Path
from typing import Annotated, Literal

import rvai
from pydantic import Field, PositiveInt

from rvai.model_pipeline.calibration import ModelPipelineDependencies
from rvai.model_pipeline.errors import ModelPipelineError
from rvai.model_pipeline.schema import Description, Sha256Digest, StrictModel


class EnvironmentCaptureError(ModelPipelineError):
    """Raised when complete deterministic environment evidence is unavailable."""


GitCommitDigest = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class MobileNetV2P43BSoftwareEnvironment(StrictModel):
    """Exact software versions that can affect pipeline output."""

    python_version: Description
    rvai_version: Description
    onnx_version: Description
    onnxruntime_version: Description
    numpy_version: Description
    pillow_version: Description


class MobileNetV2P43BExecutionEnvironment(StrictModel):
    """Non-identifying execution-platform metadata."""

    execution_provider: Literal["CPUExecutionProvider"]
    platform_system: Description
    platform_release: Description
    architecture: Description
    cpu_description: Description
    logical_core_count: PositiveInt


class MobileNetV2P43BSourceRevision(StrictModel):
    """Git revision and complete tracked/untracked worktree state."""

    vcs: Literal["git"]
    commit: GitCommitDigest
    working_tree_clean: bool = Field(
        description=(
            "A later formal package builder must reject false because an uncommitted "
            "source state is not independently reproducible."
        )
    )


class MobileNetV2P43BPipelineInputDigests(StrictModel):
    """Canonical identities of every declared or generated pipeline input."""

    pipeline_config_sha256: Sha256Digest
    source_model_config_sha256: Sha256Digest
    source_fp32_model_sha256: Sha256Digest
    calibration_manifest_sha256: Sha256Digest
    unlabeled_evaluation_manifest_sha256: Sha256Digest
    evaluation_manifest_sha256: Sha256Digest


class MobileNetV2P43BPipelineOutputDigests(StrictModel):
    """Canonical identities of all B1-through-B3 output evidence."""

    source_inspection_sha256: Sha256Digest
    generation_record_sha256: Sha256Digest
    pseudo_label_record_sha256: Sha256Digest
    calibration_validation_sha256: Sha256Digest
    evaluation_validation_sha256: Sha256Digest
    overlap_report_sha256: Sha256Digest
    quantization_record_sha256: Sha256Digest
    int8_model_sha256: Sha256Digest
    fp32_evaluation_sha256: Sha256Digest
    int8_evaluation_sha256: Sha256Digest
    comparison_sha256: Sha256Digest
    proxy_pilot_report_sha256: Sha256Digest


class MobileNetV2P43BReproducibilityRecord(StrictModel):
    """Environment-sensitive evidence, stable in the same declared environment."""

    schema_version: Literal["1.0"] = "1.0"
    pipeline: Literal["mobilenet-v2-int8"]
    report_type: Literal["synthetic-consistency-pilot"]
    status: Literal["provisional"]
    production_verified: Literal[False]
    label_source: Literal["fp32-top1-pseudo-label"]
    software: MobileNetV2P43BSoftwareEnvironment
    execution: MobileNetV2P43BExecutionEnvironment
    source_revision: MobileNetV2P43BSourceRevision
    inputs: MobileNetV2P43BPipelineInputDigests
    outputs: MobileNetV2P43BPipelineOutputDigests


def collect_software_environment(
    dependencies: ModelPipelineDependencies,
) -> MobileNetV2P43BSoftwareEnvironment:
    """Collect exact dependency versions without importing optional packages anew."""

    return MobileNetV2P43BSoftwareEnvironment(
        python_version=platform.python_version(),
        rvai_version=rvai.__version__,
        onnx_version=_module_version(dependencies.onnx, "ONNX"),
        onnxruntime_version=_module_version(
            dependencies.onnxruntime,
            "ONNX Runtime",
        ),
        numpy_version=_module_version(dependencies.numpy, "NumPy"),
        pillow_version=_module_version(dependencies.pillow_image, "Pillow"),
    )


def collect_execution_environment() -> MobileNetV2P43BExecutionEnvironment:
    """Collect platform data while excluding host and user identity."""

    architecture = platform.machine()
    cpu_description = platform.processor() or architecture
    logical_cores = os.cpu_count()
    if logical_cores is None or logical_cores <= 0:
        raise EnvironmentCaptureError("Cannot determine a positive logical-core count")
    return MobileNetV2P43BExecutionEnvironment(
        execution_provider="CPUExecutionProvider",
        platform_system=platform.system(),
        platform_release=platform.release(),
        architecture=architecture,
        cpu_description=cpu_description,
        logical_core_count=logical_cores,
    )


def collect_source_revision(
    repository: Path | str,
) -> MobileNetV2P43BSourceRevision:
    """Collect HEAD and cleanliness, including tracked and untracked files."""

    root = Path(repository)
    commit = _git_output(root, "rev-parse", "--verify", "HEAD")
    if _GIT_COMMIT.fullmatch(commit) is None:
        raise EnvironmentCaptureError(
            "Git HEAD must be a 40-character lowercase commit digest"
        )
    status = _git_output(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    return MobileNetV2P43BSourceRevision(
        vcs="git",
        commit=commit,
        working_tree_clean=not bool(status),
    )


def capture_reproducibility_record(
    *,
    dependencies: ModelPipelineDependencies,
    inputs: MobileNetV2P43BPipelineInputDigests,
    outputs: MobileNetV2P43BPipelineOutputDigests,
    repository: Path | str | None = None,
) -> MobileNetV2P43BReproducibilityRecord:
    """Capture one complete provisional reproducibility record."""

    source_repository = (
        Path(__file__).resolve().parents[3] if repository is None else Path(repository)
    )
    return MobileNetV2P43BReproducibilityRecord(
        pipeline="mobilenet-v2-int8",
        report_type="synthetic-consistency-pilot",
        status="provisional",
        production_verified=False,
        label_source="fp32-top1-pseudo-label",
        software=collect_software_environment(dependencies),
        execution=collect_execution_environment(),
        source_revision=collect_source_revision(source_repository),
        inputs=inputs,
        outputs=outputs,
    )


def _module_version(module: object, name: str) -> str:
    value = getattr(module, "__version__", None)
    if not isinstance(value, str) or not value:
        raise EnvironmentCaptureError(f"Cannot determine the {name} version")
    return value


def _git_output(repository: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ("git", "-C", str(repository), *arguments),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise EnvironmentCaptureError(
            "Cannot capture source revision from a Git repository"
        ) from exc
    return completed.stdout.strip()
