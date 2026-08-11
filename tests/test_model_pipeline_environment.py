from pathlib import Path
from types import SimpleNamespace
import subprocess

import pytest
from pydantic import ValidationError

import rvai
from rvai.model_pipeline import environment
from rvai.model_pipeline.environment import (
    EnvironmentCaptureError,
    MobileNetV2P43BExecutionEnvironment,
    MobileNetV2P43BPipelineInputDigests,
    MobileNetV2P43BPipelineOutputDigests,
    MobileNetV2P43BReproducibilityRecord,
    MobileNetV2P43BSoftwareEnvironment,
    MobileNetV2P43BSourceRevision,
    collect_execution_environment,
    collect_software_environment,
    collect_source_revision,
)
from rvai.model_pipeline.errors import PipelineIOError
from rvai.model_pipeline.io import canonical_json_bytes, canonical_json_text
from rvai.model_pipeline.io import write_canonical_json


DIGEST = "a" * 64


def dependencies():
    return SimpleNamespace(
        numpy=SimpleNamespace(__version__="2.2.6"),
        onnx=SimpleNamespace(__version__="1.22.0"),
        onnxruntime=SimpleNamespace(__version__="1.23.2"),
        pillow_image=SimpleNamespace(__version__="12.3.0"),
    )


def input_digests() -> MobileNetV2P43BPipelineInputDigests:
    return MobileNetV2P43BPipelineInputDigests(
        pipeline_config_sha256=DIGEST,
        source_model_config_sha256="b" * 64,
        source_fp32_model_sha256="c" * 64,
        calibration_manifest_sha256="d" * 64,
        unlabeled_evaluation_manifest_sha256="e" * 64,
        evaluation_manifest_sha256="f" * 64,
    )


def output_digests() -> MobileNetV2P43BPipelineOutputDigests:
    values = {
        name: f"{index:x}" * 64
        for index, name in enumerate(
            MobileNetV2P43BPipelineOutputDigests.model_fields,
            start=1,
        )
    }
    return MobileNetV2P43BPipelineOutputDigests(**values)


def reproducibility_record() -> MobileNetV2P43BReproducibilityRecord:
    return MobileNetV2P43BReproducibilityRecord(
        pipeline="mobilenet-v2-int8",
        report_type="synthetic-consistency-pilot",
        status="provisional",
        production_verified=False,
        label_source="fp32-top1-pseudo-label",
        software=MobileNetV2P43BSoftwareEnvironment(
            python_version="3.10.20",
            rvai_version="0.1.0",
            onnx_version="1.22.0",
            onnxruntime_version="1.23.2",
            numpy_version="2.2.6",
            pillow_version="12.3.0",
        ),
        execution=MobileNetV2P43BExecutionEnvironment(
            execution_provider="CPUExecutionProvider",
            platform_system="Linux",
            platform_release="test-release",
            architecture="riscv64",
            cpu_description="Test CPU",
            logical_core_count=8,
        ),
        source_revision=MobileNetV2P43BSourceRevision(
            vcs="git",
            commit="1" * 40,
            working_tree_clean=True,
        ),
        inputs=input_digests(),
        outputs=output_digests(),
    )


def initialize_repository(path: Path) -> None:
    subprocess.run(("git", "init", "-q", str(path)), check=True)
    (path / "tracked.txt").write_text("original\n", encoding="utf-8")
    subprocess.run(("git", "-C", str(path), "add", "tracked.txt"), check=True)
    subprocess.run(
        (
            "git",
            "-C",
            str(path),
            "-c",
            "user.name=RVAI Tests",
            "-c",
            "user.email=rvai@example.invalid",
            "commit",
            "-q",
            "-m",
            "fixture",
        ),
        check=True,
    )


def test_collects_exact_software_versions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(environment.platform, "python_version", lambda: "3.10.20")
    monkeypatch.setattr(rvai, "__version__", "0.1.7")

    record = collect_software_environment(dependencies())

    assert record.model_dump() == {
        "python_version": "3.10.20",
        "rvai_version": "0.1.7",
        "onnx_version": "1.22.0",
        "onnxruntime_version": "1.23.2",
        "numpy_version": "2.2.6",
        "pillow_version": "12.3.0",
    }


def test_collects_exact_non_identifying_platform_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(environment.platform, "system", lambda: "Linux")
    monkeypatch.setattr(environment.platform, "release", lambda: "6.8-test")
    monkeypatch.setattr(environment.platform, "machine", lambda: "riscv64")
    monkeypatch.setattr(environment.platform, "processor", lambda: "RV64 Test CPU")
    monkeypatch.setattr(environment.os, "cpu_count", lambda: 12)

    record = collect_execution_environment()

    assert record.model_dump() == {
        "execution_provider": "CPUExecutionProvider",
        "platform_system": "Linux",
        "platform_release": "6.8-test",
        "architecture": "riscv64",
        "cpu_description": "RV64 Test CPU",
        "logical_core_count": 12,
    }


def test_source_revision_requires_lowercase_40_character_commit(
    tmp_path: Path,
) -> None:
    initialize_repository(tmp_path)

    revision = collect_source_revision(tmp_path)

    assert len(revision.commit) == 40
    assert revision.commit == revision.commit.lower()
    assert revision.working_tree_clean is True
    with pytest.raises(ValidationError):
        MobileNetV2P43BSourceRevision(
            vcs="git",
            commit="A" * 40,
            working_tree_clean=True,
        )


def test_source_revision_detects_tracked_and_untracked_changes(
    tmp_path: Path,
) -> None:
    tracked_repository = tmp_path / "tracked"
    tracked_repository.mkdir()
    initialize_repository(tracked_repository)
    (tracked_repository / "tracked.txt").write_text("changed\n", encoding="utf-8")
    assert collect_source_revision(tracked_repository).working_tree_clean is False

    untracked_repository = tmp_path / "untracked"
    untracked_repository.mkdir()
    initialize_repository(untracked_repository)
    (untracked_repository / "new.txt").write_text("untracked\n", encoding="utf-8")
    assert collect_source_revision(untracked_repository).working_tree_clean is False


def test_source_revision_rejects_missing_git_repository(tmp_path: Path) -> None:
    with pytest.raises(EnvironmentCaptureError, match="Git repository"):
        collect_source_revision(tmp_path)


def test_reproducibility_record_is_canonical_under_identical_inputs() -> None:
    first = reproducibility_record()
    second = reproducibility_record()

    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert type(first).model_validate_json(canonical_json_bytes(first)) == first
    assert set(first.inputs.model_fields_set) == set(
        MobileNetV2P43BPipelineInputDigests.model_fields
    )
    assert set(first.outputs.model_fields_set) == set(
        MobileNetV2P43BPipelineOutputDigests.model_fields
    )
    assert first.status == "provisional"
    assert first.production_verified is False
    assert first.label_source == "fp32-top1-pseudo-label"


def test_reproducibility_record_excludes_forbidden_host_fields() -> None:
    serialized = canonical_json_text(reproducibility_record())

    for forbidden in (
        "timestamp",
        "hostname",
        "username",
        "ip_address",
        "absolute_path",
        "credential",
        "dataset_image_bytes",
    ):
        assert forbidden not in serialized
    assert "/" not in serialized


def test_reproducibility_record_write_refuses_overwrite(tmp_path: Path) -> None:
    destination = tmp_path / "reproducibility.json"
    record = reproducibility_record()
    write_canonical_json(destination, record)

    with pytest.raises(PipelineIOError, match="Refusing to overwrite"):
        write_canonical_json(destination, record)
