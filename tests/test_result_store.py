import json
from pathlib import Path

import pytest
from pydantic import ValidationError

import rvai.results.store as result_store
from rvai.manifest import ModelManifest
from rvai.registry import ModelRegistry
from rvai.results import (
    ResultStoreError,
    RunRecord,
    digest_manifest,
    load_run_record,
    save_run_record,
)


def manifest(min_memory_mb: int = 128) -> ModelManifest:
    return ModelManifest.model_validate(
        {
            "name": "builtin-gemm-int8",
            "display_name": "Builtin INT8 GEMM",
            "task": "benchmark",
            "format": "builtin",
            "quantization": "int8",
            "runtime": "builtin",
            "resources": {
                "min_memory_mb": min_memory_mb,
                "recommended_threads": "auto",
            },
            "riscv": {"require_rv64": False, "prefer_rvv": True},
        }
    )


def test_manifest_digest_is_stable_and_content_sensitive() -> None:
    first = digest_manifest(manifest())

    assert first == digest_manifest(manifest())
    assert first != digest_manifest(manifest(min_memory_mb=256))
    assert first.startswith("sha256:")
    assert len(first) == 71


def test_manifest_digest_ignores_yaml_layout_and_line_endings(tmp_path) -> None:
    ordered = """name: builtin-gemm-int8
display_name: Builtin INT8 GEMM
task: benchmark
format: builtin
quantization: int8
runtime: builtin
resources:
  min_memory_mb: 128
  recommended_threads: auto
riscv:
  require_rv64: false
  prefer_rvv: true
"""
    reordered = """runtime: builtin
quantization: int8
format: builtin
task: benchmark
display_name: Builtin INT8 GEMM
name: builtin-gemm-int8
riscv:
  prefer_rvv: true
  require_rv64: false
resources:
  recommended_threads: auto
  min_memory_mb: 128
"""
    variants = (
        ordered,
        reordered,
        ordered.replace("\n", "\r\n"),
        ordered.rstrip("\n"),
    )
    digests = []
    for index, contents in enumerate(variants):
        models_dir = tmp_path / str(index)
        models_dir.mkdir()
        (models_dir / "model.yaml").write_bytes(contents.encode("utf-8"))
        digests.append(
            digest_manifest(ModelRegistry(models_dir).get("builtin-gemm-int8"))
        )

    assert len(set(digests)) == 1


def test_run_record_rejects_unknown_schema_version(run_record_factory) -> None:
    payload = run_record_factory().model_dump(mode="json")
    payload["schema_version"] = "2.0"

    with pytest.raises(ValidationError):
        RunRecord.model_validate(payload)


def test_store_creates_parent_and_round_trips(tmp_path, run_record_factory) -> None:
    destination = tmp_path / "nested" / "record.json"
    record = run_record_factory()

    save_run_record(record, destination)

    assert load_run_record(destination) == record


def test_store_refuses_overwrite_without_force(tmp_path, run_record_factory) -> None:
    destination = tmp_path / "record.json"
    save_run_record(run_record_factory(run_id="first"), destination)

    with pytest.raises(ResultStoreError, match="use --force"):
        save_run_record(run_record_factory(run_id="second"), destination)

    assert load_run_record(destination).run_id == "first"


def test_store_force_overwrites(tmp_path, run_record_factory) -> None:
    destination = tmp_path / "record.json"
    save_run_record(run_record_factory(run_id="first"), destination)

    save_run_record(run_record_factory(run_id="second"), destination, force=True)

    assert load_run_record(destination).run_id == "second"


def test_store_flushes_complete_json_before_atomic_replace(
    tmp_path,
    run_record_factory,
    monkeypatch,
) -> None:
    destination = tmp_path / "record.json"
    record = run_record_factory()
    real_replace = result_store.os.replace
    observed = {}

    def inspect_replace(source, target):
        temporary = Path(source)
        observed["payload"] = json.loads(temporary.read_text(encoding="utf-8"))
        assert Path(target) == destination
        assert destination.exists() is False
        real_replace(source, target)

    monkeypatch.setattr(result_store.os, "replace", inspect_replace)

    save_run_record(record, destination)

    assert observed["payload"] == record.model_dump(mode="json")
    assert load_run_record(destination) == record


def test_failed_atomic_replace_preserves_existing_record(
    tmp_path,
    run_record_factory,
    monkeypatch,
) -> None:
    destination = tmp_path / "record.json"
    save_run_record(run_record_factory(run_id="first"), destination)

    def fail_replace(source, target):
        raise OSError("replace failed")

    monkeypatch.setattr(result_store.os, "replace", fail_replace)

    with pytest.raises(ResultStoreError, match="replace failed"):
        save_run_record(
            run_record_factory(run_id="second"),
            destination,
            force=True,
        )

    assert load_run_record(destination).run_id == "first"
    assert list(tmp_path.glob(".record.json.*.tmp")) == []


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("not-json", "Invalid JSON"),
        ("[]", "JSON object"),
        (json.dumps({"schema_version": "2.0"}), "Unsupported RunRecord"),
        (json.dumps({"schema_version": "1.0"}), "Invalid RunRecord"),
    ],
)
def test_load_rejects_invalid_records(tmp_path, contents, message) -> None:
    source = tmp_path / "record.json"
    source.write_text(contents, encoding="utf-8")

    with pytest.raises(ResultStoreError, match=message):
        load_run_record(source)
