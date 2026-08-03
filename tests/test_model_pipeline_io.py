import json
from pathlib import Path

import pytest

import rvai.model_pipeline.io as pipeline_io
from rvai.model_pipeline.errors import PipelineIOError
from rvai.model_pipeline.io import (
    canonical_json_bytes,
    canonical_json_text,
    load_json,
    load_yaml,
    sha256_bytes,
    sha256_canonical_json,
    sha256_file,
    write_canonical_json,
)
from rvai.model_pipeline.schema import StrictModel


class ExampleRecord(StrictModel):
    name: str
    count: int
    optional: str | None = None


def test_canonical_json_has_exact_deterministic_encoding() -> None:
    record = ExampleRecord(name="雪", count=2)
    expected = '{"count":2,"name":"雪","optional":null}'

    assert canonical_json_text(record) == expected
    assert canonical_json_bytes(record) == expected.encode("utf-8")
    assert not canonical_json_bytes(record).endswith(b"\n")
    assert canonical_json_bytes({"name": "雪", "count": 2, "optional": None}) == (
        canonical_json_bytes(record)
    )
    assert sha256_canonical_json(record) == sha256_bytes(expected.encode("utf-8"))


def test_canonical_json_rejects_non_object_and_non_finite_data() -> None:
    with pytest.raises(PipelineIOError, match="object root"):
        canonical_json_bytes([1, 2, 3])
    with pytest.raises(PipelineIOError, match="not canonical JSON"):
        canonical_json_bytes({"value": float("nan")})


def test_sha256_file_streams_file(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    payload = b"abc" * 400_000
    path.write_bytes(payload)
    assert sha256_file(path) == sha256_bytes(payload)


@pytest.mark.parametrize("suffix", ["json", "yaml"])
def test_loaders_reject_empty_and_non_object_documents(
    suffix: str, tmp_path: Path
) -> None:
    loader = load_json if suffix == "json" else load_yaml
    empty = tmp_path / f"empty.{suffix}"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(PipelineIOError, match="empty"):
        loader(empty, ExampleRecord)

    scalar = tmp_path / f"scalar.{suffix}"
    scalar.write_text("1", encoding="utf-8")
    with pytest.raises(PipelineIOError, match="object"):
        loader(scalar, ExampleRecord)


def test_json_loader_rejects_duplicate_keys_at_nested_depth(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"name":"first","name":"second","count":1}', encoding="utf-8")
    with pytest.raises(PipelineIOError, match="duplicate key"):
        load_json(path, ExampleRecord)


def test_yaml_loader_rejects_duplicate_keys_at_nested_depth(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text("name: first\nname: second\ncount: 1\n", encoding="utf-8")
    with pytest.raises(PipelineIOError, match="duplicate key"):
        load_yaml(path, ExampleRecord)


def test_loaders_validate_strict_models(tmp_path: Path) -> None:
    json_path = tmp_path / "record.json"
    json_path.write_text('{"name":"record","count":2}', encoding="utf-8")
    yaml_path = tmp_path / "record.yaml"
    yaml_path.write_text("name: record\ncount: 2\n", encoding="utf-8")
    assert load_json(json_path, ExampleRecord).count == 2
    assert load_yaml(yaml_path, ExampleRecord).count == 2

    json_path.write_text('{"name":"record","count":"2"}', encoding="utf-8")
    with pytest.raises(PipelineIOError, match="Invalid ExampleRecord"):
        load_json(json_path, ExampleRecord)


def test_atomic_write_publishes_exact_bytes_without_overwrite(tmp_path: Path) -> None:
    destination = tmp_path / "record.json"
    record = ExampleRecord(name="record", count=2)

    write_canonical_json(destination, record)

    assert destination.read_bytes() == canonical_json_bytes(record)
    with pytest.raises(PipelineIOError, match="Refusing to overwrite"):
        write_canonical_json(destination, ExampleRecord(name="changed", count=3))
    assert destination.read_bytes() == canonical_json_bytes(record)
    assert list(tmp_path.glob(".record.json.*.tmp")) == []


def test_atomic_write_cleans_temporary_file_after_link_failure(
    monkeypatch, tmp_path: Path
) -> None:
    destination = tmp_path / "record.json"

    def fail_link(source: Path, target: Path) -> None:
        raise OSError("link unavailable")

    monkeypatch.setattr(pipeline_io.os, "link", fail_link)
    with pytest.raises(PipelineIOError, match="not published"):
        write_canonical_json(destination, {"value": 1})

    assert not destination.exists()
    assert list(tmp_path.glob(".record.json.*.tmp")) == []


def test_canonical_output_is_standard_json() -> None:
    encoded = canonical_json_text(ExampleRecord(name="record", count=2))
    assert json.loads(encoded) == {"count": 2, "name": "record", "optional": None}
