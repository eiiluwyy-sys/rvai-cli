"""Canonical serialization and safe local I/O for pipeline records."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ValidationError
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

from rvai.model_pipeline.errors import PipelineIOError


ModelT = TypeVar("ModelT", bound=BaseModel)
_CHUNK_SIZE = 1024 * 1024


def _record_mapping(value: BaseModel | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=False)
    if isinstance(value, Mapping):
        return dict(value)
    raise PipelineIOError("Canonical JSON records must have an object root")


def canonical_json_bytes(value: BaseModel | Mapping[str, Any]) -> bytes:
    """Return the exact deterministic UTF-8 encoding used for record identity."""

    try:
        text = json.dumps(
            _record_mapping(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            indent=None,
        )
    except (TypeError, ValueError) as exc:
        raise PipelineIOError(f"Record is not canonical JSON data: {exc}") from exc
    return text.encode("utf-8")


def canonical_json_text(value: BaseModel | Mapping[str, Any]) -> str:
    """Return canonical JSON text with no indentation or trailing newline."""

    return canonical_json_bytes(value).decode("utf-8")


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase SHA-256 digest of bytes."""

    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str) -> str:
    """Stream a file through SHA-256 without loading it into memory."""

    source = Path(path)
    digest = hashlib.sha256()
    try:
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_CHUNK_SIZE), b""):
                digest.update(chunk)
    except OSError as exc:
        detail = exc.strerror or str(exc)
        raise PipelineIOError(f"Cannot hash file {source}: {detail}") from exc
    return digest.hexdigest()


def sha256_canonical_json(value: BaseModel | Mapping[str, Any]) -> str:
    """Digest the exact canonical JSON representation of a record."""

    return sha256_bytes(canonical_json_bytes(value))


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON number {value} is not allowed")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key {key!r}")
        result[key] = value
    return result


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys at every depth."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        detail = getattr(exc, "strerror", None) or str(exc)
        raise PipelineIOError(f"Cannot read {path}: {detail}") from exc


def _validate_object_root(raw: Any, path: Path) -> dict[str, Any]:
    if raw is None:
        raise PipelineIOError(f"Document is empty: {path}")
    if not isinstance(raw, dict):
        raise PipelineIOError(f"Document root must be an object: {path}")
    return raw


def _validate_model(raw: dict[str, Any], path: Path, model_type: type[ModelT]) -> ModelT:
    try:
        return model_type.model_validate(raw)
    except ValidationError as exc:
        raise PipelineIOError(f"Invalid {model_type.__name__} in {path}: {exc}") from exc


def load_json(path: Path | str, model_type: type[ModelT]) -> ModelT:
    """Load one strict model from duplicate-free object-root JSON."""

    source = Path(path)
    text = _read_text(source)
    if not text.strip():
        raise PipelineIOError(f"Document is empty: {source}")
    try:
        raw = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise PipelineIOError(f"Invalid JSON in {source}: {exc}") from exc
    return _validate_model(_validate_object_root(raw, source), source, model_type)


def load_yaml(path: Path | str, model_type: type[ModelT]) -> ModelT:
    """Load one strict model from duplicate-free safe object-root YAML."""

    source = Path(path)
    text = _read_text(source)
    if not text.strip():
        raise PipelineIOError(f"Document is empty: {source}")
    try:
        raw = yaml.load(text, Loader=_UniqueKeySafeLoader)
    except yaml.YAMLError as exc:
        raise PipelineIOError(f"Invalid YAML in {source}: {exc}") from exc
    return _validate_model(_validate_object_root(raw, source), source, model_type)


def write_canonical_json(
    path: Path | str,
    value: BaseModel | Mapping[str, Any],
) -> None:
    """Atomically publish canonical JSON without ever replacing a destination."""

    destination = Path(path)
    parent = destination.parent
    if not parent.is_dir():
        raise PipelineIOError(f"Destination directory does not exist: {parent}")
    payload = canonical_json_bytes(value)
    temporary: Path | None = None
    linked = False
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=parent,
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
            linked = True
        except FileExistsError as exc:
            raise PipelineIOError(f"Refusing to overwrite existing file: {destination}") from exc
        temporary.unlink()
        temporary = None
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_fd = os.open(parent, directory_flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except PipelineIOError:
        raise
    except OSError as exc:
        detail = exc.strerror or str(exc)
        state = "published but not fully synchronized" if linked else "not published"
        raise PipelineIOError(
            f"Cannot publish canonical JSON to {destination} ({state}): {detail}"
        ) from exc
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass
