"""Creation and filesystem persistence of benchmark run records."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from rvai.adapters import BenchmarkResult
from rvai.hardware import HardwareProfile
from rvai.manifest import ModelManifest
from rvai.results.schema import RunRecord


class ResultStoreError(RuntimeError):
    """Raised when a run record cannot be created, read, or written safely."""


def digest_manifest(manifest: ModelManifest) -> str:
    """Return a deterministic digest of the validated Manifest contents."""

    canonical = json.dumps(
        manifest.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def create_run_record(
    *,
    command: list[str],
    manifest: ModelManifest,
    target: str,
    hardware_profile: HardwareProfile | None,
    result: BenchmarkResult,
) -> RunRecord:
    """Create a versioned record for one completed benchmark execution."""

    return RunRecord(
        run_id=str(uuid4()),
        created_at=datetime.now(timezone.utc),
        command=command,
        model=manifest.name,
        target=target,
        manifest_digest=digest_manifest(manifest),
        hardware_profile=hardware_profile,
        result=result,
    )


def save_run_record(
    record: RunRecord,
    path: Path | str,
    *,
    force: bool = False,
) -> Path:
    """Persist a record, creating parents and refusing accidental overwrite."""

    destination = Path(path)
    temporary_path: Path | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not force:
            raise FileExistsError(destination)

        serialized = record.model_dump_json(indent=2) + "\n"
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            output.write(serialized)
            output.flush()
            os.fsync(output.fileno())

        if destination.exists() and not force:
            raise FileExistsError(destination)
        os.replace(temporary_path, destination)
        temporary_path = None
    except FileExistsError as exc:
        raise ResultStoreError(
            f"Result file already exists: {destination}; use --force to overwrite"
        ) from exc
    except OSError as exc:
        raise ResultStoreError(
            f"Cannot write result file {destination}: {exc}"
        ) from exc
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
    return destination


def load_run_record(path: Path | str) -> RunRecord:
    """Load one supported, strictly validated RunRecord JSON file."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ResultStoreError(f"Invalid JSON in result file {source}") from exc
    except OSError as exc:
        raise ResultStoreError(f"Cannot read result file {source}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ResultStoreError(f"Run record must be a JSON object: {source}")
    version = payload.get("schema_version")
    if version != "1.0":
        raise ResultStoreError(
            f"Unsupported RunRecord schema_version {version!r} in {source}"
        )
    try:
        return RunRecord.model_validate(payload)
    except ValidationError as exc:
        raise ResultStoreError(f"Invalid RunRecord in {source}: {exc}") from exc
