"""Persistent benchmark result APIs."""

from rvai.results.schema import RunRecord
from rvai.results.store import (
    ResultStoreError,
    create_run_record,
    digest_manifest,
    load_run_record,
    save_run_record,
)

__all__ = [
    "ResultStoreError",
    "RunRecord",
    "create_run_record",
    "digest_manifest",
    "load_run_record",
    "save_run_record",
]
