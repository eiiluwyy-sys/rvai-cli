"""Persistent benchmark result APIs."""

from rvai.results.schema import RunRecord
from rvai.results.render import (
    ReportFormat,
    ReportRenderError,
    escape_markdown,
    render_markdown,
    save_markdown_report,
)
from rvai.results.store import (
    ResultStoreError,
    create_run_record,
    digest_manifest,
    load_run_record,
    save_run_record,
)

__all__ = [
    "ResultStoreError",
    "ReportFormat",
    "ReportRenderError",
    "RunRecord",
    "create_run_record",
    "digest_manifest",
    "escape_markdown",
    "load_run_record",
    "render_markdown",
    "save_markdown_report",
    "save_run_record",
]
