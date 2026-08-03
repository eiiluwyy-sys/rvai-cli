"""Human-readable rendering for validated benchmark run records."""

from __future__ import annotations

import re
import shlex
from enum import Enum
from pathlib import Path

from rvai.results.schema import RunRecord


class ReportRenderError(RuntimeError):
    """Raised when a rendered report cannot be written safely."""


class ReportFormat(str, Enum):
    MARKDOWN = "markdown"


_MARKDOWN_SPECIAL = re.compile(r"([\\`*_[\]{}()#+.!<>|~-])")


def escape_markdown(value: object) -> str:
    """Escape dynamic text so it cannot change Markdown structure."""

    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\n", "<br>")
    return _MARKDOWN_SPECIAL.sub(r"\\\1", text)


def _boolean(value: bool) -> str:
    return "true" if value else "false"


def _row(label: str, value: object) -> str:
    return f"| {escape_markdown(label)} | {escape_markdown(value)} |"


def render_markdown(record: RunRecord) -> str:
    """Render every stable RunRecord and BenchmarkResult field as Markdown."""

    result = record.result
    execution = result.execution
    matrix = result.matrix
    memory = result.memory_bytes
    lines = [
        f"# RVAI Run Report: {escape_markdown(record.model)}",
        "",
        "## Run record",
        "",
        "| Field | Value |",
        "| --- | --- |",
        _row("Schema version", record.schema_version),
        _row("Run ID", record.run_id),
        _row("Created at", record.created_at.isoformat()),
        _row("Command", shlex.join(record.command)),
        _row("Model", record.model),
        _row("Target", record.target),
        _row("Manifest digest", record.manifest_digest),
        "",
        "## Benchmark result",
        "",
        "| Field | Value |",
        "| --- | --- |",
        _row("Workload", result.workload),
        _row("Status", result.status),
        _row("Backend", result.backend),
        _row("Matrix M", matrix.m),
        _row("Matrix N", matrix.n),
        _row("Matrix K", matrix.k),
        _row("Iterations", result.iterations),
        _row("Correctness verified", _boolean(result.correctness_verified)),
        _row("Mean latency (ms)", result.latency_ms.mean),
        _row("P95 latency (ms)", result.latency_ms.p95),
        _row("Throughput (GOPS)", result.throughput_gops),
        _row("Input memory (bytes)", memory.inputs),
        _row("Output memory (bytes)", memory.output),
        _row("Total memory (bytes)", memory.total),
        "",
        "## Execution environment",
        "",
        "| Field | Value |",
        "| --- | --- |",
        _row("Target architecture", execution.target_architecture),
        _row("Execution environment", execution.execution_environment),
        _row("Host architecture", execution.host_architecture),
        _row(
            "Performance representative",
            _boolean(execution.performance_representative),
        ),
        "",
        "## Hardware profile",
        "",
    ]

    hardware = record.hardware_profile
    if hardware is None:
        lines.append("Hardware profile was unavailable when this run was saved.")
    else:
        riscv = hardware.riscv
        lines.extend(
            [
                "| Field | Value |",
                "| --- | --- |",
                _row("Schema version", hardware.schema_version),
                _row("OS", hardware.platform.os),
                _row("Kernel", hardware.platform.kernel),
                _row("Architecture", hardware.platform.architecture),
                _row("Logical cores", hardware.cpu.logical_cores),
                _row("Total memory (MB)", hardware.memory.total_mb),
                _row("Available memory (MB)", hardware.memory.available_mb),
                _row("Is RISC-V", _boolean(riscv.is_riscv)),
                _row("RISC-V XLEN", riscv.xlen if riscv.xlen else "unknown"),
                _row("RISC-V ISA", riscv.isa or "unknown"),
                _row("RISC-V extensions", ", ".join(riscv.extensions) or "none"),
                _row(
                    "RVV",
                    "unknown" if riscv.rvv is None else _boolean(riscv.rvv),
                ),
                "",
                "### Runtime availability",
                "",
                "| Runtime | Available | Version | Executable |",
                "| --- | --- | --- | --- |",
            ]
        )
        for name in ("builtin", "llama_cpp", "onnxruntime"):
            status = getattr(hardware.runtimes, name)
            lines.append(
                "| "
                + " | ".join(
                    escape_markdown(value)
                    for value in (
                        name,
                        _boolean(status.available),
                        status.version or "unknown",
                        status.executable or "unknown",
                    )
                )
                + " |"
            )

    lines.append("")
    return "\n".join(lines)


def save_markdown_report(
    report: str,
    path: Path | str,
    *,
    force: bool = False,
) -> Path:
    """Save Markdown while refusing accidental overwrite by default."""

    destination = Path(path)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if force else "x"
        with destination.open(mode, encoding="utf-8") as output:
            output.write(report)
    except FileExistsError as exc:
        raise ReportRenderError(
            f"Report file already exists: {destination}; use --force to overwrite"
        ) from exc
    except OSError as exc:
        raise ReportRenderError(f"Cannot write report file {destination}: {exc}") from exc
    return destination
