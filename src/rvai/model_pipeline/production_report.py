"""Deterministic human-readable report for formal P4.3B evidence."""

from __future__ import annotations

from rvai.model_pipeline.compare import MobileNetV2P43BComparisonRecord
from rvai.model_pipeline.production import (
    MobileNetV2P43BProductionReport,
    MobileNetV2P43BProductionReproducibilityRecord,
)
from rvai.model_pipeline.quantize import MobileNetV2P43BQuantizationRecord


def render_production_markdown(
    comparison: MobileNetV2P43BComparisonRecord,
    quantization: MobileNetV2P43BQuantizationRecord,
    report: MobileNetV2P43BProductionReport,
    reproducibility: MobileNetV2P43BProductionReproducibilityRecord,
) -> str:
    """Render one accepted or rejected independently labelled result."""

    if report.acceptance_passed != comparison.decision.overall_passed:
        raise ValueError("Production report and comparison decision disagree")
    if report.status != reproducibility.status:
        raise ValueError("Production report and reproducibility status disagree")
    decision = comparison.decision
    rows = (
        ("Top-1 accuracy drop", comparison.top1_drop_percentage_points, decision.top1_accuracy_passed),
        ("Top-5 accuracy drop", comparison.top5_drop_percentage_points, decision.top5_accuracy_passed),
        ("Model size reduction", comparison.model_size_reduction_ratio, decision.model_size_passed),
        ("Top-1 agreement", comparison.top1_agreement_ratio, decision.top1_agreement_passed),
        ("Zero inference failures", comparison.total_inference_failures, decision.zero_inference_failures_passed),
        ("Finite outputs", comparison.all_outputs_finite, decision.finite_outputs_passed),
    )
    gate_lines = "\n".join(
        f"| {name} | `{value}` | {'PASS' if passed else 'FAIL'} |"
        for name, value, passed in rows
    )
    return (
        "# MobileNetV2 P4.3B formal production evidence\n\n"
        f"**Status: {report.status.upper()}**  \n"
        "Label source: independently labelled ground truth.  \n"
        f"Production verified: `{str(report.production_verified).lower()}`\n\n"
        "## Dataset selection\n\n"
        f"- Calibration: {report.calibration_actual_sample_count} / "
        f"{report.calibration_requested_sample_count}\n"
        f"- Evaluation: {report.evaluation_actual_sample_count} / "
        f"{report.evaluation_requested_sample_count}\n"
        f"- Complete short evaluation set used: "
        f"`{str(report.evaluation_used_complete_available_set).lower()}`\n\n"
        "## Acceptance gates\n\n"
        "| Gate | Observed | Result |\n"
        "|---|---:|:---:|\n"
        f"{gate_lines}\n\n"
        "## Artifact identity\n\n"
        f"- INT8 filename: `{quantization.artifact.filename}`\n"
        f"- INT8 size: `{quantization.artifact.size_bytes}` bytes\n"
        f"- INT8 SHA-256: `{quantization.artifact.sha256}`\n"
        f"- Package source revision: `{reproducibility.source_revision.commit}`\n\n"
        "A rejected result is preserved evidence and does not authorize automatic "
        "threshold or quantization changes. This report is model-production "
        "evidence, not physical RISC-V performance evidence.\n"
    )
