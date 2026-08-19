"""Deterministic Markdown reporting for validated P4.3B proxy evidence."""

from __future__ import annotations

from rvai.model_pipeline.compare import MobileNetV2P43BComparisonRecord
from rvai.model_pipeline.environment import MobileNetV2P43BReproducibilityRecord
from rvai.model_pipeline.errors import ModelPipelineError
from rvai.model_pipeline.io import sha256_canonical_json
from rvai.model_pipeline.pilot import MobileNetV2P43BProxyPilotReport
from rvai.model_pipeline.quantize import MobileNetV2P43BQuantizationRecord


class ReportValidationError(ModelPipelineError):
    """Raised when report inputs are inconsistent or unsafe to render."""


def render_comparison_markdown(
    comparison: MobileNetV2P43BComparisonRecord,
    quantization: MobileNetV2P43BQuantizationRecord,
    proxy_report: MobileNetV2P43BProxyPilotReport,
    reproducibility: MobileNetV2P43BReproducibilityRecord,
) -> str:
    """Render validated records as stable UTF-8-compatible Markdown text."""

    _validate_report_inputs(comparison, quantization, proxy_report, reproducibility)
    decision = comparison.decision
    acceptance = comparison.acceptance
    software = reproducibility.software
    execution = reproducibility.execution
    source = reproducibility.source_revision
    artifact = quantization.artifact
    structure = quantization.structure

    lines = [
        "# P4.3B MobileNetV2 Synthetic Proxy Evidence Report",
        "",
        "> **PROVISIONAL:** This report uses FP32 Top-1 pseudo-labels. It is not "
        "independently labelled ImageNet accuracy, production verification, or "
        "physical RISC-V performance.",
        "",
        "## 1. Scope and status",
        "",
        "| Field | Value |",
        "|---|---|",
        "| Status | PROVISIONAL |",
        "| Production verified | No |",
        "| Label source | `fp32-top1-pseudo-label` |",
        f"| Overall proxy acceptance | {_pass_fail(decision.overall_passed)} |",
        "",
        "Pseudo-label accuracy measures consistency with the frozen FP32 model; it "
        "is not independent ImageNet accuracy. These results are not physical "
        "RISC-V performance measurements.",
        "",
        "## 2. Artifact identity",
        "",
        "| Artifact | Size | SHA-256 |",
        "|---|---:|---|",
        f"| FP32 source | {comparison.fp32_model_size_bytes} bytes | "
        f"`{reproducibility.inputs.source_fp32_model_sha256}` |",
        f"| INT8 QDQ | {artifact.size_bytes} bytes | `{artifact.sha256}` |",
        "",
        "## 3. Pipeline configuration",
        "",
        "| Setting | Value |",
        "|---|---|",
        f"| Method | `{quantization.quantization.method}` |",
        f"| Format | `{quantization.quantization.format}` |",
        f"| Activation type | `{quantization.quantization.activation_type}` |",
        f"| Weight type | `{quantization.quantization.weight_type}` |",
        f"| Per-channel weights | {_yes_no(quantization.quantization.per_channel)} |",
        f"| Calibration method | `{quantization.quantization.calibration_method}` |",
        f"| Calibration order | `{quantization.quantization.calibration_order}` |",
        f"| Execution provider | `{quantization.execution_provider}` |",
        "",
        "## 4. Dataset and evaluation",
        "",
        "| Field | Value |",
        "|---|---:|",
        f"| Calibration samples | {proxy_report.calibration_sample_count} |",
        f"| Evaluation samples | {proxy_report.evaluation_sample_count} |",
        f"| Compared samples | {comparison.sample_count} |",
        f"| Total inference failures | {comparison.total_inference_failures} |",
        f"| All outputs finite | {_yes_no(comparison.all_outputs_finite)} |",
        "",
        "## 5. Accuracy and consistency",
        "",
        "| Metric | FP32 | INT8 | Observed change |",
        "|---|---:|---:|---:|",
        f"| Top-1 pseudo-label accuracy | {_percent(comparison.fp32_top1_accuracy_ratio)} "
        f"| {_percent(comparison.int8_top1_accuracy_ratio)} | "
        f"{_percentage_points(comparison.top1_drop_percentage_points)} drop |",
        f"| Top-5 pseudo-label accuracy | {_percent(comparison.fp32_top5_accuracy_ratio)} "
        f"| {_percent(comparison.int8_top5_accuracy_ratio)} | "
        f"{_percentage_points(comparison.top5_drop_percentage_points)} drop |",
        f"| Model size | {comparison.fp32_model_size_bytes} bytes | "
        f"{comparison.int8_model_size_bytes} bytes | "
        f"{_percent(comparison.model_size_reduction_ratio)} reduction |",
        f"| Top-1 agreement | — | — | {_percent(comparison.top1_agreement_ratio)} |",
        f"| Mean Top-5 overlap | — | — | "
        f"{_percent(comparison.mean_top5_overlap_ratio)} |",
        "",
        "## 6. Acceptance gates",
        "",
        "| Gate | Observed | Requirement | Result |",
        "|---|---:|---:|---|",
        f"| Top-1 degradation | "
        f"{_percentage_points(comparison.top1_drop_percentage_points)} | <= "
        f"{_percentage_points(acceptance.max_top1_drop_percentage_points)} | "
        f"{_pass_fail(decision.top1_accuracy_passed)} |",
        f"| Top-5 degradation | "
        f"{_percentage_points(comparison.top5_drop_percentage_points)} | <= "
        f"{_percentage_points(acceptance.max_top5_drop_percentage_points)} | "
        f"{_pass_fail(decision.top5_accuracy_passed)} |",
        f"| Model-size reduction | {_percent(comparison.model_size_reduction_ratio)} "
        f"| >= {_percent(acceptance.min_model_size_reduction_ratio)} | "
        f"{_pass_fail(decision.model_size_passed)} |",
        f"| Top-1 agreement | {_percent(comparison.top1_agreement_ratio)} | >= "
        f"{_percent(acceptance.min_top1_agreement_ratio)} | "
        f"{_pass_fail(decision.top1_agreement_passed)} |",
        f"| Inference failures | {comparison.total_inference_failures} | 0 | "
        f"{_pass_fail(decision.zero_inference_failures_passed)} |",
        f"| Finite outputs | {_yes_no(comparison.all_outputs_finite)} | Required | "
        f"{_pass_fail(decision.finite_outputs_passed)} |",
        f"| **Overall proxy acceptance** |  |  | "
        f"**{_pass_fail(decision.overall_passed)}** |",
        "",
        "## 7. Quantized model structure",
        "",
        "| Field | Value |",
        "|---|---:|",
        f"| Source opset | {quantization.source_opset_version} |",
        f"| Quantization input opset | {quantization.quantization_input_opset_version} |",
        f"| Opset conversion applied | {_yes_no(quantization.opset_conversion_applied)} |",
        f"| Nodes | {structure.node_count} |",
        f"| Initializers | {structure.initializer_count} |",
        f"| QuantizeLinear nodes | {structure.quantize_linear_count} |",
        f"| DequantizeLinear nodes | {structure.dequantize_linear_count} |",
        "",
        "## 8. Reproducibility environment",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Python | `{_escape(software.python_version)}` |",
        f"| RVAI | `{_escape(software.rvai_version)}` |",
        f"| ONNX | `{_escape(software.onnx_version)}` |",
        f"| ONNX Runtime | `{_escape(software.onnxruntime_version)}` |",
        f"| NumPy | `{_escape(software.numpy_version)}` |",
        f"| Pillow | `{_escape(software.pillow_version)}` |",
        f"| Platform | `{_escape(execution.platform_system)}` |",
        f"| Platform release | `{_escape(execution.platform_release)}` |",
        f"| Architecture | `{_escape(execution.architecture)}` |",
        f"| CPU | `{_escape(execution.cpu_description)}` |",
        f"| Logical cores | {execution.logical_core_count} |",
        f"| Run source revision | `{source.commit}` |",
        f"| Run worktree clean | {_yes_no(source.working_tree_clean)} |",
        "",
        "## 9. Evidence integrity",
        "",
        "| Record | SHA-256 |",
        "|---|---|",
        f"| Pipeline configuration | `{reproducibility.inputs.pipeline_config_sha256}` |",
        f"| Source inspection | `{reproducibility.outputs.source_inspection_sha256}` |",
        f"| Quantization | `{reproducibility.outputs.quantization_record_sha256}` |",
        f"| FP32 evaluation | `{reproducibility.outputs.fp32_evaluation_sha256}` |",
        f"| INT8 evaluation | `{reproducibility.outputs.int8_evaluation_sha256}` |",
        f"| Comparison | `{reproducibility.outputs.comparison_sha256}` |",
        f"| Proxy pilot report | `{reproducibility.outputs.proxy_pilot_report_sha256}` |",
        "",
        "## 10. Limitations",
        "",
    ]
    lines.extend(f"- {_escape(item)}" for item in proxy_report.limitations)
    return "\n".join(lines) + "\n"


def _validate_report_inputs(
    comparison: MobileNetV2P43BComparisonRecord,
    quantization: MobileNetV2P43BQuantizationRecord,
    proxy_report: MobileNetV2P43BProxyPilotReport,
    reproducibility: MobileNetV2P43BReproducibilityRecord,
) -> None:
    links = (
        (proxy_report.comparison_sha256, sha256_canonical_json(comparison)),
        (proxy_report.quantization_record_sha256, sha256_canonical_json(quantization)),
        (reproducibility.outputs.comparison_sha256, sha256_canonical_json(comparison)),
        (
            reproducibility.outputs.quantization_record_sha256,
            sha256_canonical_json(quantization),
        ),
        (
            reproducibility.outputs.proxy_pilot_report_sha256,
            sha256_canonical_json(proxy_report),
        ),
    )
    if any(declared != actual for declared, actual in links):
        raise ReportValidationError("Report record digest link mismatch")
    if proxy_report.proxy_acceptance_passed != comparison.decision.overall_passed:
        raise ReportValidationError("Proxy report acceptance does not match comparison")
    if comparison.int8_model_size_bytes != quantization.artifact.size_bytes:
        raise ReportValidationError("INT8 model size does not match quantization")
    if reproducibility.outputs.int8_model_sha256 != quantization.artifact.sha256:
        raise ReportValidationError("INT8 model digest does not match quantization")
    if comparison.sample_count != proxy_report.evaluation_sample_count:
        raise ReportValidationError("Evaluation sample counts do not match")
    if quantization.calibration_sample_count != proxy_report.calibration_sample_count:
        raise ReportValidationError("Calibration sample counts do not match")
    if (
        proxy_report.status != "provisional"
        or proxy_report.production_verified is not False
        or proxy_report.label_source != "fp32-top1-pseudo-label"
        or reproducibility.status != "provisional"
        or reproducibility.production_verified is not False
        or reproducibility.label_source != "fp32-top1-pseudo-label"
    ):
        raise ReportValidationError("Report inputs violate provisional semantics")


def _percent(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def _percentage_points(value: float) -> str:
    return f"{value:.2f} pp"


def _pass_fail(value: bool) -> str:
    return "PASS" if value else "FAIL"


def _yes_no(value: bool) -> str:
    return "Yes" if value else "No"


def _escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("`", "&#96;")
        .replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\r\n", "<br>")
        .replace("\r", "<br>")
        .replace("\n", "<br>")
    )
