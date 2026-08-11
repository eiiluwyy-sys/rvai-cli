from rvai.model_pipeline.compare import (
    MobileNetV2P43BAcceptanceDecision,
    MobileNetV2P43BComparisonRecord,
)
from rvai.model_pipeline.environment import (
    MobileNetV2P43BExecutionEnvironment,
    MobileNetV2P43BPipelineInputDigests,
    MobileNetV2P43BPipelineOutputDigests,
    MobileNetV2P43BReproducibilityRecord,
    MobileNetV2P43BSoftwareEnvironment,
    MobileNetV2P43BSourceRevision,
)
from rvai.model_pipeline.io import sha256_canonical_json
from rvai.model_pipeline.pilot import MobileNetV2P43BProxyPilotReport
from rvai.model_pipeline.quantize import (
    MobileNetV2P43BQuantizationRecord,
    MobileNetV2P43BQuantizedArtifact,
    MobileNetV2P43BQuantizedStructure,
)
from rvai.model_pipeline.report import ReportValidationError
from rvai.model_pipeline.report import render_comparison_markdown
from rvai.model_pipeline.schema import (
    MobileNetV2P43BAcceptanceConfig,
    MobileNetV2P43BQuantizationConfig,
)


DIGEST = "a" * 64
INT8_DIGEST = "b" * 64


def report_records():
    acceptance = MobileNetV2P43BAcceptanceConfig(
        max_top1_drop_percentage_points=1.0,
        max_top5_drop_percentage_points=1.0,
        min_model_size_reduction_ratio=0.50,
        min_top1_agreement_ratio=0.95,
        require_zero_inference_failures=True,
        require_finite_outputs=True,
    )
    comparison = MobileNetV2P43BComparisonRecord(
        fp32_evaluation_sha256="1" * 64,
        int8_evaluation_sha256="2" * 64,
        evaluation_manifest_sha256="3" * 64,
        sample_count=200,
        fp32_top1_accuracy_ratio=1.0,
        int8_top1_accuracy_ratio=0.965,
        top1_drop_percentage_points=3.5,
        fp32_top5_accuracy_ratio=1.0,
        int8_top5_accuracy_ratio=1.0,
        top5_drop_percentage_points=0.0,
        fp32_model_size_bytes=13_964_571,
        int8_model_size_bytes=3_832_086,
        model_size_reduction_ratio=0.7255851253862363,
        top1_agreement_count=193,
        top1_agreement_ratio=0.965,
        mean_top5_overlap_ratio=0.862,
        total_inference_failures=0,
        all_outputs_finite=True,
        acceptance=acceptance,
        decision=MobileNetV2P43BAcceptanceDecision(
            top1_accuracy_passed=False,
            top5_accuracy_passed=True,
            model_size_passed=True,
            top1_agreement_passed=True,
            zero_inference_failures_passed=True,
            finite_outputs_passed=True,
            overall_passed=False,
        ),
    )
    quantization = MobileNetV2P43BQuantizationRecord(
        source_model_sha256="4" * 64,
        source_inspection_sha256="5" * 64,
        pipeline_config_sha256="6" * 64,
        calibration_manifest_sha256="7" * 64,
        calibration_sample_count=200,
        calibration_sample_ids=tuple(f"sample-{index}" for index in range(200)),
        quantization=MobileNetV2P43BQuantizationConfig(
            method="static",
            format="qdq",
            activation_type="quint8",
            weight_type="qint8",
            per_channel=True,
            calibration_method="minmax",
            execution_provider="CPUExecutionProvider",
            calibration_order="manifest",
        ),
        execution_provider="CPUExecutionProvider",
        onnxruntime_version="1.23.2",
        source_opset_version=12,
        quantization_input_opset_version=13,
        opset_conversion_applied=True,
        checker_passed=True,
        contract_matched=True,
        structure=MobileNetV2P43BQuantizedStructure(
            node_count=100,
            initializer_count=50,
            quantize_linear_count=66,
            dequantize_linear_count=172,
        ),
        artifact=MobileNetV2P43BQuantizedArtifact(
            filename="mobilenetv2-12-int8.onnx",
            size_bytes=3_832_086,
            sha256=INT8_DIGEST,
        ),
    )
    proxy = MobileNetV2P43BProxyPilotReport(
        report_type="synthetic-consistency-pilot",
        status="provisional",
        production_verified=False,
        label_source="fp32-top1-pseudo-label",
        calibration_sample_count=200,
        evaluation_sample_count=200,
        pipeline_config_sha256="6" * 64,
        source_inspection_sha256="5" * 64,
        generation_record_sha256="8" * 64,
        pseudo_label_record_sha256="9" * 64,
        overlap_report_sha256="a" * 64,
        quantization_record_sha256=sha256_canonical_json(quantization),
        fp32_evaluation_sha256="1" * 64,
        int8_evaluation_sha256="2" * 64,
        comparison_sha256=sha256_canonical_json(comparison),
        proxy_acceptance_passed=False,
        limitations=("Limit | <unsafe> `text`",),
    )
    outputs = MobileNetV2P43BPipelineOutputDigests(
        source_inspection_sha256="5" * 64,
        generation_record_sha256="8" * 64,
        pseudo_label_record_sha256="9" * 64,
        calibration_validation_sha256="c" * 64,
        evaluation_validation_sha256="d" * 64,
        overlap_report_sha256="a" * 64,
        quantization_record_sha256=sha256_canonical_json(quantization),
        int8_model_sha256=INT8_DIGEST,
        fp32_evaluation_sha256="1" * 64,
        int8_evaluation_sha256="2" * 64,
        comparison_sha256=sha256_canonical_json(comparison),
        proxy_pilot_report_sha256=sha256_canonical_json(proxy),
    )
    reproducibility = MobileNetV2P43BReproducibilityRecord(
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
            cpu_description="CPU | <tag> `tick`",
            logical_core_count=8,
        ),
        source_revision=MobileNetV2P43BSourceRevision(
            vcs="git",
            commit="e" * 40,
            working_tree_clean=True,
        ),
        inputs=MobileNetV2P43BPipelineInputDigests(
            pipeline_config_sha256="6" * 64,
            source_model_config_sha256=DIGEST,
            source_fp32_model_sha256="4" * 64,
            calibration_manifest_sha256="7" * 64,
            unlabeled_evaluation_manifest_sha256="f" * 64,
            evaluation_manifest_sha256="3" * 64,
        ),
        outputs=outputs,
    )
    return comparison, quantization, proxy, reproducibility


def test_markdown_is_deterministic_and_discloses_failed_proxy_semantics() -> None:
    records = report_records()

    first = render_comparison_markdown(*records)
    second = render_comparison_markdown(*records)

    assert first == second
    assert first.endswith("\n") and not first.endswith("\n\n")
    assert "**PROVISIONAL:**" in first
    assert "not independent ImageNet accuracy" in first
    assert "not physical RISC-V performance" in first
    assert "| Top-1 degradation | 3.50 pp | <= 1.00 pp | FAIL |" in first
    assert "| Model-size reduction | 72.56% | >= 50.00% | PASS |" in first
    assert "| **Overall proxy acceptance** |  |  | **FAIL** |" in first
    assert "timestamp" not in first.lower()


def test_markdown_escapes_all_record_derived_text() -> None:
    markdown = render_comparison_markdown(*report_records())

    assert "CPU \\| &lt;tag&gt; &#96;tick&#96;" in markdown
    assert "Limit \\| &lt;unsafe&gt; &#96;text&#96;" in markdown


def test_markdown_rejects_invalid_digest_link() -> None:
    comparison, quantization, proxy, reproducibility = report_records()
    invalid = reproducibility.model_copy(
        update={
            "outputs": reproducibility.outputs.model_copy(
                update={"comparison_sha256": "0" * 64}
            )
        }
    )

    try:
        render_comparison_markdown(comparison, quantization, proxy, invalid)
    except ReportValidationError as exc:
        assert "digest link" in str(exc)
    else:
        raise AssertionError("invalid digest link was accepted")
