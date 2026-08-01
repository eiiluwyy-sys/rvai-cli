from pathlib import Path

from rvai.compatibility import CompatibilityMatcher
from rvai.hardware.schema import (
    CpuInfo,
    HardwareProfile,
    MemoryInfo,
    PlatformInfo,
    RiscVInfo,
    RuntimeInfo,
    RuntimeStatus,
)
from rvai.manifest import ModelManifest
from rvai.registry import ModelRegistry


MODELS_DIR = Path(__file__).parents[1] / "models"


def model(name: str) -> ModelManifest:
    return ModelRegistry(MODELS_DIR).get(name)


def hardware(
    *,
    architecture: str = "riscv64",
    total_mb: int = 8192,
    available_mb: int = 8192,
    logical_cores: int = 8,
    rvv: bool | None = True,
    llama_cpp: bool = True,
    onnxruntime: bool = True,
) -> HardwareProfile:
    is_riscv = architecture in {"riscv32", "riscv64"}
    return HardwareProfile(
        platform=PlatformInfo(
            os="linux", kernel="6.8.0", architecture=architecture
        ),
        cpu=CpuInfo(logical_cores=logical_cores),
        memory=MemoryInfo(total_mb=total_mb, available_mb=available_mb),
        riscv=RiscVInfo(
            is_riscv=is_riscv,
            xlen=64 if architecture == "riscv64" else None,
            rvv=rvv if is_riscv else None,
        ),
        runtimes=RuntimeInfo(
            builtin=RuntimeStatus(available=True),
            llama_cpp=RuntimeStatus(available=llama_cpp),
            onnxruntime=RuntimeStatus(available=onnxruntime),
        ),
    )


def ready_matcher(*adapters: str) -> CompatibilityMatcher:
    return CompatibilityMatcher(
        available_adapters=adapters,
        model_file_exists=lambda manifest: True,
    )


def issue_codes(report) -> list[str]:
    return [issue.code for issue in report.blocking_reasons]


def warning_codes(report) -> list[str]:
    return [issue.code for issue in report.warnings]


def test_x86_qwen_reports_hardware_and_execution_blockers() -> None:
    report = CompatibilityMatcher().check(
        model("qwen-small-int4"),
        hardware(architecture="x86_64", llama_cpp=False),
    )

    assert report.compatible is False
    assert report.ready is False
    assert issue_codes(report) == [
        "architecture_mismatch",
        "runtime_unavailable",
        "model_file_missing",
        "adapter_unavailable",
    ]


def test_compatible_hardware_can_still_be_not_ready() -> None:
    report = ready_matcher("llama_cpp").check(
        model("qwen-small-int4"), hardware(llama_cpp=False)
    )

    assert report.compatible is True
    assert report.ready is False
    assert issue_codes(report) == ["runtime_unavailable"]


def test_qwen_can_be_ready_when_execution_dependencies_are_injected() -> None:
    report = ready_matcher("llama_cpp").check(
        model("qwen-small-int4"), hardware()
    )

    assert report.compatible is True
    assert report.ready is True
    assert report.blocking_reasons == []


def test_insufficient_total_memory_blocks_compatibility() -> None:
    report = ready_matcher("llama_cpp").check(
        model("qwen-small-int4"),
        hardware(total_mb=1024, available_mb=512),
    )

    assert report.compatible is False
    assert "insufficient_total_memory" in issue_codes(report)


def test_missing_rvv_is_only_a_warning() -> None:
    report = ready_matcher("llama_cpp").check(
        model("qwen-small-int4"), hardware(rvv=False)
    )

    assert report.compatible is True
    assert report.ready is True
    assert warning_codes(report) == ["rvv_unavailable"]


def test_unknown_rvv_is_only_a_warning() -> None:
    report = ready_matcher("llama_cpp").check(
        model("qwen-small-int4"), hardware(rvv=None)
    )

    assert report.compatible is True
    assert report.ready is True
    assert warning_codes(report) == ["rvv_unknown"]


def test_low_available_memory_is_only_a_warning() -> None:
    report = ready_matcher("llama_cpp").check(
        model("qwen-small-int4"),
        hardware(total_mb=8192, available_mb=1024),
    )

    assert report.compatible is True
    assert report.ready is True
    assert warning_codes(report) == ["low_available_memory"]


def test_excess_recommended_threads_is_only_a_warning() -> None:
    manifest = model("builtin-gemm-int8")
    manifest = manifest.model_copy(
        update={
            "resources": manifest.resources.model_copy(
                update={"recommended_threads": 8}
            )
        }
    )

    report = ready_matcher("builtin").check(
        manifest, hardware(architecture="x86_64", logical_cores=4)
    )

    assert report.ready is True
    assert warning_codes(report) == ["recommended_threads_exceed_cpu"]


def test_builtin_model_does_not_require_an_external_file() -> None:
    report = CompatibilityMatcher(available_adapters={"builtin"}).check(
        model("builtin-gemm-int8"), hardware(architecture="x86_64")
    )

    assert report.compatible is True
    assert report.ready is True
    assert "model_file_missing" not in issue_codes(report)
