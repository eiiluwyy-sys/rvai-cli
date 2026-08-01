import pytest
from pydantic import ValidationError

from rvai.hardware.schema import MemoryInfo, RiscVInfo


def test_riscv_extensions_have_independent_defaults() -> None:
    first = RiscVInfo(is_riscv=False)
    second = RiscVInfo(is_riscv=False)

    first.extensions.append("v")

    assert second.extensions == []


def test_memory_rejects_negative_available_memory() -> None:
    with pytest.raises(ValidationError):
        MemoryInfo(total_mb=1024, available_mb=-1)


def test_riscv_profile_serializes_unknown_values_as_null() -> None:
    riscv = RiscVInfo(is_riscv=True, xlen=64)

    assert riscv.model_dump() == {
        "is_riscv": True,
        "xlen": 64,
        "isa": None,
        "extensions": [],
        "rvv": None,
    }
