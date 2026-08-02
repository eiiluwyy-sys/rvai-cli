"""Execution targets exposed by ``rvai run --target``."""

from enum import Enum

from rvai.targets.base import ExecutionTarget, TargetError
from rvai.targets.native import NativeTarget
from rvai.targets.qemu_user import QemuRiscv64Target


class TargetName(str, Enum):
    NATIVE = "native"
    QEMU_RISCV64 = "qemu-riscv64"


def create_target(name: TargetName) -> ExecutionTarget:
    """Create an execution target without validating its environment yet."""

    if name is TargetName.NATIVE:
        return NativeTarget()
    return QemuRiscv64Target()


__all__ = [
    "ExecutionTarget",
    "NativeTarget",
    "QemuRiscv64Target",
    "TargetError",
    "TargetName",
    "create_target",
]
