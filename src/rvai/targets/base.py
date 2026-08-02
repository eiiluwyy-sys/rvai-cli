"""Execution-target contracts for launching native workload binaries."""

from abc import ABC, abstractmethod
from pathlib import Path


class TargetError(RuntimeError):
    """Raised when an execution target is not configured or available."""


class ExecutionTarget(ABC):
    """Describe where a workload binary lives and how it is launched."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the stable target identifier used by the CLI."""

    @property
    @abstractmethod
    def executable(self) -> Path:
        """Return the workload binary that belongs to this target."""

    @abstractmethod
    def build_command(
        self,
        executable: Path,
        arguments: list[str],
    ) -> list[str]:
        """Validate the environment and build the full process command."""
