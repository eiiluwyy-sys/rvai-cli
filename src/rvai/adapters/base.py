"""Base contract for Python, C, or C++ workload adapters."""

from abc import ABC, abstractmethod

from rvai.manifest import ModelManifest


class WorkloadAdapter(ABC):
    """Translate a validated Manifest into an executable command."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the adapter identifier used by Manifest files."""

    @abstractmethod
    def build_command(self, manifest: ModelManifest) -> list[str]:
        """Build the subprocess command for a model workload."""
