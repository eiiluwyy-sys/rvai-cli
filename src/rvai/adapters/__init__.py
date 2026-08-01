"""Adapter contracts for workload implementations."""

from rvai.adapters.base import WorkloadAdapter
from rvai.adapters.builtin import AdapterError, BenchmarkResult, BuiltinAdapter

__all__ = [
    "AdapterError",
    "BenchmarkResult",
    "BuiltinAdapter",
    "WorkloadAdapter",
]
