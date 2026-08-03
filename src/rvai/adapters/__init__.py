"""Adapter contracts for workload implementations."""

from rvai.adapters.base import WorkloadAdapter
from rvai.adapters.builtin import AdapterError, BenchmarkResult, BuiltinAdapter
from rvai.adapters.onnxruntime import OnnxRuntimeAdapter

__all__ = [
    "AdapterError",
    "BenchmarkResult",
    "BuiltinAdapter",
    "OnnxRuntimeAdapter",
    "WorkloadAdapter",
]
