# Project status

RVAI CLI is an engineering prototype for reproducible AI workload inspection,
execution, and evidence capture across native and RISC-V environments. This
document describes the current `main` branch; historical phase specifications
remain useful as acceptance records but are not the current feature list.

## Current capabilities

| Area | Status | Boundary |
|---|---|---|
| Model Manifest registry | Available | Strict YAML schemas with user overrides and packaged defaults |
| Hardware detection | Available | Linux CPU, memory, runtime, architecture, and RISC-V ISA metadata |
| Compatibility checks | Available | Separates hardware compatibility from immediate execution readiness |
| Artifact management | Available | HTTP(S), SHA-256, size, metadata, cache, and safe replacement |
| Builtin INT8 GEMM | Available | Scalar backend on native and QEMU riscv64 targets |
| Run evidence | Available | Versioned records, Markdown rendering, and guarded comparisons |
| FP32 ONNX classification | Available | Native CPU, one image, batch one, MobileNetV2 contract |
| P4.3B INT8 production pipeline | Implemented | Class-balanced Manifest preparation, formal runner, failure retention, and deterministic package verification |
| Production-labelled P4.3B execution | Complete | Full 1,000/5,000 report and independently verified evidence package are preserved |
| MobileNetV2 INT8 CLI integration | Implemented | Exact artifact identity, verified local import, readiness check, and native ONNX inference |
| Physical RISC-V vision execution | Complete | Milk-V Jupiter paired FP32/INT8 evidence is preserved |
| Qwen INT4 text generation | Implemented | Pinned official GGUF, llama.cpp adapter, structured generation result, and `rvai generate` |
| SpacemiT llama.cpp build | Implemented | Pinned b10488 source build with RVV and SpacemiT CPU options |
| NPU backend | Not implemented | Future hardware-specific integration |

## Supported development baseline

- CPython 3.10 or 3.11
- CMake 3.16 or newer
- A C++17 compiler
- Optional riscv64 GNU cross-toolchain and QEMU user-mode runtime

The authoritative validation gates are:

```bash
python -m ruff check .
python -m mypy
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
python -m pytest --cov=rvai
./scripts/build-riscv64.sh
python -m build
python scripts/smoke_test_wheel.py dist/*.whl
```

Branch coverage is gated at 85%, below the measured P4.3B baseline of 85.84%.
Future changes should ratchet this threshold upward as production-pipeline
failure paths gain focused tests.

## Next product milestone

The active milestone is physical Milk-V Jupiter validation of the pinned
llama.cpp b10488 build and Qwen2.5-0.5B-Instruct Q4_0 model. The target evidence
captures artifact identity, runtime build identity, prompt hashing, generated
text, total latency, prompt processing rate, and token generation rate from one
repeatable `rvai generate` command.
