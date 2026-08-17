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
| Production-labelled P4.3B execution | Awaiting external inputs | Requires the frozen FP32 artifact plus reviewed calibration/evaluation data |
| MobileNetV2 INT8 CLI integration | Deferred | Requires reviewed production artifact and evidence |
| Physical RISC-V execution | Not implemented | QEMU results are explicitly non-representative for performance |
| RVV, NPU, and llama.cpp backends | Not implemented | Outside the active P4.3B scope |

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

The next validation milestone is the first external P4.3B production execution.
It requires the frozen FP32 artifact, at least 1,000 calibration images, and an
independently labelled evaluation Manifest targeting 5,000 images. The dormant
`mobilenet-int8` Manifest must not be activated until the generated artifact and
evidence package have been reviewed independently.

After an accepted package, P4.3C should review and activate the INT8 Manifest
and prove native x86 ONNX Runtime inference. P4.4 then brings up Milk-V Jupiter,
captures its RV64/RVV/software profile, and selects a board runtime based on
measured availability rather than assuming that the x86 ONNX Runtime package is
installable on riscv64.
