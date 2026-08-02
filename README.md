# RVAI CLI

RVAI CLI 是面向 RISC-V AI workload 的统一命令行入口。Python 控制层负责
模型清单校验、硬件探测、兼容性判断和运行调度；首个原生 workload 使用
C++ 实现 scalar INT8 GEMM，并由 CLI 通过 adapter 调用。

## 环境要求

- Python 3.10 或 3.11
- Typer
- Pydantic
- PyYAML
- CMake 3.16 或更高版本
- 支持 C++17 的编译器
- pytest（开发与测试）

## 安装

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

构建原生 workload：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

## 使用

```bash
rvai list
rvai show qwen-small-int4
rvai detect
rvai check builtin-gemm-int8
rvai run qwen-small-int4 --dry-run
rvai run builtin-gemm-int8
rvai run builtin-gemm-int8 --target native
rvai run builtin-gemm-int8 --target qemu-riscv64
```

默认从当前目录的 `models/` 加载 Manifest。也可通过 `RVAI_MODELS_DIR`
环境变量指定其他模型目录。

`builtin-gemm-int8` 会执行 `build/rvai-bench`，输出正确性、平均与 P95
延迟、吞吐量及矩阵内存占用的 JSON。若二进制位于其他目录，可通过
`RVAI_BENCH_BIN` 指定完整路径。其他模型当前仍只支持 `--dry-run`。

`qwen-small-int4` 和 `mobilenet-int8` 当前仅为模型注册条目，仓库中不包含
对应的 GGUF 或 ONNX 模型文件。运行计划会通过 `requires_model_file` 明确
标识这一需求。

## 测试

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
python -m pytest
```

详细范围与验收条件参见 [docs/mvp-v0.1.md](docs/mvp-v0.1.md)。
