# RVAI CLI

RVAI CLI 是面向 RISC-V AI workload 的统一命令行入口。V0.1 使用 Python
实现控制层，负责模型清单校验、模型注册和运行计划生成；高性能 workload
将在后续版本中使用 C/C++ 实现，并由 CLI 通过 adapter 调用。

## 环境要求

- Python 3.10 或 3.11
- Typer
- Pydantic
- PyYAML
- pytest（开发与测试）

## 安装

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## 使用

```bash
rvai list
rvai show qwen-small-int4
rvai run qwen-small-int4 --dry-run
```

默认从当前目录的 `models/` 加载 Manifest。也可通过 `RVAI_MODELS_DIR`
环境变量指定其他模型目录。

V0.1 的 `run` 命令只支持 `--dry-run`，不会启动真实推理或调用尚未实现的
C/C++ workload。

`qwen-small-int4` 和 `mobilenet-int8` 当前仅为模型注册条目，仓库中不包含
对应的 GGUF 或 ONNX 模型文件。运行计划会通过 `requires_model_file` 明确
标识这一需求。

## 测试

```bash
pytest
```

详细范围与验收条件参见 [docs/mvp-v0.1.md](docs/mvp-v0.1.md)。
