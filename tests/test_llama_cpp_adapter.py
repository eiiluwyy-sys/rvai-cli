import subprocess
from pathlib import Path

import pytest

from rvai.adapters import LlamaCppAdapter
from rvai.generation import GenerationError
from rvai.registry import ModelRegistry

MODELS_DIR = Path(__file__).parents[1] / "models"
MODEL_PATH = Path("/models/qwen.gguf")
PERF_OUTPUT = """
build: 10488 (9d77fa17) with GNU 13.2.0 for Linux riscv64
llama_perf_context_print: prompt eval time = 100.00 ms / 10 tokens (10.00 ms per token, 100.00 tokens per second)
llama_perf_context_print: eval time = 200.00 ms / 4 runs (50.00 ms per token, 20.00 tokens per second)
"""


def manifest():
    return ModelRegistry(MODELS_DIR).get("qwen-small-int4")


def test_adapter_builds_shell_free_offline_single_turn_command() -> None:
    adapter = LlamaCppAdapter(
        executable="/opt/llama-cli", environ={"LANG": "C.UTF-8"}
    )

    command, parameters = adapter.build_command(
        manifest(),
        model_path=MODEL_PATH,
        prompt="介绍 RISC-V",
        system_prompt="简洁回答",
    )

    assert command[0] == "/opt/llama-cli"
    assert command[command.index("--model") + 1] == str(MODEL_PATH)
    assert command[command.index("--prompt") + 1] == "介绍 RISC-V"
    assert command[command.index("--system-prompt") + 1] == "简洁回答"
    assert "--offline" in command
    assert "--single-turn" in command
    assert "--simple-io" in command
    assert parameters.threads == 4
    assert parameters.batch_size == 128


def test_adapter_parses_response_version_and_perf_metrics() -> None:
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            returncode=0,
            stdout="RISC-V 是一种开放指令集架构。\n",
            stderr=PERF_OUTPUT,
        )

    ticks = iter([10.0, 10.5])
    result = LlamaCppAdapter(
        executable="/opt/llama-cli",
        environ={"LANG": "C.UTF-8"},
        runner=runner,
        clock=lambda: next(ticks),
    ).generate(
        manifest(),
        model_path=MODEL_PATH,
        prompt="介绍 RISC-V",
    )

    assert result.status == "success"
    assert result.runtime == "llama_cpp"
    assert result.response == "RISC-V 是一种开放指令集架构。"
    assert result.prompt.characters == len("介绍 RISC-V")
    assert result.prompt.sha256 != ""
    assert result.metrics.total_ms == 500.0
    assert result.metrics.prompt_tokens == 10
    assert result.metrics.prompt_tokens_per_second == 100.0
    assert result.metrics.generated_tokens == 4
    assert result.metrics.tokens_per_second == 20.0
    assert result.execution.runtime_version == "b10488 (9d77fa17)"
    assert calls[0][1]["env"] == {"LANG": "C.UTF-8"}
    assert calls[0][1]["timeout"] == 600.0


def test_adapter_reports_missing_runtime() -> None:
    adapter = LlamaCppAdapter(environ={}, which=lambda executable: None)

    with pytest.raises(GenerationError, match="runtime was not found"):
        adapter.build_command(
            manifest(), model_path=MODEL_PATH, prompt="hello"
        )


def test_adapter_rejects_empty_prompt_and_invalid_limits() -> None:
    adapter = LlamaCppAdapter(executable="/opt/llama-cli", environ={})

    with pytest.raises(GenerationError, match="must not be empty"):
        adapter.build_command(manifest(), model_path=MODEL_PATH, prompt="  ")
    with pytest.raises(GenerationError, match="Invalid generation parameters"):
        adapter.build_command(
            manifest(), model_path=MODEL_PATH, prompt="hello", max_tokens=0
        )


def test_adapter_hides_traceback_for_runtime_failure() -> None:
    def runner(command, **kwargs):
        return subprocess.CompletedProcess(
            command, returncode=2, stdout="", stderr="invalid model"
        )

    adapter = LlamaCppAdapter(
        executable="/opt/llama-cli", environ={}, runner=runner
    )

    with pytest.raises(GenerationError, match="exit code 2: invalid model"):
        adapter.generate(manifest(), model_path=MODEL_PATH, prompt="hello")


def test_adapter_reports_timeout() -> None:
    def runner(command, **kwargs):
        raise subprocess.TimeoutExpired(command, timeout=1)

    adapter = LlamaCppAdapter(
        executable="/opt/llama-cli", environ={}, runner=runner
    )

    with pytest.raises(GenerationError, match="timed out after 1 seconds"):
        adapter.generate(
            manifest(),
            model_path=MODEL_PATH,
            prompt="hello",
            timeout_seconds=1,
        )
