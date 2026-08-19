"""llama.cpp subprocess adapter for verified GGUF chat models."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from time import perf_counter

from pydantic import ValidationError

from rvai.generation import (
    GenerationError,
    GenerationExecution,
    GenerationMetrics,
    GenerationParameters,
    GenerationResult,
    PromptInfo,
)
from rvai.manifest import ModelManifest, TextGenerationSpec

Runner = Callable[..., subprocess.CompletedProcess[str]]

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_BUILD_PATTERN = re.compile(
    r"(?:build|version)\s*[:=]\s*(?:b)?(?P<number>\d+)"
    r"(?:\s*\((?P<commit>[0-9a-f]{7,40})\))?",
    re.IGNORECASE,
)
_PROMPT_PERF_PATTERN = re.compile(
    r"prompt eval time\s*=\s*(?P<ms>[0-9.]+)\s*ms\s*/\s*"
    r"(?P<tokens>\d+)\s*tokens?.*?\(\s*(?:[0-9.]+\s*ms per token,\s*)?"
    r"(?P<rate>[0-9.]+)\s*tokens per second\s*\)",
    re.IGNORECASE,
)
_GENERATION_PERF_PATTERN = re.compile(
    r"(?<!prompt )eval time\s*=\s*(?P<ms>[0-9.]+)\s*ms\s*/\s*"
    r"(?P<tokens>\d+)\s*(?:runs?|tokens?).*?\(\s*"
    r"(?:[0-9.]+\s*ms per token,\s*)?"
    r"(?P<rate>[0-9.]+)\s*tokens per second\s*\)",
    re.IGNORECASE,
)


class LlamaCppAdapter:
    """Run one verified GGUF model through a local ``llama-cli`` binary."""

    def __init__(
        self,
        *,
        executable: Path | str | None = None,
        environ: Mapping[str, str] | None = None,
        which: Callable[[str], str | None] = shutil.which,
        runner: Runner = subprocess.run,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self.environ = os.environ if environ is None else environ
        configured = executable or self.environ.get("RVAI_LLAMA_CPP_BIN")
        resolved = str(configured) if configured is not None else which("llama-cli")
        self.executable = resolved
        self.runner = runner
        self.clock = clock

    @property
    def name(self) -> str:
        return "llama_cpp"

    @classmethod
    def supports(cls, manifest: ModelManifest) -> bool:
        """Report whether the Manifest has the supported chat contract."""

        return (
            manifest.runtime == "llama_cpp"
            and manifest.format == "gguf"
            and manifest.task == "chat"
            and manifest.quantization == "int4"
            and manifest.artifact is not None
        )

    def build_command(
        self,
        manifest: ModelManifest,
        *,
        model_path: Path,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        context_size: int | None = None,
        threads: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
    ) -> tuple[list[str], GenerationParameters]:
        """Build a shell-free, offline, single-turn llama.cpp command."""

        if not self.supports(manifest):
            raise GenerationError(
                f"LlamaCppAdapter does not support model {manifest.name}; "
                "expected an INT4 GGUF chat Manifest"
            )
        if self.executable is None:
            raise GenerationError(
                "llama.cpp runtime was not found; install llama-cli or configure "
                "RVAI_LLAMA_CPP_BIN"
            )
        cleaned_prompt = prompt.strip()
        if not cleaned_prompt:
            raise GenerationError("Prompt must not be empty")
        if "\x00" in prompt or (system_prompt is not None and "\x00" in system_prompt):
            raise GenerationError("Prompt text must not contain NUL characters")

        defaults = manifest.generation or TextGenerationSpec()
        selected_threads = (
            self._default_threads(manifest) if threads is None else threads
        )
        try:
            parameters = GenerationParameters(
                max_tokens=(
                    defaults.max_tokens if max_tokens is None else max_tokens
                ),
                context_size=(
                    defaults.context_size if context_size is None else context_size
                ),
                threads=selected_threads,
                batch_size=defaults.batch_size,
                ubatch_size=defaults.ubatch_size,
                temperature=(
                    defaults.temperature if temperature is None else temperature
                ),
                seed=defaults.seed if seed is None else seed,
            )
        except ValidationError as exc:
            raise GenerationError(f"Invalid generation parameters: {exc}") from exc

        command = [
            self.executable,
            "--model",
            str(model_path),
            "--prompt",
            cleaned_prompt,
            "--predict",
            str(parameters.max_tokens),
            "--ctx-size",
            str(parameters.context_size),
            "--threads",
            str(parameters.threads),
            "--threads-batch",
            str(parameters.threads),
            "--batch-size",
            str(parameters.batch_size),
            "--ubatch-size",
            str(parameters.ubatch_size),
            "--temperature",
            str(parameters.temperature),
            "--seed",
            str(parameters.seed),
            "--conversation",
            "--single-turn",
            "--simple-io",
            "--no-display-prompt",
            "--color",
            "off",
            "--log-colors",
            "off",
            "--perf",
            "--show-timings",
            "--offline",
            "--no-context-shift",
        ]
        if system_prompt is not None and system_prompt.strip():
            command.extend(["--system-prompt", system_prompt.strip()])
        return command, parameters

    def generate(
        self,
        manifest: ModelManifest,
        *,
        model_path: Path,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        context_size: int | None = None,
        threads: int | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        timeout_seconds: float = 600.0,
    ) -> GenerationResult:
        """Execute one single-turn generation and return validated JSON data."""

        if timeout_seconds <= 0:
            raise GenerationError("Generation timeout must be positive")
        command, parameters = self.build_command(
            manifest,
            model_path=model_path,
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            context_size=context_size,
            threads=threads,
            temperature=temperature,
            seed=seed,
        )
        started = self.clock()
        try:
            completed = self.runner(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=dict(self.environ),
            )
        except OSError as exc:
            detail = exc.strerror or str(exc)
            raise GenerationError(f"Cannot execute llama.cpp: {detail}") from exc
        except subprocess.TimeoutExpired as exc:
            raise GenerationError(
                f"llama.cpp generation timed out after {timeout_seconds:g} seconds"
            ) from exc
        total_ms = (self.clock() - started) * 1000.0

        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            if len(detail) > 4000:
                detail = detail[-4000:]
            raise GenerationError(
                f"llama.cpp failed with exit code {completed.returncode}: "
                f"{detail or 'no error output'}"
            )

        response = _ANSI_ESCAPE.sub("", completed.stdout).strip()
        if not response:
            raise GenerationError("llama.cpp returned an empty response")
        prompt_metrics = _parse_perf(completed.stderr, _PROMPT_PERF_PATTERN)
        generation_metrics = _parse_perf(
            completed.stderr, _GENERATION_PERF_PATTERN
        )
        return GenerationResult(
            model=manifest.name,
            prompt=_prompt_info(
                prompt.strip(),
                system_prompt.strip() if system_prompt is not None else None,
            ),
            parameters=parameters,
            response=response,
            metrics=GenerationMetrics(
                total_ms=total_ms,
                prompt_tokens=_metric_int(prompt_metrics, "tokens"),
                prompt_eval_ms=_metric_float(prompt_metrics, "ms"),
                prompt_tokens_per_second=_metric_float(prompt_metrics, "rate"),
                generated_tokens=_metric_int(generation_metrics, "tokens"),
                generation_ms=_metric_float(generation_metrics, "ms"),
                tokens_per_second=_metric_float(generation_metrics, "rate"),
            ),
            execution=GenerationExecution(
                runtime_version=_runtime_version(completed.stderr),
                executable=command[0],
            ),
        )

    @staticmethod
    def _default_threads(manifest: ModelManifest) -> int:
        recommended = manifest.resources.recommended_threads
        if isinstance(recommended, int):
            return recommended
        return max(1, os.cpu_count() or 1)


def _prompt_info(prompt: str, system_prompt: str | None) -> PromptInfo:
    system_text = system_prompt or ""
    return PromptInfo(
        sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        characters=len(prompt),
        system_sha256=(
            hashlib.sha256(system_text.encode("utf-8")).hexdigest()
            if system_text
            else None
        ),
        system_characters=len(system_text),
    )


def _parse_perf(stderr: str, pattern: re.Pattern[str]) -> dict[str, str] | None:
    match = pattern.search(_ANSI_ESCAPE.sub("", stderr))
    return match.groupdict() if match is not None else None


def _metric_int(values: dict[str, str] | None, key: str) -> int | None:
    return int(values[key]) if values is not None else None


def _metric_float(values: dict[str, str] | None, key: str) -> float | None:
    return float(values[key]) if values is not None else None


def _runtime_version(stderr: str) -> str:
    match = _BUILD_PATTERN.search(_ANSI_ESCAPE.sub("", stderr))
    if match is None:
        return "unknown"
    commit = match.group("commit")
    version = f"b{match.group('number')}"
    return f"{version} ({commit})" if commit else version
