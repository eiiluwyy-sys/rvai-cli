import hashlib
import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

import rvai.cli as cli
from rvai.artifacts import ArtifactCache, CachedArtifactMetadata
from rvai.generation import GenerationError, GenerationResult
from rvai.manifest import ModelManifest
from rvai.results import digest_manifest

runner = CliRunner()


def manifest_data(model_bytes: bytes) -> dict[str, object]:
    return {
        "name": "tiny-chat-int4",
        "display_name": "Tiny Chat INT4",
        "task": "chat",
        "format": "gguf",
        "quantization": "int4",
        "runtime": "llama_cpp",
        "resources": {"min_memory_mb": 1, "recommended_threads": 2},
        "riscv": {"require_rv64": False, "prefer_rvv": False},
        "artifact": {
            "filename": "tiny-chat.gguf",
            "url": "https://example.com/tiny-chat.gguf",
            "sha256": hashlib.sha256(model_bytes).hexdigest(),
            "size_bytes": len(model_bytes),
        },
        "generation": {
            "context_size": 512,
            "max_tokens": 16,
            "temperature": 0.2,
            "seed": 42,
            "batch_size": 32,
            "ubatch_size": 32,
        },
    }


def prepare_environment(tmp_path: Path) -> tuple[dict[str, str], Path]:
    model_bytes = b"GGUF-test-model"
    data = manifest_data(model_bytes)
    manifest = ModelManifest.model_validate(data)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "tiny-chat-int4.yaml").write_text(
        yaml.safe_dump(data), encoding="utf-8"
    )
    cache = ArtifactCache(root=tmp_path / "cache")
    artifact_path = cache.artifact_path(manifest.name, manifest.artifact)
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(model_bytes)
    cache.write_metadata(
        CachedArtifactMetadata(
            model=manifest.name,
            filename=manifest.artifact.filename,
            source_url=str(manifest.artifact.url),
            sha256=manifest.artifact.sha256,
            size_bytes=len(model_bytes),
            downloaded_at="2026-08-19T12:00:00Z",
            manifest_digest=digest_manifest(manifest),
        )
    )
    executable = tmp_path / "llama-cli"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    environment = {
        "RVAI_MODELS_DIR": str(models_dir),
        "RVAI_CACHE_DIR": str(cache.root),
        "RVAI_LLAMA_CPP_BIN": str(executable),
    }
    return environment, artifact_path


def generation_result() -> GenerationResult:
    return GenerationResult.model_validate(
        {
            "model": "tiny-chat-int4",
            "prompt": {"sha256": "a" * 64, "characters": 5},
            "parameters": {
                "max_tokens": 8,
                "context_size": 512,
                "threads": 2,
                "batch_size": 32,
                "ubatch_size": 32,
                "temperature": 0.1,
                "seed": 7,
            },
            "response": "你好，我是本地模型。",
            "metrics": {
                "total_ms": 100.0,
                "prompt_tokens": 3,
                "generated_tokens": 8,
                "tokens_per_second": 12.5,
            },
            "execution": {
                "runtime_version": "b10488",
                "executable": "/opt/llama-cli",
            },
        }
    )


def test_generate_outputs_structured_result(monkeypatch, tmp_path) -> None:
    environment, artifact_path = prepare_environment(tmp_path)

    class FakeAdapter:
        @classmethod
        def supports(cls, manifest):
            return manifest.runtime == "llama_cpp"

        def generate(self, manifest, **kwargs):
            assert manifest.name == "tiny-chat-int4"
            assert kwargs["model_path"] == artifact_path
            assert kwargs["prompt"] == "你好"
            assert kwargs["max_tokens"] == 8
            assert kwargs["temperature"] == 0.1
            assert kwargs["seed"] == 7
            return generation_result()

    monkeypatch.setattr(cli, "LlamaCppAdapter", FakeAdapter)
    result = runner.invoke(
        cli.app,
        [
            "generate",
            "tiny-chat-int4",
            "--prompt",
            "你好",
            "--max-tokens",
            "8",
            "--temperature",
            "0.1",
            "--seed",
            "7",
        ],
        env=environment,
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "success"
    assert payload["runtime"] == "llama_cpp"
    assert payload["response"] == "你好，我是本地模型。"
    assert payload["metrics"]["tokens_per_second"] == 12.5


def test_generate_missing_artifact_recommends_pull(monkeypatch, tmp_path) -> None:
    environment, artifact_path = prepare_environment(tmp_path)
    artifact_path.unlink()

    class FakeAdapter:
        @classmethod
        def supports(cls, manifest):
            return manifest.runtime == "llama_cpp"

    monkeypatch.setattr(cli, "LlamaCppAdapter", FakeAdapter)
    result = runner.invoke(
        cli.app,
        ["generate", "tiny-chat-int4", "--prompt", "hello"],
        env=environment,
    )

    assert result.exit_code == 1
    assert "rvai pull tiny-chat-int4" in result.output
    assert "Traceback" not in result.output


def test_generate_hides_runtime_traceback(monkeypatch, tmp_path) -> None:
    environment, _ = prepare_environment(tmp_path)

    class FailingAdapter:
        @classmethod
        def supports(cls, manifest):
            return manifest.runtime == "llama_cpp"

        def generate(self, manifest, **kwargs):
            raise GenerationError("llama.cpp returned invalid output")

    monkeypatch.setattr(cli, "LlamaCppAdapter", FailingAdapter)
    result = runner.invoke(
        cli.app,
        ["generate", "tiny-chat-int4", "--prompt", "hello"],
        env=environment,
    )

    assert result.exit_code == 1
    assert "Error: llama.cpp returned invalid output" in result.output
    assert "Traceback" not in result.output
