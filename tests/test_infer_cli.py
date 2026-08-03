import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

import rvai.cli as cli
from rvai.artifacts import ArtifactCache, CachedArtifactMetadata
from rvai.inference import InferenceDependencyError
from rvai.manifest import ModelManifest
from rvai.results import digest_manifest


runner = CliRunner()
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "onnx"
MODEL_PATH = FIXTURE_DIR / "tiny-classifier.onnx"
IMAGE_PATH = FIXTURE_DIR / "red-image.ppm"
pytestmark = pytest.mark.skipif(
    any(
        importlib.util.find_spec(module) is None
        for module in ("numpy", "onnxruntime", "PIL")
    ),
    reason="ONNX optional dependencies are not installed",
)


def manifest_data() -> dict[str, object]:
    model_bytes = MODEL_PATH.read_bytes()
    return {
        "name": "tiny-classifier",
        "display_name": "Tiny Classifier",
        "task": "image_classification",
        "format": "onnx",
        "quantization": "fp32",
        "runtime": "onnxruntime",
        "resources": {"min_memory_mb": 1, "recommended_threads": "auto"},
        "riscv": {"require_rv64": False, "prefer_rvv": False},
        "artifact": {
            "filename": "tiny-classifier.onnx",
            "url": "https://example.com/tiny-classifier.onnx",
            "sha256": hashlib.sha256(model_bytes).hexdigest(),
            "size_bytes": len(model_bytes),
        },
        "input": {
            "type": "image",
            "width": 2,
            "height": 2,
            "layout": "nchw",
            "dtype": "float32",
            "color_space": "rgb",
            "resize": "bilinear",
            "normalize": {
                "scale": 1.0,
                "mean": [0.0, 0.0, 0.0],
                "std": [1.0, 1.0, 1.0],
            },
        },
        "output": {
            "type": "classification",
            "top_k": 3,
            "labels": "rgb",
            "scores": "logits",
        },
    }


def prepare_registry_and_cache(tmp_path: Path) -> tuple[Path, Path]:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    data = manifest_data()
    (models_dir / "tiny-classifier.yaml").write_text(
        yaml.safe_dump(data), encoding="utf-8"
    )
    manifest = ModelManifest.model_validate(data)
    cache = ArtifactCache(root=tmp_path / "cache")
    artifact_path = cache.artifact_path(manifest.name, manifest.artifact)
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(MODEL_PATH.read_bytes())
    cache.write_metadata(
        CachedArtifactMetadata(
            model=manifest.name,
            filename=manifest.artifact.filename,
            source_url=str(manifest.artifact.url),
            sha256=manifest.artifact.sha256,
            size_bytes=manifest.artifact.size_bytes,
            downloaded_at="2026-08-03T12:00:00Z",
            manifest_digest=digest_manifest(manifest),
        )
    )
    return models_dir, cache.root


def cli_environment(models_dir: Path, cache_dir: Path) -> dict[str, str]:
    return {
        "RVAI_MODELS_DIR": str(models_dir),
        "RVAI_CACHE_DIR": str(cache_dir),
    }


def test_infer_executes_real_offline_onnx_fixture(tmp_path) -> None:
    models_dir, cache_dir = prepare_registry_and_cache(tmp_path)

    result = runner.invoke(
        cli.app,
        ["infer", "tiny-classifier", "--input", str(IMAGE_PATH)],
        env=cli_environment(models_dir, cache_dir),
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == "1.0"
    assert payload["predictions"][0]["index"] == 0
    assert payload["execution"]["provider"] == "CPUExecutionProvider"


def test_check_reports_configured_onnx_adapter_as_ready(tmp_path) -> None:
    models_dir, cache_dir = prepare_registry_and_cache(tmp_path)

    result = runner.invoke(
        cli.app,
        ["check", "tiny-classifier"],
        env=cli_environment(models_dir, cache_dir),
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["ready"] is True
    assert "adapter_unavailable" not in {
        issue["code"] for issue in payload["blocking_reasons"]
    }


def test_infer_rejects_tampered_artifact_without_traceback(tmp_path) -> None:
    models_dir, cache_dir = prepare_registry_and_cache(tmp_path)
    (cache_dir / "tiny-classifier" / "tiny-classifier.onnx").write_bytes(b"tampered")

    result = runner.invoke(
        cli.app,
        ["infer", "tiny-classifier", "--input", str(IMAGE_PATH)],
        env=cli_environment(models_dir, cache_dir),
    )

    assert result.exit_code == 1
    assert "Error: Artifact verification failed" in result.output
    assert "Traceback" not in result.output


def test_infer_missing_cache_recommends_pull(tmp_path) -> None:
    models_dir, cache_dir = prepare_registry_and_cache(tmp_path)
    (cache_dir / "tiny-classifier" / "tiny-classifier.onnx").unlink()

    result = runner.invoke(
        cli.app,
        ["infer", "tiny-classifier", "--input", str(IMAGE_PATH)],
        env=cli_environment(models_dir, cache_dir),
    )

    assert result.exit_code == 1
    assert "artifact is not cached with matching metadata" in result.output
    assert "rvai pull tiny-classifier" in result.output
    assert "Traceback" not in result.output


def test_infer_reports_missing_optional_dependencies(monkeypatch, tmp_path) -> None:
    models_dir, cache_dir = prepare_registry_and_cache(tmp_path)

    def missing_dependencies():
        raise InferenceDependencyError(
            'ONNX inference dependencies are missing; install with ".[onnx]"'
        )

    monkeypatch.setattr(cli, "load_onnx_dependencies", missing_dependencies)

    result = runner.invoke(
        cli.app,
        ["infer", "tiny-classifier", "--input", str(IMAGE_PATH)],
        env=cli_environment(models_dir, cache_dir),
    )

    assert result.exit_code == 1
    assert "Error: ONNX inference dependencies are missing" in result.output
    assert "Traceback" not in result.output


def test_infer_reports_missing_input_as_user_error(tmp_path) -> None:
    models_dir, cache_dir = prepare_registry_and_cache(tmp_path)

    result = runner.invoke(
        cli.app,
        ["infer", "tiny-classifier", "--input", str(tmp_path / "missing.jpg")],
        env=cli_environment(models_dir, cache_dir),
    )

    assert result.exit_code == 1
    assert "Error: Cannot read input image" in result.output
    assert "Traceback" not in result.output
