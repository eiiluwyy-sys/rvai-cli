import hashlib
import json
import threading
from collections.abc import Iterator
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

import rvai.cli as cli
from rvai.hardware.schema import (
    CpuInfo,
    HardwareProfile,
    MemoryInfo,
    PlatformInfo,
    RiscVInfo,
    RuntimeInfo,
    RuntimeStatus,
)


runner = CliRunner()
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "artifacts"
TINY_MODEL = FIXTURE_DIR / "tiny-model.bin"


class CountingHttpServer(ThreadingHTTPServer):
    request_count: int = 0


@pytest.fixture
def artifact_server() -> Iterator[CountingHttpServer]:
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, directory=str(FIXTURE_DIR), **kwargs)

        def do_GET(self) -> None:
            self.server.request_count += 1
            super().do_GET()

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = CountingHttpServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def server_url(server: CountingHttpServer, filename: str = "tiny-model.bin") -> str:
    host, port = server.server_address
    return f"http://{host}:{port}/{filename}"


def write_models(
    models_dir: Path,
    url: str,
    *,
    sha256: str | None = None,
) -> None:
    payload = TINY_MODEL.read_bytes()
    models_dir.mkdir(parents=True)
    artifact_manifest = {
        "name": "tiny-onnx",
        "display_name": "Tiny ONNX",
        "task": "image_classification",
        "format": "onnx",
        "quantization": "fp32",
        "runtime": "onnxruntime",
        "resources": {"min_memory_mb": 1, "recommended_threads": "auto"},
        "riscv": {"require_rv64": False, "prefer_rvv": False},
        "artifact": {
            "filename": "tiny-model.bin",
            "url": url,
            "sha256": sha256 or hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
            "media_type": "application/octet-stream",
        },
    }
    builtin_manifest = {
        "name": "builtin-gemm-int8",
        "display_name": "Builtin INT8 GEMM",
        "task": "benchmark",
        "format": "builtin",
        "quantization": "int8",
        "runtime": "builtin",
        "resources": {"min_memory_mb": 128, "recommended_threads": "auto"},
        "riscv": {"require_rv64": False, "prefer_rvv": True},
    }
    (models_dir / "tiny-onnx.yaml").write_text(
        yaml.safe_dump(artifact_manifest), encoding="utf-8"
    )
    (models_dir / "builtin-gemm-int8.yaml").write_text(
        yaml.safe_dump(builtin_manifest), encoding="utf-8"
    )


def cli_env(models_dir: Path, cache_dir: Path) -> dict[str, str]:
    return {
        "RVAI_MODELS_DIR": str(models_dir),
        "RVAI_CACHE_DIR": str(cache_dir),
    }


def hardware_profile(path: Path) -> None:
    profile = HardwareProfile(
        platform=PlatformInfo(os="linux", kernel="6.8", architecture="x86_64"),
        cpu=CpuInfo(logical_cores=4),
        memory=MemoryInfo(total_mb=4096, available_mb=3072),
        riscv=RiscVInfo(is_riscv=False),
        runtimes=RuntimeInfo(
            builtin=RuntimeStatus(available=True),
            llama_cpp=RuntimeStatus(available=False),
            onnxruntime=RuntimeStatus(available=True),
        ),
    )
    path.write_text(profile.model_dump_json(), encoding="utf-8")


def test_pull_downloads_from_local_http_server(artifact_server, tmp_path) -> None:
    models_dir = tmp_path / "models"
    cache_dir = tmp_path / "cache"
    write_models(models_dir, server_url(artifact_server))

    result = runner.invoke(
        cli.app,
        ["pull", "tiny-onnx", "--cache-dir", str(cache_dir)],
        env={"RVAI_MODELS_DIR": str(models_dir)},
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == "1.0"
    assert payload["status"] == "downloaded"
    assert payload["model"] == "tiny-onnx"
    assert payload["verified"] is True
    assert payload["size_bytes"] == len(TINY_MODEL.read_bytes())
    assert Path(payload["path"]).read_bytes() == TINY_MODEL.read_bytes()
    assert artifact_server.request_count == 1


def test_pull_reuses_valid_cache_without_network(artifact_server, tmp_path) -> None:
    models_dir = tmp_path / "models"
    cache_dir = tmp_path / "cache"
    write_models(models_dir, server_url(artifact_server))
    environment = cli_env(models_dir, cache_dir)

    first = runner.invoke(cli.app, ["pull", "tiny-onnx"], env=environment)
    second = runner.invoke(cli.app, ["pull", "tiny-onnx"], env=environment)

    assert first.exit_code == second.exit_code == 0
    assert json.loads(second.stdout)["status"] == "already-cached"
    assert artifact_server.request_count == 1


def test_pull_force_replaces_invalid_cache(artifact_server, tmp_path) -> None:
    models_dir = tmp_path / "models"
    cache_dir = tmp_path / "cache"
    write_models(models_dir, server_url(artifact_server))
    environment = cli_env(models_dir, cache_dir)
    first = runner.invoke(cli.app, ["pull", "tiny-onnx"], env=environment)
    path = Path(json.loads(first.stdout)["path"])
    path.write_bytes(b"tampered")

    refused = runner.invoke(cli.app, ["pull", "tiny-onnx"], env=environment)
    replaced = runner.invoke(
        cli.app, ["pull", "tiny-onnx", "--force"], env=environment
    )

    assert refused.exit_code == 1
    assert "Use --force" in refused.output
    assert replaced.exit_code == 0
    assert Path(json.loads(replaced.stdout)["path"]).read_bytes() == TINY_MODEL.read_bytes()


def test_pull_rejects_unknown_and_undeclared_models(artifact_server, tmp_path) -> None:
    models_dir = tmp_path / "models"
    write_models(models_dir, server_url(artifact_server))
    environment = cli_env(models_dir, tmp_path / "cache")

    unknown = runner.invoke(cli.app, ["pull", "missing"], env=environment)
    undeclared = runner.invoke(
        cli.app, ["pull", "builtin-gemm-int8"], env=environment
    )

    assert unknown.exit_code == 1
    assert "Error: Unknown model 'missing'" in unknown.output
    assert undeclared.exit_code == 1
    assert "does not declare a downloadable artifact" in undeclared.output
    assert "Traceback" not in unknown.output + undeclared.output
    assert artifact_server.request_count == 0


def test_pull_reports_hash_mismatch_without_publishing_file(
    artifact_server, tmp_path
) -> None:
    models_dir = tmp_path / "models"
    cache_dir = tmp_path / "cache"
    write_models(models_dir, server_url(artifact_server), sha256="0" * 64)

    result = runner.invoke(
        cli.app,
        ["pull", "tiny-onnx"],
        env=cli_env(models_dir, cache_dir),
    )

    assert result.exit_code == 1
    assert "Error: Artifact SHA-256 mismatch for 'tiny-onnx'" in result.output
    assert "Traceback" not in result.output
    assert not (cache_dir / "tiny-onnx" / "tiny-model.bin").exists()


def test_pull_reports_http_404_without_server_body(artifact_server, tmp_path) -> None:
    models_dir = tmp_path / "models"
    write_models(models_dir, server_url(artifact_server, "missing.bin"))

    result = runner.invoke(
        cli.app,
        ["pull", "tiny-onnx"],
        env=cli_env(models_dir, tmp_path / "cache"),
    )

    assert result.exit_code == 1
    assert "Error: Failed to download artifact for 'tiny-onnx'" in result.output
    assert "404" in result.output
    assert "<!DOCTYPE" not in result.output
    assert "Traceback" not in result.output


def test_show_reports_local_status_without_network(artifact_server, tmp_path) -> None:
    models_dir = tmp_path / "models"
    cache_dir = tmp_path / "cache"
    write_models(models_dir, server_url(artifact_server))
    environment = cli_env(models_dir, cache_dir)

    before = runner.invoke(cli.app, ["show", "tiny-onnx"], env=environment)
    pulled = runner.invoke(cli.app, ["pull", "tiny-onnx"], env=environment)
    after = runner.invoke(cli.app, ["show", "tiny-onnx"], env=environment)

    assert before.exit_code == pulled.exit_code == after.exit_code == 0
    assert json.loads(before.stdout)["artifact"]["cached"] is False
    status = json.loads(after.stdout)["artifact"]
    assert status["declared"] is True
    assert status["cached"] is True
    assert status["verified"] is True
    assert Path(status["path"]).is_file()
    assert artifact_server.request_count == 1


def test_show_treats_missing_or_mismatched_metadata_as_unverified(
    artifact_server, tmp_path
) -> None:
    models_dir = tmp_path / "models"
    cache_dir = tmp_path / "cache"
    write_models(models_dir, server_url(artifact_server))
    environment = cli_env(models_dir, cache_dir)
    pulled = runner.invoke(cli.app, ["pull", "tiny-onnx"], env=environment)
    assert pulled.exit_code == 0
    metadata = cache_dir / "tiny-onnx" / "artifact.json"
    metadata.write_text("{invalid", encoding="utf-8")

    shown = runner.invoke(cli.app, ["show", "tiny-onnx"], env=environment)

    assert shown.exit_code == 0
    status = json.loads(shown.stdout)["artifact"]
    assert status["cached"] is True
    assert status["verified"] is False


def test_check_uses_trusted_cache_metadata_without_network(
    artifact_server, tmp_path
) -> None:
    models_dir = tmp_path / "models"
    cache_dir = tmp_path / "cache"
    profile_path = tmp_path / "profile.json"
    write_models(models_dir, server_url(artifact_server))
    hardware_profile(profile_path)
    environment = cli_env(models_dir, cache_dir)

    before = runner.invoke(
        cli.app,
        ["check", "tiny-onnx", "--profile", str(profile_path)],
        env=environment,
    )
    pulled = runner.invoke(cli.app, ["pull", "tiny-onnx"], env=environment)
    after = runner.invoke(
        cli.app,
        ["check", "tiny-onnx", "--profile", str(profile_path)],
        env=environment,
    )

    before_codes = {item["code"] for item in json.loads(before.stdout)["blocking_reasons"]}
    after_codes = {item["code"] for item in json.loads(after.stdout)["blocking_reasons"]}
    assert "model_file_missing" in before_codes
    assert "model_file_missing" not in after_codes
    assert "adapter_unavailable" in after_codes
    assert json.loads(after.stdout)["ready"] is False
    assert pulled.exit_code == 0
    assert artifact_server.request_count == 1
