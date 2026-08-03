import json

import pytest
from typer.testing import CliRunner

import rvai.cli as cli
import rvai.inference.dependencies as dependency_module
import rvai.hardware.runtime as runtime_module
from rvai.hardware.runtime import RuntimeProbe
from rvai.inference import InferenceDependencyError, load_onnx_dependencies


runner = CliRunner()


def test_dependency_loader_reports_all_missing_packages(monkeypatch) -> None:
    def missing_import(name: str):
        raise ImportError(name)

    monkeypatch.setattr(dependency_module.importlib, "import_module", missing_import)

    with pytest.raises(InferenceDependencyError) as error:
        load_onnx_dependencies()

    assert "numpy, onnxruntime, Pillow" in str(error.value)
    assert 'pip install -e ".[onnx]"' in str(error.value)


def test_non_inference_cli_does_not_import_optional_packages(monkeypatch) -> None:
    def unexpected_import(name: str):
        raise AssertionError(f"optional import attempted: {name}")

    monkeypatch.setattr(dependency_module.importlib, "import_module", unexpected_import)

    result = runner.invoke(cli.app, ["list"])

    assert result.exit_code == 0
    assert "builtin-gemm-int8" in result.stdout


def test_runtime_probe_reports_onnxruntime_version_without_importing_it(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        runtime_module.importlib.metadata,
        "version",
        lambda distribution: "1.2.3",
    )
    probe = RuntimeProbe(
        environ={},
        which=lambda executable: None,
        find_spec=lambda module: object() if module == "onnxruntime" else None,
    )

    status = probe.detect().onnxruntime

    assert status.available is True
    assert status.version == "1.2.3"


def test_detect_still_serializes_runtime_status() -> None:
    result = runner.invoke(cli.app, ["detect"])

    assert result.exit_code == 0
    assert "available" in json.loads(result.stdout)["runtimes"]["onnxruntime"]
