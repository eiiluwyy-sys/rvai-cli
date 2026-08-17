#!/usr/bin/env python3
"""Install one wheel outside the checkout and verify packaged defaults."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

EXPECTED_MODELS = (
    "builtin-gemm-int8",
    "mobilenet-int8",
    "mobilenet-v2-fp32-onnx",
    "qwen-small-int4",
)


def _venv_executable(environment: Path, name: str) -> Path:
    directory = "Scripts" if os.name == "nt" else "bin"
    suffix = ".exe" if os.name == "nt" else ""
    return environment / directory / f"{name}{suffix}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke-test an RVAI wheel outside its source checkout."
    )
    parser.add_argument("wheel", type=Path)
    arguments = parser.parse_args()
    wheel = arguments.wheel.resolve()
    if not wheel.is_file() or wheel.suffix != ".whl":
        parser.error(f"wheel does not exist: {wheel}")

    with tempfile.TemporaryDirectory(prefix="rvai-wheel-smoke-") as temporary:
        root = Path(temporary)
        environment = root / "venv"
        outside_checkout = root / "outside-checkout"
        outside_checkout.mkdir()

        subprocess.run(
            [
                sys.executable,
                "-m",
                "venv",
                str(environment),
            ],
            check=True,
        )
        python = _venv_executable(environment, "python")
        pip = _venv_executable(environment, "pip")
        rvai = _venv_executable(environment, "rvai")
        subprocess.run(
            [
                str(pip),
                "install",
                "--disable-pip-version-check",
                str(wheel),
            ],
            check=True,
        )

        clean_environment = os.environ.copy()
        clean_environment.pop("RVAI_MODELS_DIR", None)
        listed = subprocess.run(
            [str(rvai), "list"],
            cwd=outside_checkout,
            env=clean_environment,
            check=True,
            capture_output=True,
            text=True,
        )
        actual_models = tuple(listed.stdout.splitlines())
        if actual_models != EXPECTED_MODELS:
            raise RuntimeError(
                f"wheel model inventory mismatch: expected {EXPECTED_MODELS}, "
                f"got {actual_models}"
            )

        versions = subprocess.run(
            [
                str(python),
                "-c",
                (
                    "import importlib.metadata, rvai, rvai.model_pipeline; "
                    "print(rvai.__version__); "
                    "print(importlib.metadata.version('rvai-cli'))"
                ),
            ],
            cwd=outside_checkout,
            env=clean_environment,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        if len(versions) != 2 or versions[0] != versions[1]:
            raise RuntimeError(f"wheel version sources disagree: {versions}")

        print(
            f"wheel smoke test passed: version={versions[0]}, "
            f"models={len(actual_models)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
