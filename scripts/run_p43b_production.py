#!/usr/bin/env python3
"""Run formal P4.3B against pre-provisioned independently labelled data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rvai.model_pipeline.errors import ModelPipelineError
from rvai.model_pipeline.production import run_production_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run formal MobileNetV2 P4.3B production validation."
    )
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--calibration-manifest", required=True, type=Path)
    parser.add_argument("--calibration-root", required=True, type=Path)
    parser.add_argument("--evaluation-manifest", required=True, type=Path)
    parser.add_argument("--evaluation-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    try:
        result = run_production_pipeline(
            arguments.model,
            arguments.calibration_manifest,
            arguments.calibration_root,
            arguments.evaluation_manifest,
            arguments.evaluation_root,
            arguments.output,
            configuration_directory=arguments.config,
            repository=arguments.repository,
        )
    except ModelPipelineError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": result.report.status,
                "production_verified": result.report.production_verified,
                "int8_sha256": result.quantization.artifact.sha256,
                "output": str(arguments.output),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0 if result.report.production_verified else 2


if __name__ == "__main__":
    raise SystemExit(main())
