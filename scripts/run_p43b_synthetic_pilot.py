#!/usr/bin/env python3
"""Run the non-production P4.3B synthetic consistency pilot."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from rvai.model_pipeline.io import canonical_json_text
from rvai.model_pipeline.pilot import run_synthetic_proxy_pilot


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the provisional MobileNetV2 synthetic consistency pilot."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--configuration",
        type=Path,
        default=Path("model-pipeline/mobilenet-v2"),
    )
    parser.add_argument("--seed", type=int, default=4302)
    parser.add_argument("--calibration-samples", type=int)
    parser.add_argument("--evaluation-samples", type=int)
    arguments = parser.parse_args()

    started = time.monotonic()
    result = run_synthetic_proxy_pilot(
        arguments.model,
        arguments.output,
        configuration_directory=arguments.configuration,
        calibration_sample_count=arguments.calibration_samples,
        evaluation_sample_count=arguments.evaluation_samples,
        seed=arguments.seed,
    )
    elapsed = time.monotonic() - started
    print(canonical_json_text(result.report))
    print(f"Proxy pilot elapsed seconds: {elapsed:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
