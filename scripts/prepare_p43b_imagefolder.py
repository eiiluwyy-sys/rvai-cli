#!/usr/bin/env python3
"""Prepare formal P4.3B Manifests from numeric ImageFolder data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rvai.model_pipeline.dataset_prepare import (
    DatasetPreparationError,
    prepare_numeric_imagefolder_manifests,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Select one calibration and five evaluation images per numeric "
            "ImageNet class in deterministic filename order."
        )
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--license", dest="license_description", required=True)
    arguments = parser.parse_args()
    try:
        prepared = prepare_numeric_imagefolder_manifests(
            arguments.root,
            arguments.output,
            dataset_name=arguments.dataset_name,
            dataset_version=arguments.dataset_version,
            provenance=arguments.provenance,
            license_description=arguments.license_description,
        )
    except DatasetPreparationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            prepared.record.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
