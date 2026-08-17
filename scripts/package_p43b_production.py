#!/usr/bin/env python3
"""Build or independently verify a formal P4.3B evidence package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rvai.model_pipeline.errors import ModelPipelineError
from rvai.model_pipeline.production_package import (
    build_production_evidence_package,
    verify_production_evidence_package,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build or verify formal MobileNetV2 P4.3B evidence."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--run", required=True, type=Path)
    build.add_argument("--output", required=True, type=Path)
    build.add_argument("--repository", required=True, type=Path)
    verify = commands.add_parser("verify")
    verify.add_argument("--package", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        if arguments.command == "build":
            manifest = build_production_evidence_package(
                arguments.run,
                arguments.output,
                repository=arguments.repository,
            )
        else:
            manifest = verify_production_evidence_package(arguments.package)
    except ModelPipelineError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": manifest.status,
                "production_verified": manifest.production_verified,
                "content_sha256": manifest.content_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
