#!/usr/bin/env python3
"""Build or independently verify a non-production P4.3B evidence package."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from rvai.model_pipeline.errors import ModelPipelineError
from rvai.model_pipeline.package import (
    build_evidence_package,
    verify_evidence_package,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build or verify a provisional P4.3B directory evidence package."
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    build = subcommands.add_parser("build", help="build a new evidence package")
    build.add_argument("--run", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--repository", type=Path, default=Path.cwd())
    verify = subcommands.add_parser("verify", help="verify an existing package")
    verify.add_argument("--package", type=Path, required=True)
    arguments = parser.parse_args()

    try:
        if arguments.command == "build":
            manifest = build_evidence_package(
                arguments.run,
                arguments.output,
                repository=arguments.repository,
            )
            print(
                f"built {arguments.output}: {manifest.entry_count} entries, "
                f"content_sha256={manifest.content_sha256}"
            )
        else:
            manifest = verify_evidence_package(arguments.package)
            print(
                f"verified {arguments.package}: {manifest.entry_count} entries, "
                f"content_sha256={manifest.content_sha256}"
            )
    except (ModelPipelineError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
