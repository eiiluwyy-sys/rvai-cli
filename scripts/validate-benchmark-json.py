#!/usr/bin/env python3
"""Validate the stable fields of a QEMU benchmark result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    args = parser.parse_args()

    payload = json.loads(args.result.read_text(encoding="utf-8"))
    execution = payload["execution"]

    expected_matrix = {"m": args.m, "n": args.n, "k": args.k}
    assert payload["status"] == "success"
    assert payload["correctness_verified"] is True
    assert payload["matrix"] == expected_matrix
    assert payload["iterations"] == args.iterations
    assert execution["target_architecture"] == "riscv64"
    assert execution["execution_environment"] == "qemu-user"
    assert execution["performance_representative"] is False


if __name__ == "__main__":
    main()
