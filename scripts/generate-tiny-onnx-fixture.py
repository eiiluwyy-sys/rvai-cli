#!/usr/bin/env python3
"""Generate the tiny deterministic ONNX classifier committed for offline tests.

This maintenance script requires the build-only ``onnx`` package. The generated
model itself is executed by the supported ``onnxruntime`` optional dependency.
"""

import argparse
from pathlib import Path

import onnx
from onnx import TensorProto, helper


EXISTING_OPSET_VERSION = 13
COMPATIBLE_IR_VERSION = 8
DEFAULT_OUTPUT = (
    Path(__file__).parents[1]
    / "tests"
    / "fixtures"
    / "onnx"
    / "tiny-classifier.onnx"
)


def generate(output_path: Path) -> None:
    """Write one checked IR-8 model while retaining the existing opset."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    graph = helper.make_graph(
        [
            helper.make_node(
                "ReduceMean",
                inputs=["image"],
                outputs=["scores"],
                axes=[2, 3],
                keepdims=0,
            )
        ],
        "rvai-tiny-rgb-mean-classifier",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 2, 2])],
        [helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 3])],
    )
    model = helper.make_model(
        graph,
        producer_name="rvai-cli-test-fixture",
        opset_imports=[helper.make_operatorsetid("", EXISTING_OPSET_VERSION)],
    )
    model.ir_version = COMPATIBLE_IR_VERSION
    onnx.checker.check_model(model)
    onnx.save_model(model, output_path)

    saved = onnx.load(output_path)
    if saved.ir_version != COMPATIBLE_IR_VERSION:
        raise RuntimeError(
            f"saved fixture uses IR {saved.ir_version}, expected {COMPATIBLE_IR_VERSION}"
        )
    onnx.checker.check_model(saved)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    generate(arguments.output)


if __name__ == "__main__":
    main()
