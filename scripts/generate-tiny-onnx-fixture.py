#!/usr/bin/env python3
"""Generate the tiny deterministic ONNX classifier committed for offline tests.

This maintenance script requires the build-only ``onnx`` package. The generated
model itself is executed by the supported ``onnxruntime`` optional dependency.
"""

from pathlib import Path

import onnx
from onnx import TensorProto, helper


def main() -> None:
    output_path = (
        Path(__file__).parents[1]
        / "tests"
        / "fixtures"
        / "onnx"
        / "tiny-classifier.onnx"
    )
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
        opset_imports=[helper.make_opsetid("", 13)],
    )
    onnx.checker.check_model(model)
    onnx.save_model(model, output_path)


if __name__ == "__main__":
    main()
