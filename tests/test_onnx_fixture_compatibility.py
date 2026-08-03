import subprocess
import sys
from pathlib import Path

import numpy
import onnx
import onnxruntime


ROOT = Path(__file__).parents[1]
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "onnx" / "tiny-classifier.onnx"
GENERATOR_PATH = ROOT / "scripts" / "generate-tiny-onnx-fixture.py"
EXPECTED_IR_VERSION = 8
EXPECTED_OPSET_VERSION = 13


def assert_model_contract(path: Path) -> None:
    model = onnx.load(path)
    assert model.ir_version == EXPECTED_IR_VERSION
    assert [(item.domain, item.version) for item in model.opset_import] == [
        ("", EXPECTED_OPSET_VERSION)
    ]
    onnx.checker.check_model(model)

    session = onnxruntime.InferenceSession(
        str(path),
        providers=["CPUExecutionProvider"],
    )
    input_metadata = session.get_inputs()
    output_metadata = session.get_outputs()
    assert session.get_providers()[0] == "CPUExecutionProvider"
    assert len(input_metadata) == len(output_metadata) == 1
    assert input_metadata[0].name == "image"
    assert input_metadata[0].shape == [1, 3, 2, 2]
    assert input_metadata[0].type == "tensor(float)"
    assert output_metadata[0].name == "scores"
    assert output_metadata[0].shape == [1, 3]
    assert output_metadata[0].type == "tensor(float)"

    tensor = numpy.asarray(
        [[[[1.0, 3.0], [5.0, 7.0]], [[2.0, 4.0], [6.0, 8.0]], [[0.0] * 2] * 2]],
        dtype=numpy.float32,
    )
    result = session.run(["scores"], {"image": tensor})[0]
    numpy.testing.assert_allclose(
        result,
        numpy.asarray([[4.0, 5.0, 0.0]], dtype=numpy.float32),
    )


def test_committed_tiny_classifier_uses_compatible_ir_version() -> None:
    assert_model_contract(FIXTURE_PATH)


def test_generator_reproduces_compatible_executable_contract(tmp_path) -> None:
    generated = tmp_path / "tiny-classifier.onnx"

    subprocess.run(
        [sys.executable, str(GENERATOR_PATH), "--output", str(generated)],
        check=True,
    )

    assert_model_contract(generated)
