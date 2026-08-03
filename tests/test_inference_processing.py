import importlib.util
from pathlib import Path

import pytest

from rvai.inference import (
    InferenceInputError,
    InferenceOutputError,
    classification_top_k,
    classification_label,
    load_onnx_dependencies,
    preprocess_image,
)
from rvai.manifest import ClassificationOutputSpec, ImageInputSpec


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "onnx"
pytestmark = pytest.mark.skipif(
    any(
        importlib.util.find_spec(module) is None
        for module in ("numpy", "onnxruntime", "PIL")
    ),
    reason="ONNX optional dependencies are not installed",
)


def input_spec() -> ImageInputSpec:
    return ImageInputSpec.model_validate(
        {
            "type": "image",
            "width": 2,
            "height": 2,
            "layout": "nchw",
            "dtype": "float32",
            "color_space": "rgb",
            "resize": "bilinear",
            "normalize": {
                "scale": 1.0,
                "mean": [0.0, 0.0, 0.0],
                "std": [1.0, 1.0, 1.0],
            },
        }
    )


def output_spec(scores: str = "logits") -> ClassificationOutputSpec:
    return ClassificationOutputSpec(
        type="classification",
        top_k=2,
        labels="rgb",
        scores=scores,
    )


def test_preprocess_builds_batch_one_nchw_float_tensor() -> None:
    dependencies = load_onnx_dependencies()

    tensor, info = preprocess_image(
        FIXTURE_DIR / "red-image.ppm",
        input_spec(),
        numpy=dependencies.numpy,
        pillow_image=dependencies.pillow_image,
    )

    assert tensor.shape == (1, 3, 2, 2)
    assert str(tensor.dtype) == "float32"
    assert float(tensor[0, 0].mean()) > float(tensor[0, 1].mean())
    assert info.original_width == info.original_height == 2


def test_preprocess_supports_manifest_driven_resize_and_center_crop(tmp_path) -> None:
    dependencies = load_onnx_dependencies()
    numpy = dependencies.numpy
    image = numpy.zeros((2, 4, 3), dtype=numpy.uint8)
    image[:, :, 2] = 255
    image[:, 1:3, 2] = 0
    image[:, 1:3, 0] = 255
    source = tmp_path / "wide.png"
    dependencies.pillow_image.fromarray(image).save(source)
    spec = input_spec().model_copy(
        update={"resize_shorter": 2, "crop": "center"}
    )

    tensor, _ = preprocess_image(
        source,
        spec,
        numpy=numpy,
        pillow_image=dependencies.pillow_image,
    )

    assert tensor.shape == (1, 3, 2, 2)
    assert float(tensor[0, 0].mean()) == 255.0
    assert float(tensor[0, 2].mean()) == 0.0


def test_preprocess_reports_missing_image_without_traceback_details(tmp_path) -> None:
    dependencies = load_onnx_dependencies()

    with pytest.raises(InferenceInputError, match="Cannot read input image"):
        preprocess_image(
            tmp_path / "missing.jpg",
            input_spec(),
            numpy=dependencies.numpy,
            pillow_image=dependencies.pillow_image,
        )


def test_top_k_softmax_is_stable_and_deterministic() -> None:
    numpy = load_onnx_dependencies().numpy

    predictions = classification_top_k(
        numpy.asarray([[1000.0, 1000.0, 999.0]]),
        output_spec(),
        numpy=numpy,
    )

    assert [prediction.index for prediction in predictions] == [0, 1]
    assert predictions[0].label == "rgb:0"
    assert 0.4 < predictions[0].score < 0.5


def test_top_k_rejects_non_classification_shape() -> None:
    numpy = load_onnx_dependencies().numpy

    with pytest.raises(InferenceOutputError, match="Expected classification"):
        classification_top_k(
            numpy.zeros((1, 2, 2)),
            output_spec(),
            numpy=numpy,
        )


def test_packaged_imagenet_catalog_resolves_human_labels() -> None:
    assert classification_label("imagenet-1k", 975) == "lakeside, lakeshore"
    assert classification_label("custom-labels", 7) == "custom-labels:7"
