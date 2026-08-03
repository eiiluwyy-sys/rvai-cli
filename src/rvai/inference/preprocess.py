"""Manifest-driven image decoding and tensor preparation."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any

from rvai.inference.errors import InferenceInputError
from rvai.inference.schema import InputInfo
from rvai.manifest import ImageInputSpec


def preprocess_image(
    path: Path,
    spec: ImageInputSpec,
    *,
    numpy: ModuleType,
    pillow_image: ModuleType,
) -> tuple[Any, InputInfo]:
    """Decode one RGB image and return a batch-one NCHW float32 tensor."""

    try:
        with pillow_image.open(path) as source:
            original_width, original_height = source.size
            image = source.convert("RGB")
            if spec.resize_shorter is not None:
                ratio = spec.resize_shorter / min(image.size)
                resized = (
                    int(round(image.size[0] * ratio)),
                    int(round(image.size[1] * ratio)),
                )
                image = image.resize(
                    resized,
                    resample=pillow_image.Resampling.BILINEAR,
                )
            else:
                image = image.resize(
                    (spec.width, spec.height),
                    resample=pillow_image.Resampling.BILINEAR,
                )
            if spec.crop == "center":
                left = (image.size[0] - spec.width) // 2
                top = (image.size[1] - spec.height) // 2
                image = image.crop(
                    (left, top, left + spec.width, top + spec.height)
                )
            array = numpy.asarray(image, dtype=numpy.float32)
    except (OSError, ValueError) as exc:
        raise InferenceInputError(f"Cannot read input image {path}: {exc}") from exc

    normalize = spec.normalize
    array = array * normalize.scale
    mean = numpy.asarray(normalize.mean, dtype=numpy.float32)
    std = numpy.asarray(normalize.std, dtype=numpy.float32)
    array = (array - mean) / std
    tensor = numpy.transpose(array, (2, 0, 1))[numpy.newaxis, ...]
    tensor = numpy.ascontiguousarray(tensor, dtype=numpy.float32)
    return tensor, InputInfo(
        path=str(path),
        original_width=original_width,
        original_height=original_height,
        tensor_shape=tuple(int(value) for value in tensor.shape),
    )
