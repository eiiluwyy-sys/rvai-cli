from pathlib import Path

import pytest

from rvai.model_pipeline.dataset import validate_dataset
from rvai.model_pipeline.io import canonical_json_bytes
from rvai.model_pipeline.synthetic import (
    SyntheticDataError,
    _write_ppm,
    generate_synthetic_dataset,
)


def test_synthetic_generation_is_byte_reproducible(tmp_path: Path) -> None:
    first = generate_synthetic_dataset(
        tmp_path / "first",
        calibration_sample_count=2,
        evaluation_sample_count=2,
        seed=4302,
    )
    second = generate_synthetic_dataset(
        tmp_path / "second",
        calibration_sample_count=2,
        evaluation_sample_count=2,
        seed=4302,
    )

    assert canonical_json_bytes(first.record) == canonical_json_bytes(second.record)
    assert first.calibration_manifest == second.calibration_manifest
    assert first.unlabeled_evaluation_manifest == second.unlabeled_evaluation_manifest
    first_samples = (
        first.calibration_manifest.samples
        + first.unlabeled_evaluation_manifest.samples
    )
    second_samples = (
        second.calibration_manifest.samples
        + second.unlabeled_evaluation_manifest.samples
    )
    for left, right in zip(
        first_samples,
        second_samples,
        strict=True,
    ):
        assert (first.root / left.path).read_bytes() == (
            second.root / right.path
        ).read_bytes()


def test_synthetic_splits_validate_and_do_not_share_content(tmp_path: Path) -> None:
    generated = generate_synthetic_dataset(
        tmp_path / "dataset",
        calibration_sample_count=2,
        evaluation_sample_count=2,
    )

    calibration = validate_dataset(generated.calibration_manifest, generated.root)
    evaluation = validate_dataset(
        generated.unlabeled_evaluation_manifest,
        generated.root,
    )

    calibration_digests = {
        sample.record.observed_sha256 for sample in calibration.samples
    }
    evaluation_digests = {
        sample.record.observed_sha256 for sample in evaluation.samples
    }
    assert calibration_digests.isdisjoint(evaluation_digests)
    assert (generated.root / "calibration-manifest.json").is_file()
    assert (generated.root / "evaluation-unlabeled-manifest.json").is_file()
    assert (generated.root / "generation.json").is_file()


def test_synthetic_generation_refuses_to_overwrite(tmp_path: Path) -> None:
    destination = tmp_path / "dataset"
    generate_synthetic_dataset(
        destination,
        calibration_sample_count=1,
        evaluation_sample_count=1,
    )

    with pytest.raises(SyntheticDataError, match="Refusing to overwrite"):
        generate_synthetic_dataset(
            destination,
            calibration_sample_count=1,
            evaluation_sample_count=1,
        )


def test_synthetic_pattern_does_not_repeat_at_256_images(tmp_path: Path) -> None:
    first = tmp_path / "first.ppm"
    later = tmp_path / "later.ppm"

    _write_ppm(first, index=0, seed=4302)
    _write_ppm(later, index=256, seed=4302)

    assert first.read_bytes() != later.read_bytes()
