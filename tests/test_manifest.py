import pytest
from pydantic import ValidationError

from rvai.manifest import ModelManifest


def valid_manifest() -> dict[str, object]:
    return {
        "name": "demo-int8",
        "display_name": "Demo INT8",
        "task": "benchmark",
        "format": "builtin",
        "quantization": "int8",
        "runtime": "builtin",
        "resources": {
            "min_memory_mb": 128,
            "recommended_threads": "auto",
        },
        "riscv": {"require_rv64": False, "prefer_rvv": True},
    }


def test_manifest_accepts_valid_data() -> None:
    manifest = ModelManifest.model_validate(valid_manifest())

    assert manifest.name == "demo-int8"
    assert manifest.runtime == "builtin"
    assert manifest.resources.recommended_threads == "auto"


def test_manifest_rejects_unknown_fields() -> None:
    data = valid_manifest()
    data["unexpected"] = True

    with pytest.raises(ValidationError):
        ModelManifest.model_validate(data)
