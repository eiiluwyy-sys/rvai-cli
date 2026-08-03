"""Resolve Manifest label-set identifiers without runtime network access."""

from functools import lru_cache
from importlib.resources import files


@lru_cache(maxsize=None)
def load_label_catalog(name: str) -> tuple[str, ...] | None:
    """Load a known packaged catalog, or return ``None`` for custom identifiers."""

    if name != "imagenet-1k":
        return None
    content = files("rvai.data").joinpath("imagenet-1k.txt").read_text(encoding="utf-8")
    labels = []
    for line in content.splitlines():
        _, separator, description = line.partition(" ")
        labels.append(description if separator else line)
    return tuple(labels)


def classification_label(catalog: str, index: int) -> str:
    """Return a human label when packaged, with a deterministic fallback."""

    labels = load_label_catalog(catalog)
    if labels is not None and 0 <= index < len(labels):
        return labels[index]
    return f"{catalog}:{index}"
