from pathlib import Path
from typing import Any

import torch


def load_cache(path: str | Path) -> dict[str, Any]:
    """Load a saved retrieval cache onto CPU.

    Args:
        path: Cache file path.

    Returns:
        Deserialized cache payload.
    """

    path = Path(path)
    try:
        return torch.load(path, map_location="cpu", mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def save_cache(payload: dict[str, Any], path: str | Path) -> None:
    """Atomically save a retrieval cache payload.

    Args:
        payload: Cache payload to serialize.
        path: Destination file path.

    Returns:
        None.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)
