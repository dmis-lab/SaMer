from collections.abc import Sequence

import torch


def make_visual_coords(
    num_tokens: int,
    image_token_count: int = 1024,
    image_grid_thw: torch.Tensor | Sequence[int] | None = None,
    merge_size: int = 1,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create normalized coordinates for fixed or dynamic visual-token grids.

    Args:
        num_tokens: Number of visual tokens in the sequence.
        image_token_count: Expected number of image-grid tokens before special tokens
            for fixed-resolution models such as ColPali.
        image_grid_thw: Optional dynamic Qwen grid ``(temporal, height, width)``
            before the vision encoder's spatial merge.
        merge_size: Per-axis visual merge factor used by the encoder. ColQwen2
            uses ``2``.
        device: Optional output device.
        dtype: Output coordinate dtype.

    Returns:
        Coordinate tensor with shape ``[num_tokens, 2]``.
    """

    if image_grid_thw is not None:
        return _coords_from_dynamic_grid(
            num_tokens=num_tokens,
            image_grid_thw=image_grid_thw,
            merge_size=merge_size,
            device=device,
            dtype=dtype,
        )

    grid_tokens = min(num_tokens, image_token_count)
    side = int(grid_tokens**0.5)
    if side * side != grid_tokens:
        side = int(image_token_count**0.5)
        grid_tokens = min(num_tokens, side * side)
    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, side, device=device, dtype=dtype),
        torch.linspace(0.0, 1.0, side, device=device, dtype=dtype),
        indexing="ij",
    )
    coords = torch.full((num_tokens, 2), 0.5, device=device, dtype=dtype)
    coords[:grid_tokens] = torch.stack([xx.flatten(), yy.flatten()], dim=-1)[:grid_tokens]
    return coords


def _coords_from_dynamic_grid(
    num_tokens: int,
    image_grid_thw: torch.Tensor | Sequence[int],
    merge_size: int,
    device: torch.device | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build coordinates for the actual ColQwen2 visual-token grid.

    Args:
        num_tokens: Number of post-projector image tokens selected from the
            model sequence.
        image_grid_thw: Qwen image grid before spatial merging.
        merge_size: Per-axis encoder merge factor.
        device: Output device.
        dtype: Output dtype.

    Returns:
        Normalized coordinates with shape ``[num_tokens, 2]``.
    """

    grid = torch.as_tensor(image_grid_thw, dtype=torch.long).flatten()
    if grid.numel() != 3:
        raise ValueError(f"image_grid_thw must contain three values, got {grid.tolist()}.")
    temporal, height, width = (int(value) for value in grid.tolist())
    merge_size = max(int(merge_size), 1)
    if height % merge_size or width % merge_size:
        raise ValueError(
            "The dynamic visual grid must be divisible by merge_size, got "
            f"grid={tuple(grid.tolist())}, merge_size={merge_size}."
        )
    grid_height = height // merge_size
    grid_width = width // merge_size
    expected_tokens = temporal * grid_height * grid_width
    if num_tokens != expected_tokens:
        raise ValueError(
            "Selected image-token count does not match image_grid_thw: "
            f"tokens={num_tokens}, expected={expected_tokens}, grid={tuple(grid.tolist())}, "
            f"merge_size={merge_size}."
        )

    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, grid_height, device=device, dtype=dtype),
        torch.linspace(0.0, 1.0, grid_width, device=device, dtype=dtype),
        indexing="ij",
    )
    coords = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
    return coords.repeat(temporal, 1)


def dtype_from_name(name: str) -> torch.dtype:
    """Map a cache dtype string to a torch dtype.

    Args:
        name: User-facing dtype name.

    Returns:
        Corresponding torch dtype.
    """

    dtypes = {
        "float32": torch.float32,
        "float": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    if name not in dtypes:
        raise ValueError(f"Unsupported dtype: {name}")
    return dtypes[name]
