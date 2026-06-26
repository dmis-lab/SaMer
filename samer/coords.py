import torch


def make_visual_coords(num_tokens: int, image_token_count: int = 1024) -> torch.Tensor:
    """Create normalized 2D coordinates for visual tokens on the image grid.

    Args:
        num_tokens: Number of visual tokens in the sequence.
        image_token_count: Expected number of image-grid tokens before special tokens.

    Returns:
        Coordinate tensor with shape ``[num_tokens, 2]``.
    """

    grid_tokens = min(num_tokens, image_token_count)
    side = int(grid_tokens**0.5)
    if side * side != grid_tokens:
        side = int(image_token_count**0.5)
        grid_tokens = min(num_tokens, side * side)
    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, side),
        torch.linspace(0.0, 1.0, side),
        indexing="ij",
    )
    coords = torch.full((num_tokens, 2), 0.5)
    coords[:grid_tokens] = torch.stack([xx.flatten(), yy.flatten()], dim=-1)[:grid_tokens]
    return coords


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
