import torch


def mean_maxsim(
    query_tokens: torch.Tensor,
    image_tokens: torch.Tensor,
    query_mask: torch.Tensor | None = None,
    image_mask: torch.Tensor | None = None,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Score query-image pairs with ColPali-style mean MaxSim over query tokens.

    Args:
        query_tokens: Query token embeddings with shape ``[num_queries, q_len, dim]``.
        image_tokens: Image token embeddings with shape ``[num_images, i_len, dim]``.
        query_mask: Optional valid-token mask for query tokens.
        image_mask: Optional valid-token mask for image tokens.
        chunk_size: Optional number of images to score per chunk.

    Returns:
        Score matrix with shape ``[num_queries, num_images]``.
    """

    if chunk_size is None:
        sim = torch.einsum("qmd,ind->qimn", query_tokens.float(), image_tokens.float())
        if image_mask is not None:
            sim = sim.masked_fill(~image_mask[None, :, None, :].bool(), torch.finfo(sim.dtype).min)
        maxsim = sim.max(dim=-1).values
        return _masked_mean(maxsim, query_mask)

    rows = []
    for start in range(0, image_tokens.size(0), chunk_size):
        sim = torch.einsum("qmd,ind->qimn", query_tokens.float(), image_tokens[start : start + chunk_size].float())
        if image_mask is not None:
            sim = sim.masked_fill(
                ~image_mask[start : start + chunk_size][None, :, None, :].bool(),
                torch.finfo(sim.dtype).min,
            )
        rows.append(_masked_mean(sim.max(dim=-1).values, query_mask))
    return torch.cat(rows, dim=1)


def _masked_mean(maxsim: torch.Tensor, query_mask: torch.Tensor | None) -> torch.Tensor:
    if query_mask is None:
        return maxsim.mean(dim=-1)
    mask = query_mask.to(dtype=maxsim.dtype)
    denom = mask.sum(dim=-1).clamp_min(1.0)
    return (maxsim * mask[:, None, :]).sum(dim=-1) / denom[:, None]
