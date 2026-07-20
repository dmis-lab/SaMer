import torch


def multi_positive_t2i_loss(scores: torch.Tensor, positive_mask: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Compute the multi-positive InfoNCE retrieval loss.

    Args:
        scores: Query-image score matrix with shape ``[num_queries, num_images]``.
        positive_mask: Boolean matrix marking positive images per query.
        temperature: Logit temperature for the retrieval objective.

    Returns:
        Scalar retrieval loss.
    """

    logits = scores.float() / max(float(temperature), 1e-6)
    positives = positive_mask.to(device=scores.device, dtype=torch.bool)
    if positives.shape != logits.shape:
        raise ValueError(
            f"positive_mask must match scores, got {tuple(positives.shape)} and {tuple(logits.shape)}."
        )
    if not bool(positives.any(dim=1).all()):
        raise ValueError("Each query must have at least one positive image.")

    positive_logits = logits.masked_fill(~positives, torch.finfo(logits.dtype).min)
    return -(torch.logsumexp(positive_logits, dim=1) - torch.logsumexp(logits, dim=1)).mean()
