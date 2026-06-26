import torch
import torch.nn.functional as F


def multi_positive_t2i_loss(scores: torch.Tensor, positive_mask: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Compute cross-entropy retrieval loss with one or more positive images per query.

    Args:
        scores: Query-image score matrix with shape ``[num_queries, num_images]``.
        positive_mask: Boolean matrix marking positive images per query.
        temperature: Logit temperature for the retrieval objective.

    Returns:
        Scalar retrieval loss.
    """

    logits = scores / max(float(temperature), 1e-6)
    log_probs = F.log_softmax(logits, dim=1)
    positives = positive_mask.to(device=scores.device, dtype=torch.bool)
    positive_counts = positives.sum(dim=1).clamp_min(1)
    loss = -(log_probs * positives.to(log_probs.dtype)).sum(dim=1) / positive_counts
    return loss.mean()
