import torch


def text_to_image_metrics(scores: torch.Tensor, positive_mask: torch.Tensor) -> dict[str, float]:
    """Compute common text-to-image retrieval metrics from a score matrix.

    Args:
        scores: Query-image score matrix.
        positive_mask: Boolean matrix marking positive images per query.

    Returns:
        Dictionary of percentage-scaled retrieval metrics.
    """

    pos = positive_mask.bool()
    order = scores.argsort(dim=1, descending=True)
    ranked_pos = pos.gather(1, order)
    ranks = torch.arange(1, scores.size(1) + 1, dtype=torch.float32)[None, :]

    first_rank = torch.where(ranked_pos, ranks, torch.full_like(ranks, float("inf"))).min(dim=1).values
    mrr = torch.where(torch.isfinite(first_rank), 1.0 / first_rank, torch.zeros_like(first_rank)).mean()

    discounts = 1.0 / torch.log2(ranks + 1.0)
    dcg = (ranked_pos.float() * discounts).sum(dim=1)
    ideal = torch.zeros_like(dcg)
    for idx, count in enumerate(pos.sum(dim=1).clamp_min(1).tolist()):
        ideal[idx] = discounts[0, : int(count)].sum()
    ndcg = torch.where(ideal > 0, dcg / ideal, torch.zeros_like(dcg)).mean()

    metrics = {
        "mrr": round(float(mrr * 100.0), 3),
        "ndcg": round(float(ndcg * 100.0), 3),
        "r@1": round(float((first_rank <= 1).float().mean() * 100.0), 3),
        "r@5": round(float((first_rank <= 5).float().mean() * 100.0), 3),
        "r@10": round(float((first_rank <= 10).float().mean() * 100.0), 3),
    }
    metrics["hit_rate@3"] = round(float((first_rank <= 3).float().mean() * 100.0), 3)
    metrics["hit_rate_at_3"] = metrics["hit_rate@3"]
    for k in (1, 5, 10):
        metrics[f"mrr@{k}"] = _round_pct(_mrr_at_k(first_rank, k))
        metrics[f"ndcg@{k}"] = _round_pct(_ndcg_at_k(ranked_pos, pos, discounts, k))
        metrics[f"mrr_at_{k}"] = metrics[f"mrr@{k}"]
        metrics[f"ndcg_at_{k}"] = metrics[f"ndcg@{k}"]
    return metrics


def _mrr_at_k(first_rank: torch.Tensor, k: int) -> torch.Tensor:
    reciprocal = torch.where(first_rank <= k, 1.0 / first_rank, torch.zeros_like(first_rank))
    return reciprocal.mean()


def _ndcg_at_k(ranked_pos: torch.Tensor, pos: torch.Tensor, discounts: torch.Tensor, k: int) -> torch.Tensor:
    dcg = (ranked_pos[:, :k].float() * discounts[:, :k]).sum(dim=1)
    ideal = torch.zeros_like(dcg)
    positive_counts = pos.sum(dim=1).clamp_min(1).tolist()
    for idx, count in enumerate(positive_counts):
        ideal[idx] = discounts[0, : min(int(count), k)].sum()
    return torch.where(ideal > 0, dcg / ideal, torch.zeros_like(dcg)).mean()


def _round_pct(value: torch.Tensor) -> float:
    return round(float(value * 100.0), 3)
