from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class MergeConfig:
    """Configuration for SaMer feature-spatial token merging.

    Attributes:
        num_regions: Number of merged image-side tokens to produce.
        cluster_iters: Number of feature-spatial clustering iterations.
        spatial_weight: Weight for normalized 2D coordinate distance.
        object_penalty_eta: Strength of the object inconsistency penalty.
        assignment_temperature: Temperature for differentiable assignments.
        seed: Reserved random seed for deterministic extensions.
    """

    num_regions: int = 64
    cluster_iters: int = 3
    spatial_weight: float = 0.1
    object_penalty_eta: float = 1.0
    assignment_temperature: float = 0.07
    seed: int = 42


def merge_tokens(tokens: torch.Tensor, coords: torch.Tensor | None, config: MergeConfig) -> dict[str, torch.Tensor]:
    """Merge image-side tokens without bbox labels for cache construction and inference.

    Args:
        tokens: Image token embeddings with shape ``[num_tokens, dim]``.
        coords: Normalized token coordinates with shape ``[num_tokens, 2]``.
        config: SaMer merge configuration.

    Returns:
        A dictionary containing merged tokens, merged coordinates, hard
        assignments, empty-cluster count, and merge metadata.
    """
    tokens = F.normalize(tokens.float(), p=2, dim=-1)
    coords = _prepare_coords(tokens, coords)
    assignments, init_idx, empty, feature_centers, spatial_centers = _feature_spatial_clusters(tokens, coords, config)
    merged_tokens, merged_coords, weights = _differentiable_centroids(
        tokens=tokens,
        coords=coords,
        feature_centers=feature_centers,
        spatial_centers=spatial_centers,
        config=config,
        bbox_labels=None,
    )
    return _with_metadata(
        {
            "tokens": merged_tokens,
            "coords": merged_coords,
            "assignments": assignments.detach().cpu(),
            "num_empty_clusters": torch.as_tensor(empty),
            "assignment_entropy": _assignment_entropy(weights),
        },
        config,
    )


def merge_tokens_differentiable(
    tokens: torch.Tensor,
    coords: torch.Tensor | None,
    config: MergeConfig,
    bbox_labels: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Training-time SaMer merge.

    Cluster centers are selected with bbox-free feature-spatial clustering. If
    object labels are available, an additive instance inconsistency term biases
    the differentiable assignment before constructing merged centroids.

    Args:
        tokens: Image token embeddings with shape ``[num_tokens, dim]``.
        coords: Normalized token coordinates with shape ``[num_tokens, 2]``.
        config: SaMer merge configuration.
        bbox_labels: Optional per-token object instance labels. Background is
            expected to be label ``0``.

    Returns:
        A dictionary containing differentiable merged tokens, coordinates,
        hard assignments, and training diagnostics.
    """
    tokens = F.normalize(tokens.float(), p=2, dim=-1)
    coords = _prepare_coords(tokens, coords)
    with torch.no_grad():
        assignments, init_idx, empty, feature_centers, spatial_centers = _feature_spatial_clusters(tokens.detach(), coords, config)
    labels = bbox_labels.to(device=tokens.device, dtype=torch.long) if bbox_labels is not None else None
    merged_tokens, merged_coords, weights = _differentiable_centroids(
        tokens=tokens,
        coords=coords,
        feature_centers=feature_centers,
        spatial_centers=spatial_centers,
        config=config,
        bbox_labels=labels,
        hard_assignments=assignments,
    )
    out = {
        "tokens": merged_tokens,
        "coords": merged_coords,
        "assignments": assignments.detach().cpu(),
        "num_empty_clusters": torch.as_tensor(empty),
        "assignment_entropy": _assignment_entropy(weights),
        "object_penalty_eta": torch.as_tensor(config.object_penalty_eta, device=tokens.device, dtype=tokens.dtype),
        "assignment_tau": torch.as_tensor(config.assignment_temperature, device=tokens.device, dtype=tokens.dtype),
        "train_infer_assignment_agreement": (weights.argmax(dim=-1) == assignments).float().mean().detach(),
    }
    if labels is not None:
        stats = _object_label_stats(assignments, labels, init_idx.numel(), dtype=tokens.dtype)
        out.update(
            {
                "object_consistent_assignment_mass": _object_consistent_mass(
                    weights,
                    labels,
                    stats["majority_labels"].to(tokens.device),
                    stats["non_empty"].to(tokens.device),
                ).detach(),
                "cluster_label_purity": stats["cluster_label_purity"].to(tokens.device).detach(),
                "instance_mixing_rate": stats["instance_mixing_rate"].to(tokens.device).detach(),
                "object_assignment_kl": _assignment_kl_without_object_penalty(
                    tokens,
                    coords,
                    feature_centers,
                    spatial_centers,
                    weights,
                    config,
                ).detach(),
            }
        )
    return _with_metadata(out, config)


def _feature_spatial_clusters(
    tokens: torch.Tensor,
    coords: torch.Tensor,
    config: MergeConfig,
) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor]:
    """Run feature-spatial clustering and return hard assignments plus centroids.

    Args:
        tokens: Normalized token embeddings.
        coords: Normalized token coordinates.
        config: SaMer merge configuration.

    Returns:
        Hard assignments, initial center indices, empty-cluster count, feature
        centers, and spatial centers.
    """

    num_tokens = tokens.size(0)
    k = min(int(config.num_regions), num_tokens)
    init_idx = _uniform_indices(num_tokens, k, tokens.device)
    feature_centers = tokens[init_idx].clone()
    spatial_centers = coords[init_idx].clone()
    assignments = torch.zeros(num_tokens, dtype=torch.long, device=tokens.device)
    for _ in range(max(int(config.cluster_iters), 1)):
        dist = _feature_spatial_distance(tokens, coords, feature_centers, spatial_centers, config.spatial_weight)
        assignments = dist.argmin(dim=-1)
        feature_centers, spatial_centers = _update_centers(tokens, coords, assignments, feature_centers, spatial_centers)
    empty = _count_empty_clusters(assignments, k)
    return assignments, init_idx, empty, feature_centers, spatial_centers


def _differentiable_centroids(
    tokens: torch.Tensor,
    coords: torch.Tensor,
    feature_centers: torch.Tensor,
    spatial_centers: torch.Tensor,
    config: MergeConfig,
    bbox_labels: torch.Tensor | None,
    hard_assignments: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build differentiable merged centroids from feature-spatial assignment logits.

    Args:
        tokens: Normalized token embeddings.
        coords: Normalized token coordinates.
        feature_centers: Current feature centroids.
        spatial_centers: Current spatial centroids.
        config: SaMer merge configuration.
        bbox_labels: Optional per-token object labels for the training prior.
        hard_assignments: Hard assignments used to estimate cluster label
            consistency.

    Returns:
        Merged tokens, merged coordinates, and token-to-cluster assignment
        weights.
    """

    dist = _feature_spatial_distance(tokens, coords, feature_centers, spatial_centers, config.spatial_weight)
    if bbox_labels is not None and hard_assignments is not None:
        penalty = _instance_inconsistency_penalty(hard_assignments, bbox_labels, feature_centers.size(0))
        dist = dist + float(config.object_penalty_eta) * penalty.to(device=tokens.device, dtype=tokens.dtype)
    tau = max(float(config.assignment_temperature), 1e-6)
    weights = torch.softmax(-dist / tau, dim=-1)
    denom = weights.sum(dim=0).clamp_min(1e-6)
    merged_tokens = F.normalize(weights.T @ tokens / denom[:, None], p=2, dim=-1)
    merged_coords = weights.T @ coords / denom[:, None]
    return merged_tokens, merged_coords, weights


def _feature_spatial_distance(
    tokens: torch.Tensor,
    coords: torch.Tensor,
    feature_centers: torch.Tensor,
    spatial_centers: torch.Tensor,
    spatial_weight: float,
) -> torch.Tensor:
    """Compute the SaMer distance combining cosine feature distance and spatial distance.

    Args:
        tokens: Normalized token embeddings.
        coords: Normalized token coordinates.
        feature_centers: Feature centroids.
        spatial_centers: Spatial centroids.
        spatial_weight: Weight applied to the coordinate distance.

    Returns:
        Pairwise token-to-cluster distances with shape ``[num_tokens, k]``.
    """

    dist = 1.0 - tokens @ feature_centers.T
    if spatial_weight > 0:
        dist = dist + float(spatial_weight) * (coords[:, None, :] - spatial_centers[None, :, :]).square().sum(dim=-1)
    return dist


def _instance_inconsistency_penalty(assignments: torch.Tensor, labels: torch.Tensor, k: int) -> torch.Tensor:
    """Return the additive penalty for assigning a token to object-inconsistent clusters.

    Args:
        assignments: Hard token-to-cluster assignments.
        labels: Per-token object instance labels.
        k: Number of clusters.

    Returns:
        Penalty matrix with shape ``[num_tokens, k]``.
    """

    penalty = torch.ones(labels.size(0), k, dtype=torch.float32, device=labels.device)
    for cluster_idx in range(k):
        cluster_mask = assignments == cluster_idx
        if not bool(cluster_mask.any()):
            continue
        cluster_labels = labels[cluster_mask]
        for label in torch.unique(cluster_labels):
            same_label_fraction = (cluster_labels == label).float().mean()
            penalty[labels == label, cluster_idx] = 1.0 - same_label_fraction
    return penalty


def _object_label_stats(
    assignments: torch.Tensor,
    labels: torch.Tensor,
    k: int,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Summarize cluster object purity and instance mixing from hard assignments.

    Args:
        assignments: Hard token-to-cluster assignments.
        labels: Per-token object instance labels.
        k: Number of clusters.
        dtype: Floating dtype used for diagnostic tensors.

    Returns:
        A dictionary with majority labels, non-empty mask, purity, and mixing
        diagnostics.
    """

    labels = labels.to(device=assignments.device, dtype=torch.long)
    max_label = int(labels.max().item()) if labels.numel() else 0
    counts = torch.zeros(k, max_label + 1, dtype=dtype, device=labels.device)
    cluster_sizes = torch.zeros(k, dtype=dtype, device=labels.device)
    for cluster_idx in range(k):
        cluster_mask = assignments == cluster_idx
        if not bool(cluster_mask.any()):
            continue
        cluster_labels = labels[cluster_mask]
        cluster_sizes[cluster_idx] = cluster_labels.numel()
        counts[cluster_idx].scatter_add_(0, cluster_labels, torch.ones_like(cluster_labels, dtype=dtype))

    non_empty = cluster_sizes > 0
    majority_counts, majority_labels = counts.max(dim=-1)
    object_counts = counts[:, 1:] if max_label >= 1 else torch.zeros(k, 0, dtype=dtype, device=labels.device)
    object_sizes = object_counts.sum(dim=-1)
    if max_label >= 1:
        object_majority_counts, object_majority_offsets = object_counts.max(dim=-1)
        has_objects = object_sizes > 0
        majority_labels = torch.where(has_objects, object_majority_offsets + 1, majority_labels)
        purity = torch.where(
            has_objects,
            object_majority_counts / object_sizes.clamp_min(1.0),
            majority_counts / cluster_sizes.clamp_min(1.0),
        )
        mixed = object_counts.gt(0).sum(dim=-1) > 1
        instance_mixing_rate = mixed[non_empty].to(dtype).mean() if bool(non_empty.any()) else torch.zeros((), dtype=dtype, device=labels.device)
    else:
        purity = majority_counts / cluster_sizes.clamp_min(1.0)
        instance_mixing_rate = torch.zeros((), dtype=dtype, device=labels.device)

    cluster_label_purity = purity[non_empty].mean() if bool(non_empty.any()) else torch.zeros((), dtype=dtype, device=labels.device)
    return {
        "majority_labels": majority_labels,
        "non_empty": non_empty,
        "cluster_label_purity": cluster_label_purity,
        "instance_mixing_rate": instance_mixing_rate,
    }


def _object_consistent_mass(
    weights: torch.Tensor,
    labels: torch.Tensor,
    majority_labels: torch.Tensor,
    non_empty: torch.Tensor,
) -> torch.Tensor:
    """Measure how much assignment mass object tokens place on matching-object clusters.

    Args:
        weights: Soft token-to-cluster assignment weights.
        labels: Per-token object instance labels.
        majority_labels: Majority object label for each cluster.
        non_empty: Boolean mask for non-empty clusters.

    Returns:
        Mean object-consistent assignment mass over foreground tokens.
    """

    labels = labels.to(device=weights.device, dtype=torch.long)
    object_mask = labels > 0
    if not bool(object_mask.any()):
        return torch.zeros((), dtype=weights.dtype, device=weights.device)
    consistent = (majority_labels[None, :] == labels[:, None]) & non_empty[None, :]
    mass = (weights * consistent.to(dtype=weights.dtype)).sum(dim=-1)
    return mass[object_mask].mean()


def _assignment_kl_without_object_penalty(
    tokens: torch.Tensor,
    coords: torch.Tensor,
    feature_centers: torch.Tensor,
    spatial_centers: torch.Tensor,
    object_weights: torch.Tensor,
    config: MergeConfig,
) -> torch.Tensor:
    """Measure how much the object penalty changes assignments from the bbox-free base.

    Args:
        tokens: Normalized token embeddings.
        coords: Normalized token coordinates.
        feature_centers: Feature centroids.
        spatial_centers: Spatial centroids.
        object_weights: Assignment weights after applying the object penalty.
        config: SaMer merge configuration.

    Returns:
        Average KL divergence from object-aware assignments to bbox-free base
        assignments.
    """

    dist = _feature_spatial_distance(tokens, coords, feature_centers, spatial_centers, config.spatial_weight)
    base_weights = torch.softmax(-dist / max(float(config.assignment_temperature), 1e-6), dim=-1)
    return (object_weights * (object_weights.clamp_min(1e-12).log() - base_weights.clamp_min(1e-12).log())).sum(dim=-1).mean()


def _assignment_entropy(weights: torch.Tensor) -> torch.Tensor:
    entropy = -(weights * weights.clamp_min(1e-12).log()).sum(dim=-1)
    normalizer = torch.log(torch.as_tensor(max(weights.size(-1), 2), device=weights.device, dtype=weights.dtype))
    return (entropy / normalizer).mean().detach()


def _update_centers(
    tokens: torch.Tensor,
    coords: torch.Tensor,
    assignments: torch.Tensor,
    old_features: torch.Tensor,
    old_spatial: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    feature_centers = old_features.clone()
    spatial_centers = old_spatial.clone()
    for idx in range(old_features.size(0)):
        mask = assignments == idx
        if not bool(mask.any()):
            continue
        feature_centers[idx] = F.normalize(tokens[mask].mean(dim=0), p=2, dim=-1)
        spatial_centers[idx] = coords[mask].mean(dim=0)
    return feature_centers, spatial_centers


def _prepare_coords(tokens: torch.Tensor, coords: torch.Tensor | None) -> torch.Tensor:
    """Validate provided coordinates or infer a normalized square-grid layout.

    Args:
        tokens: Token embeddings used to infer token count and device.
        coords: Optional normalized token coordinates.

    Returns:
        A normalized coordinate tensor with shape ``[num_tokens, 2]``.
    """

    if coords is None:
        side = int(tokens.size(0) ** 0.5)
        if side * side != tokens.size(0):
            raise ValueError("coords are required when token count is not a square grid.")
        ys, xs = torch.meshgrid(
            torch.arange(side, device=tokens.device, dtype=tokens.dtype),
            torch.arange(side, device=tokens.device, dtype=tokens.dtype),
            indexing="ij",
        )
        coords = torch.stack([(xs.reshape(-1) + 0.5) / side, (ys.reshape(-1) + 0.5) / side], dim=-1)
    coords = coords.to(device=tokens.device, dtype=tokens.dtype)
    if tuple(coords.shape) != (tokens.size(0), 2):
        raise ValueError(f"coords must have shape {(tokens.size(0), 2)}, got {tuple(coords.shape)}.")
    return coords


def _uniform_indices(num_tokens: int, k: int, device: torch.device) -> torch.Tensor:
    if k == 1:
        return torch.zeros(1, dtype=torch.long, device=device)
    return torch.linspace(0, num_tokens - 1, k, device=device).round().long().unique(sorted=True)


def _count_empty_clusters(assignments: torch.Tensor, k: int) -> int:
    return sum(int((assignments == idx).sum().item() == 0) for idx in range(k))


def _with_metadata(out: dict[str, torch.Tensor], config: MergeConfig) -> dict[str, torch.Tensor]:
    out["actual_num_tokens"] = torch.as_tensor(out["tokens"].shape[0])
    out["requested_num_regions"] = torch.as_tensor(config.num_regions)
    return out
