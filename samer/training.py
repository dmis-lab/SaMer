import torch
from torch import nn
import torch.nn.functional as F

from samer.colpali import extract_embeddings, extract_visual_tokens, prepare_image_inputs
from samer.losses import multi_positive_t2i_loss
from samer.merging import MergeConfig, merge_tokens_differentiable
from samer.scoring import mean_maxsim


MERGE_DIAGNOSTIC_KEYS = (
    "assignment_tau",
    "assignment_entropy",
    "object_assignment_kl",
    "object_consistent_assignment_mass",
    "cluster_label_purity",
    "instance_mixing_rate",
    "train_infer_assignment_agreement",
)


class MergedColPaliForTraining(nn.Module):
    """Wrap a ColPali retriever with SaMer merging and retrieval-only training loss.

    Args:
        colpali: Base ColPali-style retrieval model.
        merge_config: SaMer merge configuration.
        temperature: Temperature for the retrieval loss.
    """

    def __init__(
        self,
        colpali: nn.Module,
        merge_config: MergeConfig,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.colpali = colpali
        self.merge_config = merge_config
        self.temperature = temperature
        self._projector_grad_sq = 0.0
        for name, param in self.colpali.named_parameters():
            if param.requires_grad and "embedding_proj_layer" in name:
                param.register_hook(self._capture_projector_grad)

    def forward(
        self,
        image_inputs: dict[str, torch.Tensor] | None = None,
        text_inputs: dict[str, torch.Tensor] | None = None,
        image_ids: list[str] | None = None,
        phrase_annotations: list[list[dict[str, torch.Tensor]]] | None = None,
        **_: object,
    ) -> dict[str, torch.Tensor]:
        """Encode image/text batches, merge image tokens, and compute retrieval loss.

        Args:
            image_inputs: Processor output for image inputs.
            text_inputs: Processor output for text inputs.
            image_ids: Image ids used to build the multi-positive target mask.
            phrase_annotations: Optional phrase-box annotations used only to
                derive object labels for training-time merging.
            **_: Ignored extra fields passed by the Trainer.

        Returns:
            A dictionary containing retrieval loss, logits, and SaMer diagnostics.
        """

        if image_inputs is None or text_inputs is None or image_ids is None:
            raise ValueError("image_inputs, text_inputs, and image_ids are required.")

        image_inputs = prepare_image_inputs(self.colpali, image_inputs)
        image_outputs = self.colpali(**image_inputs)
        visual_sequences = extract_visual_tokens(image_outputs, image_inputs, self.colpali)
        query_outputs = self.colpali(**text_inputs)
        query_tokens = F.normalize(extract_embeddings(query_outputs).float(), p=2, dim=-1)

        merged_tokens = []
        empty_counts = []
        diagnostics = {key: [] for key in MERGE_DIAGNOSTIC_KEYS}
        for batch_idx, sequence in enumerate(visual_sequences):
            tokens = sequence.tokens
            coords = sequence.coords
            bbox_labels = None
            if self.training and phrase_annotations is not None:
                bbox_labels = _bbox_token_labels(
                    coords,
                    phrase_annotations[batch_idx] if batch_idx < len(phrase_annotations) else [],
                )
            merged = merge_tokens_differentiable(
                tokens=tokens,
                coords=coords,
                config=self.merge_config,
                bbox_labels=bbox_labels,
            )
            merged_tokens.append(merged["tokens"])
            empty_counts.append(merged["num_empty_clusters"].to(tokens.device).float())
            for key in MERGE_DIAGNOSTIC_KEYS:
                value = merged.get(key)
                if value is not None:
                    diagnostics[key].append(value.to(tokens.device).float())

        image_tokens_merged = torch.stack(merged_tokens, dim=0)
        image_mask = torch.ones(image_tokens_merged.shape[:2], dtype=torch.bool, device=image_tokens_merged.device)
        query_mask = _query_mask_from_inputs(text_inputs, query_tokens)
        scores = mean_maxsim(query_tokens, image_tokens_merged, query_mask=query_mask, image_mask=image_mask)
        retrieval_loss = multi_positive_t2i_loss(scores, _positive_mask(image_ids, scores.device), self.temperature)

        grad_norm_projector = self._projector_grad_sq ** 0.5
        self._projector_grad_sq = 0.0
        zero = torch.zeros((), device=scores.device)
        outputs = {
            "loss": retrieval_loss,
            "retrieval_loss": retrieval_loss.detach(),
            "grad_norm_projector": torch.as_tensor(grad_norm_projector, device=scores.device),
            "logits": scores.detach(),
            "avg_empty_clusters": torch.stack(empty_counts).mean() if empty_counts else zero,
        }
        for key, values in diagnostics.items():
            outputs[key] = torch.stack(values).mean().detach() if values else zero
        return outputs

    def _capture_projector_grad(self, grad: torch.Tensor) -> torch.Tensor:
        self._projector_grad_sq += float(grad.detach().float().norm().cpu().item() ** 2)
        return grad


def _query_mask_from_inputs(text_inputs: dict[str, torch.Tensor], query_tokens: torch.Tensor) -> torch.Tensor:
    attention_mask = text_inputs.get("attention_mask")
    if attention_mask is None or attention_mask.size(1) != query_tokens.size(1):
        return torch.ones(query_tokens.shape[:2], dtype=torch.bool, device=query_tokens.device)
    return attention_mask.to(device=query_tokens.device, dtype=torch.bool)


def _positive_mask(image_ids: list[str], device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [[left == right for right in image_ids] for left in image_ids],
        dtype=torch.bool,
        device=device,
    )


@torch.no_grad()
def _bbox_token_labels(coords: torch.Tensor, phrases: list[dict[str, torch.Tensor]]) -> torch.Tensor:
    """Map normalized token coordinates to phrase-instance labels from training boxes.

    Args:
        coords: Normalized token coordinates with shape ``[num_tokens, 2]``.
        phrases: Phrase annotations containing normalized boxes.

    Returns:
        Per-token integer labels, where ``0`` denotes background.
    """

    labels = torch.zeros(coords.size(0), dtype=torch.long, device=coords.device)
    best_area = torch.full((coords.size(0),), float("inf"), dtype=torch.float32, device=coords.device)
    next_label = 1
    x = coords[:, 0]
    y = coords[:, 1]
    for phrase in phrases:
        boxes = phrase.get("boxes")
        if boxes is None:
            continue
        boxes = boxes.to(device=coords.device, dtype=torch.float32)
        for box in boxes:
            x1, y1, x2, y2 = box
            area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
            inside = (x >= x1) & (x <= x2) & (y >= y1) & (y <= y2)
            update = inside & (area < best_area)
            labels[update] = next_label
            best_area[update] = area
            next_label += 1
    return labels
