import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoProcessor, ColPaliForRetrieval, ColQwen2ForRetrieval

from samer.coords import make_visual_coords


@dataclass
class VisualTokenSequence:
    """Post-projector visual tokens and the spatial grid they originate from.

    Args:
        tokens: Normalized visual-token embeddings with shape ``[num_tokens, dim]``.
        coords: Normalized coordinates aligned with ``tokens``.
        grid_thw: Optional pre-merge Qwen visual grid for dynamic-resolution
            models.
    """

    tokens: torch.Tensor
    coords: torch.Tensor
    grid_thw: tuple[int, int, int] | None = None


def resolve_local_model_path(model_name: str) -> str:
    """Resolve a Hugging Face model id to the latest local snapshot when available.

    Args:
        model_name: Hugging Face model id or local path.

    Returns:
        A local snapshot path when found, otherwise the original model id.
    """

    path = Path(model_name)
    if path.exists():
        return str(path)
    repo_cache = "models--" + model_name.replace("/", "--")
    roots = []
    for env_name in ("TRANSFORMERS_CACHE", "HF_HOME"):
        value = os.environ.get(env_name)
        if value:
            roots.append(Path(value))
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    for root in roots:
        hub = root if root.name == "hub" else root / "hub"
        snapshots = hub / repo_cache / "snapshots"
        if snapshots.exists():
            candidates = sorted([x for x in snapshots.iterdir() if x.is_dir()])
            if candidates:
                return str(candidates[-1])
    return model_name


def load_colpali(
    model_name: str,
    device: torch.device,
    local_files_only: bool = True,
    freeze: bool = True,
    eval_mode: bool = True,
    adapter_path: str | None = None,
):
    """Load a ColPali/ColQwen2 retriever and optional projector adapter.

    Args:
        model_name: Hugging Face model id or local checkpoint path.
        device: Device on which to place the model.
        local_files_only: Whether to avoid network downloads.
        freeze: Whether to freeze all loaded model parameters.
        eval_mode: Whether to switch the model to evaluation mode.
        adapter_path: Optional projector/adapter checkpoint path.

    Returns:
        The loaded model and processor.
    """

    projector_checkpoint = None
    if adapter_path:
        projector_path = Path(adapter_path) / "projector.pt"
        if projector_path.exists():
            projector_checkpoint = torch.load(projector_path, map_location="cpu", weights_only=False)
            adapter_path = None
    if adapter_path and not (Path(adapter_path) / "adapter_config.json").exists():
        model_name = adapter_path
        adapter_path = None
    model_path = resolve_local_model_path(model_name) if local_files_only else model_name
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=local_files_only)
    kwargs = {
        "dtype": torch.bfloat16 if device.type == "cuda" else torch.float32,
        "local_files_only": local_files_only,
    }
    try:
        model = AutoModel.from_pretrained(model_path, **kwargs)
    except ValueError as exc:
        if "ColPaliConfig" in str(exc):
            model = ColPaliForRetrieval.from_pretrained(model_path, **kwargs)
        elif "ColQwen2Config" in str(exc):
            model = ColQwen2ForRetrieval.from_pretrained(model_path, **kwargs)
        else:
            raise
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=not freeze)
    if projector_checkpoint is not None:
        load_projector_checkpoint(model, projector_checkpoint)
    model.to(device)
    if eval_mode:
        model.eval()
    if freeze:
        for param in model.parameters():
            param.requires_grad_(False)
    return model, processor


def save_projector_checkpoint(model: nn.Module, output_dir: str | Path) -> Path:
    """Save only SaMer's trained shared projection layer.

    Args:
        model: Retrieval model containing ``embedding_proj_layer``.
        output_dir: Directory where ``projector.pt`` is written.

    Returns:
        Path to the saved projector checkpoint.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    projector = find_projector_layer(model)
    config = next(iter(_model_configs(model)), None)
    payload = {
        "format": "samer_projector_v1",
        "base_model_name_or_path": getattr(config, "_name_or_path", None),
        "projector_state_dict": {
            key: value.detach().cpu().clone() for key, value in projector.state_dict().items()
        },
    }
    path = output_path / "projector.pt"
    torch.save(payload, path)
    return path


def load_projector_checkpoint(model: nn.Module, checkpoint: dict[str, Any]) -> None:
    """Load a projector-only SaMer checkpoint into a compatible base model.

    Args:
        model: Base retrieval model that provides ``embedding_proj_layer``.
        checkpoint: Payload read from a SaMer ``projector.pt`` file.

    Returns:
        None.
    """

    state_dict = checkpoint.get("projector_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError("Invalid SaMer projector checkpoint payload.")
    find_projector_layer(model).load_state_dict(state_dict, strict=True)


def find_projector_layer(model: nn.Module) -> nn.Linear:
    """Find the shared image-text retrieval projector in a base or PEFT model.

    Args:
        model: Retrieval model or lightweight wrapper.

    Returns:
        The shared projection layer.
    """

    for current in _model_chain(model):
        projector = getattr(current, "embedding_proj_layer", None)
        if isinstance(projector, nn.Linear):
            return projector
    for name, module in model.named_modules():
        if name.endswith("embedding_proj_layer") and isinstance(module, nn.Linear):
            return module
    raise RuntimeError("Could not find embedding_proj_layer in the retrieval model.")


def extract_embeddings(outputs) -> torch.Tensor:
    """Extract token embeddings from the model output formats used by ColPali models.

    Args:
        outputs: Model output object, dictionary, or tuple.

    Returns:
        Token embeddings tensor.
    """

    for name in ("embeddings", "last_hidden_state"):
        value = getattr(outputs, name, None)
        if value is not None:
            return value
    if isinstance(outputs, dict):
        for name in ("embeddings", "last_hidden_state"):
            if name in outputs:
                return outputs[name]
    if isinstance(outputs, (tuple, list)) and outputs:
        return outputs[0]
    raise RuntimeError("Could not find ColPali token embeddings in model output.")


def to_device(batch: dict, device: torch.device) -> dict:
    """Move tensor values in a processor batch to the target device.

    Args:
        batch: Processor output dictionary.
        device: Target torch device.

    Returns:
        Batch dictionary with tensor values moved to ``device``.
    """

    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def prepare_image_inputs(model, image_inputs: dict[str, Any]) -> dict[str, Any]:
    """Preserve ColQwen2 dynamic-resolution metadata for a model image forward.

    The ColQwen2 processor emits one padded sequence per batch together with
    ``image_grid_thw``. The grid is required both by the vision encoder and by
    SaMer when it reconstructs spatial coordinates after selecting image-token
    positions from the multimodal sequence.

    Args:
        model: ColPali or ColQwen2 retrieval model.
        image_inputs: Image processor output.

    Returns:
        A dictionary suitable for the model forward call.
    """

    prepared = dict(image_inputs)
    if "image_grid_thw" not in prepared:
        return prepared

    image_token_id = _image_token_id(model)
    input_ids = prepared.get("input_ids")
    if image_token_id is None or input_ids is None:
        raise RuntimeError("ColQwen2 image inputs require input_ids and an image_token_id.")
    if "mm_token_type_ids" not in prepared:
        prepared["mm_token_type_ids"] = (input_ids == image_token_id).long()
    return prepared


def extract_visual_tokens(
    outputs,
    image_inputs: dict[str, Any],
    model,
) -> list[VisualTokenSequence]:
    """Select visual tokens and build their true image-grid coordinates.

    ColQwen2 returns embeddings for the whole multimodal prompt. For dynamic
    images, this function keeps only positions occupied by image placeholders
    and derives coordinates from ``image_grid_thw`` after the vision encoder's
    spatial merge. Fixed-resolution ColPali keeps its established full visual
    token sequence and coordinate convention.

    Args:
        outputs: Retrieval-model outputs containing token embeddings.
        image_inputs: The exact processed image batch used for ``outputs``.
        model: ColPali or ColQwen2 retrieval model.

    Returns:
        One visual-token sequence per input image.
    """

    embeddings = F.normalize(extract_embeddings(outputs).float(), p=2, dim=-1)
    if embeddings.ndim != 3:
        raise ValueError(f"Expected [batch, tokens, dim] embeddings, got {tuple(embeddings.shape)}.")

    image_grid_thw = image_inputs.get("image_grid_thw")
    if image_grid_thw is None:
        return [
            VisualTokenSequence(
                tokens=embeddings[index],
                coords=make_visual_coords(
                    embeddings.size(1),
                    device=embeddings.device,
                    dtype=embeddings.dtype,
                ),
            )
            for index in range(embeddings.size(0))
        ]

    image_token_id = _image_token_id(model)
    input_ids = image_inputs.get("input_ids")
    if image_token_id is None or input_ids is None:
        raise RuntimeError("Cannot select ColQwen2 visual tokens without input_ids and image_token_id.")
    if input_ids.shape != embeddings.shape[:2]:
        raise ValueError(
            "ColQwen2 input_ids and embeddings must have matching batch and sequence dimensions, got "
            f"{tuple(input_ids.shape)} and {tuple(embeddings.shape)}."
        )
    grids = torch.as_tensor(image_grid_thw, dtype=torch.long)
    if grids.ndim != 2 or grids.shape != (embeddings.size(0), 3):
        raise ValueError(
            "image_grid_thw must have shape [batch, 3], got "
            f"{tuple(grids.shape)} for batch size {embeddings.size(0)}."
        )

    merge_size = _spatial_merge_size(model)
    image_mask = input_ids.to(device=embeddings.device) == image_token_id
    sequences = []
    for index in range(embeddings.size(0)):
        grid = tuple(int(value) for value in grids[index].tolist())
        tokens = embeddings[index][image_mask[index]]
        coords = make_visual_coords(
            tokens.size(0),
            image_grid_thw=grid,
            merge_size=merge_size,
            device=embeddings.device,
            dtype=embeddings.dtype,
        )
        sequences.append(VisualTokenSequence(tokens=tokens, coords=coords, grid_thw=grid))
    return sequences


@torch.inference_mode()
def encode_images(model, processor, images: list, device: torch.device) -> list[VisualTokenSequence]:
    """Encode images into visual tokens with fixed or dynamic spatial grids.

    Args:
        model: ColPali-style retrieval model.
        processor: Matching Hugging Face processor.
        images: List of PIL images.
        device: Device used for encoding.

    Returns:
        Per-image normalized token sequences and matching normalized coordinates
        on CPU.
    """

    batch = prepare_image_inputs(model, processor(images=images, return_tensors="pt"))
    outputs = model(**to_device(batch, device))
    sequences = extract_visual_tokens(outputs, batch, model)
    return [
        VisualTokenSequence(
            tokens=sequence.tokens.cpu(),
            coords=sequence.coords.cpu(),
            grid_thw=sequence.grid_thw,
        )
        for sequence in sequences
    ]


@torch.inference_mode()
def encode_texts(model, processor, texts: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode text queries into normalized tokens and a valid-token mask.

    Args:
        model: ColPali-style retrieval model.
        processor: Matching Hugging Face processor.
        texts: List of query strings.
        device: Device used for encoding.

    Returns:
        Normalized text token embeddings and a boolean valid-token mask on CPU.
    """

    batch = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    embeddings = extract_embeddings(model(**to_device(batch, device)))
    embeddings = F.normalize(embeddings.float().cpu(), p=2, dim=-1)
    attention_mask = batch.get("attention_mask")
    if attention_mask is None or attention_mask.size(1) != embeddings.size(1):
        attention_mask = torch.ones(embeddings.shape[:2], dtype=torch.bool)
    else:
        attention_mask = attention_mask.cpu().bool()
    return embeddings, attention_mask


def _image_token_id(model) -> int | None:
    """Find the image placeholder token id across base and PEFT wrappers."""

    for config in _model_configs(model):
        for candidate in (config, getattr(config, "vlm_config", None)):
            value = getattr(candidate, "image_token_id", None) if candidate is not None else None
            if value is not None:
                return int(value)
    return None


def _spatial_merge_size(model) -> int:
    """Return the Qwen vision encoder's per-axis spatial merge factor."""

    for config in _model_configs(model):
        vlm_config = getattr(config, "vlm_config", None)
        vision_config = getattr(vlm_config, "vision_config", None)
        if vision_config is None:
            vision_config = getattr(config, "vision_config", None)
        value = getattr(vision_config, "spatial_merge_size", None) if vision_config is not None else None
        if value is not None:
            return int(value)
    raise RuntimeError("Could not determine the ColQwen2 vision spatial_merge_size.")


def _model_configs(model) -> list[Any]:
    """Collect configurations from a model and any lightweight wrapper."""

    configs = []
    for current in _model_chain(model):
        config = getattr(current, "config", None)
        if config is not None:
            configs.append(config)
    return configs


def _model_chain(model) -> list[Any]:
    """Traverse lightweight base-model wrappers without entering a cycle."""

    chain = []
    seen = set()
    current = model
    for _ in range(4):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        chain.append(current)
        next_model = getattr(current, "base_model", None)
        if next_model is current:
            break
        current = next_model
    return chain
