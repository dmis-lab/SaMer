import os
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor, ColPaliForRetrieval, ColQwen2ForRetrieval


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
    model.to(device)
    if eval_mode:
        model.eval()
    if freeze:
        for param in model.parameters():
            param.requires_grad_(False)
    return model, processor


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


@torch.inference_mode()
def encode_images(model, processor, images: list, device: torch.device) -> torch.Tensor:
    """Encode images into normalized multi-vector retrieval tokens.

    Args:
        model: ColPali-style retrieval model.
        processor: Matching Hugging Face processor.
        images: List of PIL images.
        device: Device used for encoding.

    Returns:
        Normalized image token embeddings on CPU.
    """

    batch = processor(images=images, return_tensors="pt")
    embeddings = extract_embeddings(model(**to_device(batch, device)))
    return F.normalize(embeddings.float().cpu(), p=2, dim=-1)


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
