from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from samer.flickr import load_image, read_caption_entities, read_captions, read_split_ids


class FlickrCaptionDataset:
    """Flatten Flickr30k-Entities images and captions into retrieval training rows.

    Args:
        root: Flickr30k-Entities root directory.
        split: Dataset split name.
        limit_images: Optional maximum number of images to load.
        include_entities: Whether to include entity spans and boxes.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        limit_images: int | None = None,
        include_entities: bool = False,
    ) -> None:
        self.root = Path(root)
        image_ids = read_split_ids(self.root, split)
        if limit_images is not None:
            image_ids = image_ids[:limit_images]
        self.rows: list[dict[str, Any]] = []
        for image_id in image_ids:
            if include_entities:
                rows = read_caption_entities(self.root, image_id)
                for row in rows:
                    self.rows.append(
                        {
                            "image_id": str(image_id),
                            "caption_id": f"{image_id}_{row['caption_idx']}",
                            "text": row["text"],
                            "entities": row["entities"],
                        }
                    )
            else:
                for caption_idx, caption in enumerate(read_captions(self.root, image_id)):
                    self.rows.append(
                        {
                            "image_id": str(image_id),
                            "caption_id": f"{image_id}_{caption_idx}",
                            "text": caption,
                        }
                    )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


@dataclass
class ColPaliTrainCollator:
    """Create ColPali image/text batches and optional phrase-box annotations.

    Attributes:
        processor: Hugging Face processor for ColPali-style models.
        data_root: Flickr30k-Entities root directory.
        max_bbox_phrases_per_caption: Maximum phrase boxes kept per caption.
        min_bbox_area: Minimum normalized box area to keep.
        max_bbox_area: Maximum normalized box area to keep.
    """

    processor: Any
    data_root: str | Path
    max_bbox_phrases_per_caption: int = 4
    min_bbox_area: float = 0.005
    max_bbox_area: float = 1.0

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        """Load images, tokenize captions, and attach phrase annotations when present.

        Args:
            features: Dataset rows for one training/evaluation batch.

        Returns:
            A Trainer-ready batch containing image inputs, text inputs, image ids,
            and optional phrase annotations.
        """

        image_ids = [str(item["image_id"]) for item in features]
        images = [load_image(self.data_root, image_id) for image_id in image_ids]
        texts = [str(item["text"]) for item in features]
        image_inputs = self.processor(images=images, return_tensors="pt")
        text_inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        batch = {
            "image_inputs": image_inputs,
            "text_inputs": text_inputs,
            "image_ids": image_ids,
        }
        phrase_annotations = self._phrase_annotations(features, texts, text_inputs)
        if phrase_annotations is not None:
            batch["phrase_annotations"] = phrase_annotations
        return batch

    def _phrase_annotations(
        self,
        features: list[dict[str, Any]],
        texts: list[str],
        text_inputs: dict[str, Any],
    ) -> list[list[dict[str, torch.Tensor]]] | None:
        """Align entity phrases to tokenizer offsets and normalized bounding boxes.

        Args:
            features: Dataset rows that may contain entity annotations.
            texts: Plain caption strings for the batch.
            text_inputs: Tokenized text inputs produced by the processor.

        Returns:
            Phrase annotations per sample, or ``None`` when unavailable.
        """

        if not any(item.get("entities") for item in features):
            return None
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            return None
        encoded = tokenizer(texts, return_offsets_mapping=True, padding=True, truncation=True)
        offsets = encoded.get("offset_mapping")
        if offsets is None:
            return None
        max_len = text_inputs["input_ids"].shape[1]
        annotations = []
        for item, item_offsets in zip(features, offsets):
            phrases = []
            filtered_entities = []
            for entity in item.get("entities", []):
                boxes = _filter_boxes(entity.get("boxes", []), self.min_bbox_area, self.max_bbox_area)
                if boxes:
                    filtered_entities.append({**entity, "boxes": boxes})
            entities = sorted(filtered_entities, key=lambda entity: _max_box_area(entity.get("boxes", [])), reverse=True)
            for entity in entities:
                boxes = entity["boxes"]
                token_mask = torch.zeros(max_len, dtype=torch.bool)
                for token_idx, (start, end) in enumerate(item_offsets[:max_len]):
                    if start == end:
                        continue
                    if start < entity["end"] and end > entity["start"]:
                        token_mask[token_idx] = True
                if not bool(token_mask.any()):
                    continue
                phrases.append(
                    {
                        "token_mask": token_mask,
                        "boxes": torch.tensor(boxes, dtype=torch.float32),
                    }
                )
                if len(phrases) >= self.max_bbox_phrases_per_caption:
                    break
            annotations.append(phrases)
        return annotations


def _box_area(box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _max_box_area(boxes: list[tuple[float, float, float, float]]) -> float:
    return max((_box_area(box) for box in boxes), default=0.0)


def _filter_boxes(
    boxes: list[tuple[float, float, float, float]],
    min_area: float,
    max_area: float,
) -> list[tuple[float, float, float, float]]:
    return [box for box in boxes if min_area <= _box_area(box) <= max_area]
