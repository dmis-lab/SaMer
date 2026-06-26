import re
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image


ENTITY_RE = re.compile(r"\[/EN#(?P<entity_id>\d+)/(?P<entity_type>[^ ]+) (?P<phrase>[^\]]+)\]")


def strip_flickr_entities(line: str) -> str:
    """Remove Flickr30k-Entities markup while preserving phrase text.

    Args:
        line: Raw sentence line with entity markup.

    Returns:
        Plain caption text.
    """

    line = ENTITY_RE.sub(lambda match: match.group("phrase"), line)
    return line.replace("[", "").replace("]", "").strip()


def parse_flickr_sentence(line: str) -> tuple[str, list[dict]]:
    """Parse a Flickr30k-Entities caption into clean text and phrase spans.

    Args:
        line: Raw Flickr30k-Entities sentence line.

    Returns:
        Clean caption text and entity span metadata.
    """

    entities = []
    clean_parts = []
    cursor = 0
    clean_cursor = 0
    for match in ENTITY_RE.finditer(line):
        prefix = line[cursor : match.start()]
        clean_parts.append(prefix)
        clean_cursor += len(prefix)
        phrase = match.group("phrase")
        start = clean_cursor
        clean_parts.append(phrase)
        clean_cursor += len(phrase)
        entities.append(
            {
                "entity_id": match.group("entity_id"),
                "entity_type": match.group("entity_type"),
                "phrase": phrase,
                "start": start,
                "end": clean_cursor,
            }
        )
        cursor = match.end()
    suffix = line[cursor:]
    clean_parts.append(suffix)
    clean = "".join(clean_parts).replace("[", "").replace("]", "").strip()
    return clean, entities


def read_split_ids(root: str | Path, split: str) -> list[str]:
    """Read image ids for a Flickr30k-Entities split.

    Args:
        root: Flickr30k-Entities root directory.
        split: Split name.

    Returns:
        Image ids in split order.
    """

    path = Path(root) / "annotations" / f"{split}.txt"
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def read_captions(root: str | Path, image_id: str) -> list[str]:
    """Read plain captions for one Flickr image.

    Args:
        root: Flickr30k-Entities root directory.
        image_id: Flickr image id without extension.

    Returns:
        Plain captions with entity markup removed.
    """

    path = Path(root) / "annotations" / "Sentences" / f"{image_id}.txt"
    with path.open("r", encoding="utf-8") as f:
        return [strip_flickr_entities(line) for line in f if line.strip()]


def read_caption_entities(root: str | Path, image_id: str) -> list[dict]:
    """Read captions with entity spans and attach normalized entity boxes.

    Args:
        root: Flickr30k-Entities root directory.
        image_id: Flickr image id without extension.

    Returns:
        Caption rows containing text and entity annotations.
    """

    path = Path(root) / "annotations" / "Sentences" / f"{image_id}.txt"
    boxes = read_entity_boxes(root, image_id)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for caption_idx, line in enumerate(line.strip() for line in f if line.strip()):
            text, entities = parse_flickr_sentence(line)
            kept = []
            for entity in entities:
                entity_boxes = boxes.get(entity["entity_id"], [])
                if not entity_boxes:
                    continue
                kept.append({**entity, "boxes": entity_boxes})
            rows.append({"caption_idx": caption_idx, "text": text, "entities": kept})
    return rows


def read_entity_boxes(root: str | Path, image_id: str) -> dict[str, list[tuple[float, float, float, float]]]:
    """Read normalized Flickr30k-Entities boxes from the XML annotation file.

    Args:
        root: Flickr30k-Entities root directory.
        image_id: Flickr image id without extension.

    Returns:
        Mapping from entity id to normalized ``(x1, y1, x2, y2)`` boxes.
    """

    path = Path(root) / "annotations" / "Annotations" / f"{image_id}.xml"
    if not path.exists():
        return {}
    tree = ET.parse(path)
    root_node = tree.getroot()
    width = float(root_node.findtext("size/width", default="1"))
    height = float(root_node.findtext("size/height", default="1"))
    boxes: dict[str, list[tuple[float, float, float, float]]] = {}
    for obj in root_node.findall("object"):
        box = obj.find("bndbox")
        if box is None:
            continue
        xmin = float(box.findtext("xmin", default="0")) / max(width, 1.0)
        ymin = float(box.findtext("ymin", default="0")) / max(height, 1.0)
        xmax = float(box.findtext("xmax", default="0")) / max(width, 1.0)
        ymax = float(box.findtext("ymax", default="0")) / max(height, 1.0)
        clipped = (
            min(max(xmin, 0.0), 1.0),
            min(max(ymin, 0.0), 1.0),
            min(max(xmax, 0.0), 1.0),
            min(max(ymax, 0.0), 1.0),
        )
        for name in obj.findall("name"):
            if name.text:
                boxes.setdefault(name.text, []).append(clipped)
    return boxes


def load_image(root: str | Path, image_id: str) -> Image.Image:
    """Load one Flickr image as RGB.

    Args:
        root: Flickr30k-Entities root directory.
        image_id: Flickr image id without extension.

    Returns:
        RGB PIL image.
    """

    path = Path(root) / "images" / "Images" / f"{image_id}.jpg"
    return Image.open(path).convert("RGB")
