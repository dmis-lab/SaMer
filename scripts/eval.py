import argparse
import json
from pathlib import Path
import time

import torch
from tqdm import tqdm

from samer.cache_io import load_cache
from samer.metrics import text_to_image_metrics
from samer.scoring import mean_maxsim


def pad_tokens(items: list[torch.Tensor]) -> torch.Tensor:
    """Pad variable-length token tensors into a batch tensor.

    Args:
        items: List of token tensors with shape ``[seq_len, dim]``.

    Returns:
        Padded tensor with shape ``[batch, max_seq_len, dim]``.
    """

    return torch.nn.utils.rnn.pad_sequence([x.float() for x in items], batch_first=True)


def pad_mask(items: list[torch.Tensor]) -> torch.Tensor:
    """Pad variable-length boolean masks into a batch mask.

    Args:
        items: List of one-dimensional boolean masks.

    Returns:
        Padded boolean mask with shape ``[batch, max_seq_len]``.
    """

    return torch.nn.utils.rnn.pad_sequence([x.bool() for x in items], batch_first=True)


@torch.inference_mode()
def main() -> None:
    """Evaluate text-to-image retrieval from a saved SaMer cache.

    Args:
        None.

    Returns:
        None.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--score-chunk-size", type=int, default=128)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--full-metrics-json", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    payload = load_cache(Path(args.cache_dir) / f"{args.split}.pt")
    image_ids = list(payload["images"].keys())
    image_index = {image_id: idx for idx, image_id in enumerate(image_ids)}

    image_tokens = pad_tokens([payload["images"][image_id]["tokens"] for image_id in image_ids]).to(device)
    image_mask = pad_mask(
        [
            payload["images"][image_id].get(
                "mask",
                torch.ones(payload["images"][image_id]["tokens"].size(0), dtype=torch.bool),
            )
            for image_id in image_ids
        ]
    ).to(device)
    scores = []
    positives = []

    started = time.perf_counter()
    queries = payload["queries"]
    for start in tqdm(range(0, len(queries), args.batch_size), desc=f"{args.split}: score"):
        batch = queries[start : start + args.batch_size]
        query_tokens = pad_tokens([query["tokens"] for query in batch]).to(device)
        query_mask = torch.nn.utils.rnn.pad_sequence([query["mask"].bool() for query in batch], batch_first=True).to(device)
        scores.append(
            mean_maxsim(
                query_tokens,
                image_tokens,
                query_mask=query_mask,
                image_mask=image_mask,
                chunk_size=args.score_chunk_size,
            ).cpu()
        )
        pos = torch.zeros(len(batch), len(image_ids), dtype=torch.bool)
        for row, query in enumerate(batch):
            pos[row, image_index[str(query["image_id"])]] = True
        positives.append(pos)

    score_matrix = torch.cat(scores, dim=0)
    positive_mask = torch.cat(positives, dim=0)
    metrics = text_to_image_metrics(score_matrix, positive_mask)
    metrics["latency_seconds"] = round(time.perf_counter() - started, 3)
    metrics["num_images"] = len(image_ids)
    metrics["num_queries"] = len(queries)
    if "metadata" in payload:
        metrics.update(payload["metadata"])
    if args.full_metrics_json is not None:
        full_metrics = json.loads(Path(args.full_metrics_json).read_text(encoding="utf-8"))
        for key in ("r@1", "r@5", "r@10", "mrr", "ndcg"):
            denom = full_metrics.get(key, 0.0)
            metrics[f"relative_{key}"] = round(metrics[key] / denom, 6) if denom else None

    print(json.dumps(metrics, indent=2, sort_keys=True))
    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
