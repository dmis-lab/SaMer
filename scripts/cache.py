import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from samer.cache_io import save_cache
from samer.colpali import encode_images, encode_texts, load_colpali
from samer.coords import dtype_from_name, make_visual_coords
from samer.flickr import load_image, read_captions, read_split_ids
from samer.merging import MergeConfig, merge_tokens


def build_split(args, split: str, model, processor, device: torch.device) -> None:
    """Encode one dataset split and save a bbox-free SaMer retrieval cache.

    Args:
        args: Parsed command-line arguments.
        split: Dataset split to encode.
        model: ColPali-style retrieval model.
        processor: Matching Hugging Face processor.
        device: Device used for encoding.

    Returns:
        None.
    """

    image_ids = read_split_ids(args.data_root, split)
    if args.limit is not None:
        image_ids = image_ids[: args.limit]
    out_dtype = dtype_from_name(args.cache_dtype)
    config = MergeConfig(
        num_regions=args.num_regions,
        cluster_iters=args.cluster_iters,
        spatial_weight=args.spatial_weight,
        assignment_temperature=args.assignment_temperature,
    )

    images = {}
    queries = []
    token_counts = []
    empty_counts = []

    for start in tqdm(range(0, len(image_ids), args.batch_size), desc=f"{split}: images"):
        batch_ids = image_ids[start : start + args.batch_size]
        batch_images = [load_image(args.data_root, image_id) for image_id in batch_ids]
        embeddings = encode_images(model, processor, batch_images, device)
        for image_id, tokens in zip(batch_ids, embeddings):
            coords = make_visual_coords(tokens.size(0))
            full_num_tokens = tokens.size(0)
            merged = merge_tokens(tokens, coords, config)
            tokens = merged["tokens"]
            coords = merged["coords"]
            empty_counts.append(int(merged["num_empty_clusters"].item()))
            token_counts.append((full_num_tokens, tokens.size(0)))
            images[str(image_id)] = {
                "tokens": tokens.to(out_dtype),
                "coords": coords,
                "mask": torch.ones(tokens.size(0), dtype=torch.bool),
                "full_num_tokens": full_num_tokens,
            }

    caption_rows = []
    for image_id in image_ids:
        for caption_idx, caption in enumerate(read_captions(args.data_root, image_id)):
            caption_rows.append((str(image_id), caption_idx, caption))

    for start in tqdm(range(0, len(caption_rows), args.batch_size), desc=f"{split}: text"):
        rows = caption_rows[start : start + args.batch_size]
        query_tokens, query_masks = encode_texts(model, processor, [row[2] for row in rows], device)
        for (image_id, caption_idx, caption), tokens, mask in zip(rows, query_tokens, query_masks):
            queries.append(
                {
                    "caption_id": f"{image_id}_{caption_idx}",
                    "image_id": image_id,
                    "text": caption,
                    "tokens": tokens.to(out_dtype),
                    "mask": mask,
                }
            )

    full_avg = sum(x[0] for x in token_counts) / max(len(token_counts), 1)
    comp_avg = sum(x[1] for x in token_counts) / max(len(token_counts), 1)
    payload = {
        "split": split,
        "kind": "samer_cache",
        "model_name": args.model_name,
        "images": images,
        "queries": queries,
        "metadata": {
            "num_regions": args.num_regions,
            "cluster_iters": args.cluster_iters,
            "spatial_weight": args.spatial_weight,
            "assignment_temperature": args.assignment_temperature,
            "avg_full_tokens": full_avg,
            "avg_compressed_tokens": comp_avg,
            "avg_empty_clusters": sum(empty_counts) / max(len(empty_counts), 1) if empty_counts else 0.0,
            "compression_factor": full_avg / max(comp_avg, 1e-9),
            "memory_ratio": comp_avg / max(full_avg, 1e-9),
        },
    }
    save_cache(payload, Path(args.output_dir) / f"{split}.pt")


def main() -> None:
    """Command-line entrypoint for building compressed SaMer caches.

    Args:
        None.

    Returns:
        None.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-name", default="vidore/colpali-v1.3-hf")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--output-dir", default="outputs/flickr_samer_k64")
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cache-dtype", default="bfloat16", choices=["float32", "float", "bfloat16", "bf16", "float16", "fp16"])
    parser.add_argument("--num-regions", type=int, default=64)
    parser.add_argument("--cluster-iters", type=int, default=3)
    parser.add_argument("--spatial-weight", type=float, default=0.1)
    parser.add_argument("--assignment-temperature", type=float, default=0.07)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    model, processor = load_colpali(
        args.model_name,
        device,
        local_files_only=args.local_files_only,
        adapter_path=args.adapter_path,
    )
    for split in args.splits:
        build_split(args, split, model, processor, device)


if __name__ == "__main__":
    main()
