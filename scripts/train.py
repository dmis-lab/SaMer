import argparse
import os
from pathlib import Path
from typing import Any

from accelerate.utils import extract_model_from_parallel
import torch
from transformers import Trainer, TrainingArguments
import yaml

from samer.colpali import load_colpali, save_projector_checkpoint
from samer.merging import MergeConfig
from samer.train_data import ColPaliTrainCollator, FlickrCaptionDataset
from samer.training import MERGE_DIAGNOSTIC_KEYS, MergedColPaliForTraining


class SaMerTrainer(Trainer):
    """Trainer that logs SaMer diagnostics while optimizing retrieval loss.

    Args:
        *args: Positional arguments forwarded to ``transformers.Trainer``.
        **kwargs: Keyword arguments forwarded to ``transformers.Trainer``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._log_sums: dict[str, float] = {}
        self._log_count = 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        self._accumulate_logs(outputs)
        return (outputs["loss"], outputs) if return_outputs else outputs["loss"]

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs["loss"].detach().mean()
            self._accumulate_logs(outputs)
        return loss, None, None

    def log(self, logs: dict[str, Any], *args, **kwargs) -> None:
        if ("loss" in logs or "eval_loss" in logs) and self._log_count > 0:
            for key, value in self._log_sums.items():
                logs[key] = round(value / self._log_count, 6)
            self._log_sums.clear()
            self._log_count = 0
        super().log(logs, *args, **kwargs)

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False):
        output_dir = output_dir or self.args.output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        model = extract_model_from_parallel(self.model)
        save_projector_checkpoint(model.colpali, output_dir)
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)

    def _accumulate_logs(self, outputs: dict[str, Any]) -> None:
        keys = (
            "retrieval_loss",
            "grad_norm_projector",
            "avg_empty_clusters",
            *MERGE_DIAGNOSTIC_KEYS,
        )
        logged = False
        for key in keys:
            value = outputs.get(key)
            if value is None:
                continue
            if torch.is_tensor(value):
                value = value.detach().float().mean().cpu().item()
            self._log_sums[key] = self._log_sums.get(key, 0.0) + float(value)
            logged = True
        if logged:
            self._log_count += 1


def main() -> None:
    """Command-line entrypoint for projector-only SaMer training.

    Args:
        None.

    Returns:
        None.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    merge_cfg = cfg["merge"]
    loss_cfg = cfg["loss"]

    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if torch.cuda.is_available() and local_rank >= 0:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model, processor = load_colpali(
        model_cfg["model_name_or_path"],
        device,
        local_files_only=bool(model_cfg.get("local_files_only", True)),
        freeze=False,
        eval_mode=False,
    )
    _freeze_all(base_model)
    _unfreeze_matching(base_model, list(model_cfg.get("projector_train_patterns", ["embedding_proj_layer"])))
    if hasattr(base_model, "config"):
        base_model.config.use_cache = False

    model = MergedColPaliForTraining(
        base_model,
        MergeConfig(
            num_regions=int(merge_cfg.get("num_regions", 64)),
            cluster_iters=int(merge_cfg.get("cluster_iters", 3)),
            spatial_weight=float(merge_cfg.get("spatial_weight", 0.1)),
            assignment_temperature=float(merge_cfg.get("assignment_temperature", 0.07)),
            seed=int(merge_cfg.get("seed", 42)),
        ),
        temperature=float(loss_cfg.get("temperature", 0.07)),
    )
    if int(os.environ.get("RANK", "0")) == 0:
        _print_trainable_summary(model)

    train_dataset = FlickrCaptionDataset(
        data_cfg["root"],
        data_cfg.get("train_split", "train"),
        data_cfg.get("limit_train_images"),
        include_entities=True,
    )
    eval_dataset = FlickrCaptionDataset(
        data_cfg["root"],
        data_cfg.get("eval_split", "val"),
        data_cfg.get("limit_eval_images"),
        include_entities=True,
    )
    collator = ColPaliTrainCollator(
        processor=processor,
        data_root=data_cfg["root"],
        max_bbox_phrases_per_caption=int(data_cfg.get("max_bbox_phrases_per_caption", 4)),
        min_bbox_area=float(data_cfg.get("min_bbox_area", 0.005)),
        max_bbox_area=float(data_cfg.get("max_bbox_area", 1.0)),
    )

    output_dir = train_cfg["output_dir"]
    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=train_cfg.get("run_name"),
        per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 16)),
        per_device_eval_batch_size=int(train_cfg.get("per_device_eval_batch_size", train_cfg.get("per_device_train_batch_size", 16))),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 4)),
        learning_rate=float(train_cfg.get("learning_rate", 1.0e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.05)),
        num_train_epochs=float(train_cfg.get("num_train_epochs", 3)),
        bf16=bool(train_cfg.get("bf16", True)),
        logging_steps=int(train_cfg.get("logging_steps", 20)),
        eval_strategy=train_cfg.get("eval_strategy", "epoch"),
        save_strategy=train_cfg.get("save_strategy", "epoch"),
        save_total_limit=int(train_cfg.get("save_total_limit", 2)),
        remove_unused_columns=False,
        dataloader_num_workers=int(train_cfg.get("dataloader_num_workers", 0)),
        report_to=train_cfg.get("report_to", "none"),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
    )
    trainer = SaMerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=processor,
    )
    trainer.train(resume_from_checkpoint=train_cfg.get("resume_from_checkpoint"))
    trainer.save_model(output_dir)


def _freeze_all(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad_(False)


def _unfreeze_matching(model: torch.nn.Module, patterns: list[str]) -> None:
    matched = 0
    for name, param in model.named_parameters():
        if any(pattern in name for pattern in patterns):
            param.requires_grad_(True)
            matched += param.numel()
    if matched == 0:
        print(f"[warn] no projector parameters matched patterns={patterns}")


def _print_trainable_summary(model: torch.nn.Module) -> None:
    trainable = 0
    total = 0
    for param in model.parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    print(f"trainable parameters: {trainable:,} / {total:,} ({100.0 * trainable / max(total, 1):.4f}%)")


if __name__ == "__main__":
    main()
