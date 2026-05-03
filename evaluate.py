#!/usr/bin/env python3
"""
Evaluation script.

Evaluates any trained model (VMoE, ensemble, or student) on the test split and
prints a full classification report + regression metrics.

Usage
-----
  # Evaluate the stacking ensemble:
  python evaluate.py --model ensemble --checkpoint checkpoints/ensemble/best.pth

  # Evaluate the student (distilled) model:
  python evaluate.py --model student --checkpoint checkpoints/distillation/best.pth

  # Evaluate a single VMoE:
  python evaluate.py --model vmoe --checkpoint checkpoints/vmoe_balanced/best.pth
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import classification_report

from data import (
    PatchDataset, CombinedPatchDataset,
    val_transforms, LABEL_TO_IDX, IDX_TO_LABEL, NUM_CLASSES,
)
from models import VMoE, StackingEnsemble, build_student
from trainers.base_trainer import get_device
from utils import setup_logging, aggregate_metrics

logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(model_type: str, cfg: dict, checkpoint: str) -> torch.nn.Module:
    vmoe_cfg    = cfg["model"]["vmoe"]
    student_cfg = cfg["model"]["student"]
    ens_cfg     = cfg["model"]["ensemble"]

    vmoe_base_cfg = dict(
        backbone=vmoe_cfg["backbone"],
        pretrained=False,
        num_experts=vmoe_cfg["num_experts"],
        top_k=vmoe_cfg["top_k"],
        num_classes=NUM_CLASSES,
        load_balance_coeff=vmoe_cfg["load_balance_coeff"],
        expert_dropout=vmoe_cfg["expert_dropout"],
        cls_head_hidden=vmoe_cfg["cls_head_hidden"],
        attn_head_hidden=vmoe_cfg["attn_head_hidden"],
    )

    if model_type == "vmoe":
        model = VMoE(**vmoe_base_cfg)
    elif model_type == "ensemble":
        model = StackingEnsemble(
            vmoe_configs=[vmoe_base_cfg] * 3,
            checkpoints=None,   # VMoE weights embedded in ensemble checkpoint
            meta_hidden_dims=ens_cfg["hidden_dims"],
            meta_dropout=ens_cfg["dropout"],
            num_classes=NUM_CLASSES,
        )
    elif model_type == "student":
        model = build_student(student_cfg, num_classes=NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["model"])
    return model


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    all_labels: list[int] = []
    all_logits: list[list[float]] = []
    all_attn_pred: list[float] = []
    all_attn_true: list[float] = []

    for batch in loader:
        images    = batch["image"].to(device)
        labels    = batch["label"]
        attentions = batch["attention"]

        out = model(images)
        logits     = out["logits"].cpu().numpy()
        attn_score = out["attn_score"].cpu().squeeze(-1).numpy()

        all_labels.extend(labels.numpy().tolist())
        all_logits.extend(logits.tolist())
        all_attn_pred.extend(attn_score.tolist())
        all_attn_true.extend(attentions.numpy().tolist())

    metrics = aggregate_metrics(
        all_labels, all_logits, all_attn_pred, all_attn_true,
        class_names=list(IDX_TO_LABEL.values()),
    )

    # Full sklearn classification report
    preds = np.array(all_logits).argmax(axis=-1)
    report = classification_report(
        all_labels, preds,
        target_names=list(IDX_TO_LABEL.values()),
        zero_division=0,
    )
    return metrics, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WSI Patch Classifier Evaluation")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model", choices=["vmoe", "ensemble", "student"], required=True)
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--dataset", choices=["balanced", "neg_heavy", "pos_heavy", "all"],
                        default="all", help="Which CSV(s) to evaluate on.")
    parser.add_argument("--output", default=None, help="Optional JSON file to save metrics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)
    setup_logging(cfg["logging"]["dir"], name="evaluate")

    device    = get_device(cfg.get("device", "auto"))
    image_size = cfg["data"]["image_size"]
    patch_dir  = Path(cfg["data"]["root"]) / cfg["data"]["patch_dir"]
    data_cfg   = cfg["data"]

    model = build_model(args.model, cfg, args.checkpoint)
    model.to(device)
    logger.info("Loaded %s model from %s", args.model, args.checkpoint)

    if args.dataset == "all":
        csv_paths = {
            name: Path(data_cfg["root"]) / data_cfg["csv_files"][name]
            for name in ["balanced", "neg_heavy", "pos_heavy"]
        }
        ds = CombinedPatchDataset(
            csv_paths=csv_paths,
            patch_dir=patch_dir,
            transform=val_transforms(image_size),
            split=args.split,
            val_split=data_cfg["val_split"],
            test_split=data_cfg["test_split"],
        )
    else:
        csv_path = Path(data_cfg["root"]) / data_cfg["csv_files"][args.dataset]
        ds = PatchDataset(
            csv_path, patch_dir,
            transform=val_transforms(image_size),
            split=args.split,
            val_split=data_cfg["val_split"],
            test_split=data_cfg["test_split"],
        )

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=cfg["training"]["vmoe"]["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )

    metrics, report = evaluate(model, loader, device)

    logger.info("\n%s\n", report)
    logger.info("Aggregate metrics:")
    for k, v in sorted(metrics.items()):
        logger.info("  %-25s %.4f", k, v)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Metrics saved to %s", args.output)


if __name__ == "__main__":
    main()
