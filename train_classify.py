#!/usr/bin/env python3
"""
Two-phase classification training pipeline (Phase 1 + Phase 2 only).

  Phase 1 – VMoE per CSV dataset
  Phase 2 – Stacking ensemble meta-learner

No knowledge distillation (Phase 3).

Usage
-----
  # Run both phases with default config (configs/config.yaml):
  python train_classify.py

  # Use 3-class config:
  python train_classify.py --config configs/config_3class.yaml

  # Run only Phase 1:
  python train_classify.py --phase vmoe

  # Run only Phase 2 (resume with existing VMoE checkpoints):
  python train_classify.py --phase ensemble \
      --vmoe-ckpts balanced:checkpoints/vmoe/epoch_010_f1_0.84.pth \
                   neg_heavy:checkpoints/vmoe/epoch_008_f1_0.82.pth \
                   ...
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from data import (
    PatchDataset, CombinedPatchDataset,
    train_transforms, val_transforms,
    LABEL_TO_IDX, NUM_CLASSES,
)
from models import VMoE, StackingEnsemble
from trainers import VMoETrainer, EnsembleTrainer
from utils import setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_label_map(cfg: dict) -> dict[str, int]:
    lbl_cfg = cfg.get("labels", {})
    if "label_to_idx" in lbl_cfg:
        return lbl_cfg["label_to_idx"]
    return LABEL_TO_IDX


def get_num_classes(cfg: dict) -> int:
    return cfg.get("labels", {}).get("num_classes", NUM_CLASSES)


def get_dataset_names(cfg: dict) -> list[str]:
    return list(cfg["data"]["csv_files"].keys())


# ---------------------------------------------------------------------------
# Seed & class-weight helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(dataset: PatchDataset, num_classes: int) -> torch.Tensor:
    from collections import Counter
    counts = Counter(r["label"] for r in dataset.records)
    total  = sum(counts.values())
    weights = torch.zeros(num_classes)
    for cls_idx, n in counts.items():
        weights[cls_idx] = total / (num_classes * max(n, 1))
    return weights


def make_balanced_sampler(dataset: PatchDataset) -> torch.utils.data.WeightedRandomSampler:
    from collections import Counter
    counts = Counter(r["label"] for r in dataset.records)
    sample_weights = torch.tensor(
        [1.0 / counts[r["label"]] for r in dataset.records],
        dtype=torch.float64,
    )
    return torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True,
    )


def make_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    balanced: bool = False,
) -> torch.utils.data.DataLoader:
    sampler = None
    if balanced and shuffle:
        sampler = make_balanced_sampler(dataset)
        shuffle = False
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(sampler is not None or shuffle),
    )


# ---------------------------------------------------------------------------
# Phase 1: Train one VMoE per CSV
# ---------------------------------------------------------------------------

def phase1_train_vmoe(cfg: dict) -> dict[str, str]:
    """Train one VMoE per CSV. Returns {dataset_name: best_checkpoint_path}."""
    logger.info("=" * 60)
    logger.info("PHASE 1: Training VMoE models")
    logger.info("=" * 60)

    vmoe_cfg    = cfg["model"]["vmoe"]
    data_cfg    = cfg["data"]
    batch_size  = cfg["training"]["vmoe"]["batch_size"]
    num_workers = data_cfg["num_workers"]
    image_size  = data_cfg["image_size"]
    patch_dir   = Path(data_cfg["root"]) / data_cfg["patch_dir"]
    label_map   = get_label_map(cfg)
    n_classes   = get_num_classes(cfg)

    dataset_names = get_dataset_names(cfg)
    checkpoints: dict[str, str] = {}

    for ds_name in dataset_names:
        csv_path = Path(data_cfg["root"]) / data_cfg["csv_files"][ds_name]
        logger.info("--- Dataset: %s (%s) ---", ds_name, csv_path)

        train_ds = PatchDataset(
            csv_path, patch_dir,
            transform=train_transforms(image_size),
            split="train",
            val_split=data_cfg["val_split"],
            test_split=data_cfg["test_split"],
            label_map=label_map,
        )
        val_ds = PatchDataset(
            csv_path, patch_dir,
            transform=val_transforms(image_size),
            split="val",
            val_split=data_cfg["val_split"],
            test_split=data_cfg["test_split"],
            label_map=label_map,
        )

        if len(train_ds) == 0:
            logger.warning("No training samples for %s; skipping.", ds_name)
            continue

        class_weights = compute_class_weights(train_ds, n_classes)
        logger.info("Class weights for %s: %s", ds_name, class_weights.tolist())

        model = VMoE(
            backbone=vmoe_cfg["backbone"],
            pretrained=vmoe_cfg["pretrained"],
            num_experts=vmoe_cfg["num_experts"],
            top_k=vmoe_cfg["top_k"],
            num_classes=n_classes,
            load_balance_coeff=vmoe_cfg["load_balance_coeff"],
            expert_dropout=vmoe_cfg["expert_dropout"],
            freeze_blocks=vmoe_cfg.get("freeze_blocks", 5),
            cls_head_hidden=vmoe_cfg["cls_head_hidden"],
            attn_head_hidden=vmoe_cfg["attn_head_hidden"],
        )

        trainer = VMoETrainer(
            model=model,
            train_loader=make_dataloader(train_ds, batch_size, shuffle=True,  num_workers=num_workers, balanced=True),
            val_loader=  make_dataloader(val_ds,   batch_size, shuffle=False, num_workers=num_workers),
            cfg=cfg,
            class_weights=class_weights,
            dataset_name=ds_name,
        )
        trainer.train()
        trainer.load_best_checkpoint()

        best_path = trainer._top_checkpoints[0][1] if trainer._top_checkpoints else None
        if best_path:
            checkpoints[ds_name] = str(best_path)
            logger.info("Best VMoE [%s] checkpoint: %s", ds_name, best_path)

    return checkpoints


# ---------------------------------------------------------------------------
# Phase 2: Train stacking ensemble meta-learner
# ---------------------------------------------------------------------------

def phase2_train_ensemble(cfg: dict, vmoe_checkpoints: dict[str, str]) -> str:
    """Train the MetaLearner on combined val predictions. Returns best checkpoint path."""
    logger.info("=" * 60)
    logger.info("PHASE 2: Training stacking ensemble meta-learner")
    logger.info("=" * 60)

    vmoe_cfg    = cfg["model"]["vmoe"]
    ens_cfg     = cfg["model"]["ensemble"]
    data_cfg    = cfg["data"]
    batch_size  = cfg["training"]["ensemble"]["batch_size"]
    num_workers = data_cfg["num_workers"]
    image_size  = data_cfg["image_size"]
    patch_dir   = Path(data_cfg["root"]) / data_cfg["patch_dir"]
    label_map   = get_label_map(cfg)
    n_classes   = get_num_classes(cfg)

    vmoe_model_cfg = dict(
        backbone=vmoe_cfg["backbone"],
        pretrained=False,
        num_experts=vmoe_cfg["num_experts"],
        top_k=vmoe_cfg["top_k"],
        num_classes=n_classes,
        load_balance_coeff=vmoe_cfg["load_balance_coeff"],
        expert_dropout=vmoe_cfg["expert_dropout"],
        freeze_blocks=0,
        cls_head_hidden=vmoe_cfg["cls_head_hidden"],
        attn_head_hidden=vmoe_cfg["attn_head_hidden"],
    )

    dataset_names = get_dataset_names(cfg)
    ckpt_paths = [vmoe_checkpoints.get(n) for n in dataset_names]

    for name, ckpt in zip(dataset_names, ckpt_paths):
        if ckpt is None:
            logger.warning("No checkpoint for VMoE [%s]; base model will use random weights.", name)

    ensemble = StackingEnsemble(
        vmoe_configs=[vmoe_model_cfg] * len(dataset_names),
        checkpoints=ckpt_paths if any(ckpt_paths) else None,
        meta_hidden_dims=ens_cfg["hidden_dims"],
        meta_dropout=ens_cfg["dropout"],
        num_classes=n_classes,
    )

    csv_paths = {
        name: Path(data_cfg["root"]) / data_cfg["csv_files"][name]
        for name in dataset_names
    }

    # MetaLearner trains on the Phase-1 val split, evaluates on the test split
    train_ds = CombinedPatchDataset(
        csv_paths=csv_paths,
        patch_dir=patch_dir,
        transform=train_transforms(image_size),
        split="val",
        val_split=data_cfg["val_split"],
        test_split=data_cfg["test_split"],
        label_map=label_map,
    )
    val_ds = CombinedPatchDataset(
        csv_paths=csv_paths,
        patch_dir=patch_dir,
        transform=val_transforms(image_size),
        split="test",
        val_split=data_cfg["val_split"],
        test_split=data_cfg["test_split"],
        label_map=label_map,
    )

    trainer = EnsembleTrainer(
        model=ensemble,
        train_loader=make_dataloader(train_ds, batch_size, shuffle=True,  num_workers=num_workers),
        val_loader=  make_dataloader(val_ds,   batch_size, shuffle=False, num_workers=num_workers),
        cfg=cfg,
    )
    trainer.train()
    trainer.load_best_checkpoint()

    best_path = trainer._top_checkpoints[0][1] if trainer._top_checkpoints else "none"
    logger.info("Best ensemble checkpoint: %s", best_path)
    return str(best_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VMoE + Stacking Ensemble training (Phase 1 & 2 only)"
    )
    parser.add_argument(
        "--config", default="configs/config.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--phase", choices=["vmoe", "ensemble", "all"],
        default="all",
        help="Which phase to run (default: all).",
    )
    parser.add_argument(
        "--vmoe-ckpts", nargs="+", default=None,
        metavar="NAME:PATH",
        help="Pre-trained VMoE checkpoint paths for Phase 2 resume. "
             "Format: name:path/to/ckpt.pth  (one per dataset)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)

    setup_logging(cfg["logging"]["dir"], name="train_classify")
    set_seed(cfg.get("seed", 42))

    vmoe_ckpts: dict[str, str] = {}
    if args.vmoe_ckpts:
        for item in args.vmoe_ckpts:
            name, path = item.split(":", 1)
            vmoe_ckpts[name] = path

    if args.phase in ("vmoe", "all"):
        new_ckpts = phase1_train_vmoe(cfg)
        vmoe_ckpts.update(new_ckpts)

    if args.phase in ("ensemble", "all"):
        phase2_train_ensemble(cfg, vmoe_ckpts)

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
