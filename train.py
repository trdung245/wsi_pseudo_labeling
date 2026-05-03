#!/usr/bin/env python3
"""
Main training entry point.

Three-phase training pipeline:

  Phase 1 – VMoE per dataset
  ~~~~~~~~~~~~~~~~~~~~~~~~~~
  Train one VMoE model on each of the three CSVs (balanced, neg_heavy,
  pos_heavy) independently with the uncertainty-weighted multi-task loss.

  Phase 2 – Stacking ensemble meta-learner
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Freeze the three VMoE models.  Collect their predictions on a combined
  validation set and train a shallow MetaLearner on top.

  Phase 3 – Knowledge distillation
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Use the stacking ensemble as a teacher. Train a smaller student VMoE
  (EfficientNet-B0 backbone) on the balanced dataset with the combined
  distillation + hard-label loss.

Usage
-----
  # Run all three phases:
  python train.py

  # Run a specific phase:
  python train.py --phase vmoe
  python train.py --phase ensemble
  python train.py --phase distillation

  # Override config values:
  python train.py --config configs/config.yaml
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


def get_label_map(cfg: dict) -> dict[str, int]:
    """Return label→index map from config, falling back to module-level default."""
    lbl_cfg = cfg.get("labels", {})
    if "label_to_idx" in lbl_cfg:
        return lbl_cfg["label_to_idx"]
    return LABEL_TO_IDX


def get_num_classes(cfg: dict) -> int:
    return cfg.get("labels", {}).get("num_classes", NUM_CLASSES)
from models import VMoE, PatchClassifier, StackingEnsemble, build_student
from trainers import VMoETrainer, EnsembleTrainer, DistillationTrainer
from utils import setup_logging

logger = logging.getLogger(__name__)

# Ordered list of VMoE dataset names — must match csv_files keys in config.yaml.
VMOE_DATASET_NAMES = ["balanced", "neg_heavy", "benign_heavy", "atypical_heavy", "malignant_heavy"]


def get_dataset_names(cfg: dict) -> list[str]:
    """Return dataset names from config csv_files keys, preserving order."""
    return list(cfg["data"]["csv_files"].keys())

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(dataset: PatchDataset, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights to handle imbalance."""
    from collections import Counter
    counts = Counter(r["label"] for r in dataset.records)
    total = sum(counts.values())
    weights = torch.zeros(num_classes)
    for cls_idx, n in counts.items():
        weights[cls_idx] = total / (num_classes * max(n, 1))
    return weights


def make_balanced_sampler(dataset: PatchDataset) -> torch.utils.data.WeightedRandomSampler:
    """WeightedRandomSampler that draws each class with equal probability.

    This is the most effective fix for class imbalance: every training batch
    will contain roughly equal numbers of each class, regardless of how skewed
    the underlying dataset is.  Combined with focal loss it gives the model
    both equal exposure and a loss gradient that further penalises confident
    wrong predictions on minority classes.
    """
    from collections import Counter
    counts = Counter(r["label"] for r in dataset.records)
    # Weight per sample = inverse of its class frequency
    sample_weights = torch.tensor([
        1.0 / counts[r["label"]] for r in dataset.records
    ], dtype=torch.float64)
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
        shuffle = False   # sampler and shuffle are mutually exclusive
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(sampler is not None or shuffle),
    )


def resolve_path(cfg: dict, key: str) -> Path:
    return Path(cfg["data"]["root"]) / cfg["data"][key]


# ---------------------------------------------------------------------------
# Phase 1: Train VMoE models
# ---------------------------------------------------------------------------

def phase1_train_vmoe(cfg: dict) -> dict[str, str]:
    """Train one VMoE per CSV.  Returns a dict of {dataset_name: checkpoint_path}."""
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

        # Persist best checkpoint path for Phase 2
        best_path = trainer._top_checkpoints[0][1] if trainer._top_checkpoints else None
        if best_path:
            checkpoints[ds_name] = str(best_path)
            logger.info("Best VMoE [%s] checkpoint: %s", ds_name, best_path)

    return checkpoints


# ---------------------------------------------------------------------------
# Phase 2: Train stacking meta-learner
# ---------------------------------------------------------------------------

def phase2_train_ensemble(cfg: dict, vmoe_checkpoints: dict[str, str]) -> str:
    """Train the MetaLearner on predictions from all three VMoE models."""
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

    # Build the stacking ensemble with loaded VMoE checkpoints
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

    # Meta-learner is trained on combined validation data from all datasets
    csv_paths = {
        name: Path(data_cfg["root"]) / data_cfg["csv_files"][name]
        for name in dataset_names
    }

    train_ds = CombinedPatchDataset(
        csv_paths=csv_paths,
        patch_dir=patch_dir,
        transform=train_transforms(image_size),
        split="val",    # use val split from phase 1 as meta-training data
        val_split=data_cfg["val_split"],
        test_split=data_cfg["test_split"],
        label_map=label_map,
    )
    val_ds = CombinedPatchDataset(
        csv_paths=csv_paths,
        patch_dir=patch_dir,
        transform=val_transforms(image_size),
        split="test",   # test split is the held-out meta-validation set
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
    return str(best_path)


# ---------------------------------------------------------------------------
# Phase 3: Knowledge distillation
# ---------------------------------------------------------------------------

def phase3_distillation(cfg: dict, ensemble_ckpt: str, vmoe_checkpoints: dict[str, str]) -> None:
    """Train the student VMoE via knowledge distillation from the ensemble."""
    logger.info("=" * 60)
    logger.info("PHASE 3: Knowledge distillation (ensemble → student)")
    logger.info("=" * 60)

    vmoe_cfg    = cfg["model"]["vmoe"]
    ens_cfg     = cfg["model"]["ensemble"]
    student_cfg = cfg["model"]["student"]
    data_cfg    = cfg["data"]
    batch_size  = cfg["training"]["distillation"]["batch_size"]
    num_workers = data_cfg["num_workers"]
    image_size  = data_cfg["image_size"]
    patch_dir   = Path(data_cfg["root"]) / data_cfg["patch_dir"]
    label_map   = get_label_map(cfg)
    n_classes   = get_num_classes(cfg)

    # Rebuild the teacher ensemble and load its checkpoint
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

    teacher = StackingEnsemble(
        vmoe_configs=[vmoe_model_cfg] * len(dataset_names),
        checkpoints=[c for c in ckpt_paths if c is not None] if any(ckpt_paths) else None,
        meta_hidden_dims=ens_cfg["hidden_dims"],
        meta_dropout=ens_cfg["dropout"],
        num_classes=n_classes,
    )

    # Load the meta-learner checkpoint from Phase 2
    if ensemble_ckpt and Path(ensemble_ckpt).exists():
        state = torch.load(ensemble_ckpt, map_location="cpu")
        teacher.load_state_dict(state["model"])
        logger.info("Loaded ensemble checkpoint: %s", ensemble_ckpt)

    # Build student model
    student = build_student(student_cfg, num_classes=n_classes)

    # Student trains on the balanced dataset — one clean unbiased exposure to
    # all classes. The 5 CSVs are ~67% overlapping patches so combining
    # them would just upweight repeated patches, not add real diversity.
    csv_path = Path(data_cfg["root"]) / data_cfg["csv_files"]["balanced"]

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

    class_weights = compute_class_weights(train_ds, n_classes)

    trainer = DistillationTrainer(
        student=student,
        teacher=teacher,
        train_loader=make_dataloader(train_ds, batch_size, shuffle=True,  num_workers=num_workers, balanced=True),
        val_loader=  make_dataloader(val_ds,   batch_size, shuffle=False, num_workers=num_workers),
        cfg=cfg,
        class_weights=class_weights,
    )
    trainer.train()
    trainer.load_best_checkpoint()

    best_path = trainer._top_checkpoints[0][1] if trainer._top_checkpoints else "none"
    logger.info("Best student checkpoint: %s", best_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WSI Patch Classifier Training Pipeline")
    parser.add_argument("--config", default="configs/config.yaml",
                        help="Path to the YAML config file.")
    parser.add_argument("--phase", choices=["vmoe", "ensemble", "distillation", "all"],
                        default="all", help="Which phase to run.")
    # Allow overriding checkpoint paths when resuming from a specific phase
    parser.add_argument("--vmoe-ckpts", nargs="+", default=None,
                        metavar="NAME:PATH",
                        help="Pre-trained VMoE checkpoint paths (for Phase 2/3 resume). "
                             "E.g. balanced:path/to/ckpt.pth neg_heavy:path/to/ckpt.pth")
    parser.add_argument("--ensemble-ckpt", default=None,
                        help="Pre-trained ensemble checkpoint path (for Phase 3 resume).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)

    setup_logging(cfg["logging"]["dir"], name="train")
    set_seed(cfg.get("seed", 42))

    # Parse pre-supplied checkpoint paths (for resuming)
    vmoe_ckpts: dict[str, str] = {}
    if args.vmoe_ckpts:
        for item in args.vmoe_ckpts:
            name, path = item.split(":", 1)
            vmoe_ckpts[name] = path

    ensemble_ckpt: str = args.ensemble_ckpt or ""

    if args.phase in ("vmoe", "all"):
        new_ckpts = phase1_train_vmoe(cfg)
        vmoe_ckpts.update(new_ckpts)

    if args.phase in ("ensemble", "all"):
        ensemble_ckpt = phase2_train_ensemble(cfg, vmoe_ckpts)

    if args.phase in ("distillation", "all"):
        phase3_distillation(cfg, ensemble_ckpt, vmoe_ckpts)

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
