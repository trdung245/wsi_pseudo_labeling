#!/usr/bin/env python3
"""
Classification-only two-phase training pipeline (no attention regression).

  Phase 1 – VMoE per CSV  (CB Focal loss + load-balance, NO regression)
  Phase 2 – Stacking ensemble meta-learner  (CE loss only)

Usage
-----
  # Run both phases (4-class):
  python train_cls_only.py

  # Use 3-class config:
  python train_cls_only.py --config configs/config_3class.yaml

  # Run only Phase 1:
  python train_cls_only.py --phase vmoe

  # Run only Phase 2 (resume with existing VMoE checkpoints):
  python train_cls_only.py --phase ensemble \
      --vmoe-ckpts balanced:checkpoints/vmoe/epoch_010_f1_0.84.pth \
                   neg_heavy:checkpoints/vmoe/epoch_008_f1_0.82.pth ...
"""

from __future__ import annotations

import argparse
import logging
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from data import (
    PatchDataset, CombinedPatchDataset,
    train_transforms, val_transforms,
    LABEL_TO_IDX, NUM_CLASSES,
)
from models import VMoE, StackingEnsemble
from trainers.base_trainer import BaseTrainer
from utils import setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Classification-only losses
# ---------------------------------------------------------------------------

class CBFocalLoss(nn.Module):
    """Class-Balanced Focal Loss — classification only, no regression.

    Parameters
    ----------
    samples_per_class:
        Raw training-set counts per class (in label-index order).
    num_classes:
        Number of label classes.
    beta:
        CB smoothing hyperparameter (Cui et al., CVPR 2019).
    focal_gamma:
        Focal focusing parameter γ.
    load_balance_coeff:
        Multiplier for the VMoE auxiliary load-balancing loss.
    """

    def __init__(
        self,
        samples_per_class: list[int] | np.ndarray,
        num_classes: int = 4,
        beta: float = 0.9999,
        focal_gamma: float = 2.0,
        load_balance_coeff: float = 0.01,
    ) -> None:
        super().__init__()
        self.focal_gamma = focal_gamma
        self.load_balance_coeff = load_balance_coeff

        samples = np.array(samples_per_class, dtype=np.float64)
        effective_num = 1.0 - np.power(beta, samples)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * num_classes
        self.register_buffer("class_weights", torch.tensor(weights, dtype=torch.float32))

    def forward(
        self,
        logits: torch.Tensor,                    # (B, C)
        labels: torch.Tensor,                    # (B,) long
        load_bal_loss: torch.Tensor | None = None,
        **_ignored,                              # absorb attn_score / attentions kwargs
    ) -> dict[str, torch.Tensor]:
        ce = F.cross_entropy(logits, labels, weight=self.class_weights, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.focal_gamma * ce).mean()

        total = focal
        lb = torch.tensor(0.0, device=logits.device)
        if load_bal_loss is not None:
            lb = self.load_balance_coeff * load_bal_loss
            total = total + lb

        return {
            "total":         total,
            "cls_loss":      focal.detach(),
            "load_bal_loss": lb.detach(),
        }


class CELoss(nn.Module):
    """Plain cross-entropy for the ensemble meta-learner — classification only."""

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        **_ignored,
    ) -> dict[str, torch.Tensor]:
        loss = F.cross_entropy(logits, labels)
        return {"total": loss, "cls_loss": loss.detach()}


# ---------------------------------------------------------------------------
# Zero-attn helpers
# BaseTrainer._run_epoch expects _train_step / _eval_step to return
#   (loss_dict, logits, attn_pred, labels, attns)
# and later passes attn_pred / attns to aggregate_metrics.
# Since we have no regression, we return zero tensors of the right shape.
# ---------------------------------------------------------------------------

def _zero_attn(batch_size: int, device: torch.device) -> torch.Tensor:
    return torch.zeros(batch_size, 1, device=device)


# ---------------------------------------------------------------------------
# Phase-1 trainer: VMoE, classification only
# ---------------------------------------------------------------------------

class ClassifyVMoETrainer(BaseTrainer):
    """Phase 1 trainer with CB Focal loss only — no attention regression."""

    def __init__(
        self,
        model: VMoE,
        train_loader,
        val_loader,
        cfg: dict,
        class_weights: torch.Tensor | None = None,
        dataset_name: str = "balanced",
    ) -> None:
        super().__init__(model, train_loader, val_loader, cfg, phase_name="vmoe")
        self.phase_name = f"vmoe_{dataset_name}"

        num_classes = cfg["labels"]["num_classes"]
        loss_cfg    = cfg.get("loss", {})
        counts_map  = Counter(r["label"] for r in train_loader.dataset.records)
        samples_per_class = [counts_map.get(i, 1) for i in range(num_classes)]

        self.criterion = CBFocalLoss(
            samples_per_class=samples_per_class,
            num_classes=num_classes,
            focal_gamma=loss_cfg.get("focal_gamma", 2.0),
            load_balance_coeff=cfg["model"]["vmoe"]["load_balance_coeff"],
        ).to(self.device)

        # val_criterion must accept the base_trainer's call signature:
        #   (logits, attn_score, labels, attentions)
        self.val_criterion = self.criterion

    def _train_step(self, batch):
        images = batch["image"]
        labels = batch["label"]

        out = self.model(images)
        loss_dict = self.criterion(
            logits=out["logits"],
            labels=labels,
            load_bal_loss=out["load_bal_loss"],
        )

        attn_dummy = _zero_attn(images.size(0), self.device)
        attn_true  = torch.zeros(images.size(0), device=self.device)
        return loss_dict, out["logits"], attn_dummy, labels, attn_true

    def _eval_step(self, batch):
        images = batch["image"]
        labels = batch["label"]

        out = self.model(images)
        attn_dummy = _zero_attn(images.size(0), self.device)
        attn_true  = torch.zeros(images.size(0), device=self.device)
        return out["logits"], attn_dummy, labels, attn_true


# ---------------------------------------------------------------------------
# Phase-2 trainer: Ensemble, classification only
# ---------------------------------------------------------------------------

class ClassifyEnsembleTrainer(BaseTrainer):
    """Phase 2 trainer with plain CE loss — no attention regression."""

    def __init__(
        self,
        model: StackingEnsemble,
        train_loader,
        val_loader,
        cfg: dict,
    ) -> None:
        super().__init__(model, train_loader, val_loader, cfg, phase_name="ensemble")
        self.criterion = CELoss().to(self.device)
        self.val_criterion = self.criterion

    def _train_step(self, batch):
        images = batch["image"]
        labels = batch["label"]

        out = self.model(images)
        loss_dict = self.criterion(logits=out["logits"], labels=labels)

        attn_dummy = _zero_attn(images.size(0), self.device)
        attn_true  = torch.zeros(images.size(0), device=self.device)
        return loss_dict, out["logits"], attn_dummy, labels, attn_true

    def _eval_step(self, batch):
        images = batch["image"]
        labels = batch["label"]

        out = self.model(images)
        attn_dummy = _zero_attn(images.size(0), self.device)
        attn_true  = torch.zeros(images.size(0), device=self.device)
        return out["logits"], attn_dummy, labels, attn_true


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_label_map(cfg: dict) -> dict[str, int]:
    lbl_cfg = cfg.get("labels", {})
    return lbl_cfg["label_to_idx"] if "label_to_idx" in lbl_cfg else LABEL_TO_IDX


def get_num_classes(cfg: dict) -> int:
    return cfg.get("labels", {}).get("num_classes", NUM_CLASSES)


def get_dataset_names(cfg: dict) -> list[str]:
    return list(cfg["data"]["csv_files"].keys())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(dataset: PatchDataset, num_classes: int) -> torch.Tensor:
    counts = Counter(r["label"] for r in dataset.records)
    total  = sum(counts.values())
    weights = torch.zeros(num_classes)
    for cls_idx, n in counts.items():
        weights[cls_idx] = total / (num_classes * max(n, 1))
    return weights


def make_balanced_sampler(dataset: PatchDataset) -> torch.utils.data.WeightedRandomSampler:
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
# Phase 1
# ---------------------------------------------------------------------------

def phase1_train_vmoe(cfg: dict, skip_datasets: list[str] | None = None) -> dict[str, str]:
    """Train one VMoE per CSV (classification only). Returns {name: ckpt_path}."""
    logger.info("=" * 60)
    logger.info("PHASE 1: Training VMoE models (classification only)")
    logger.info("=" * 60)

    vmoe_cfg    = cfg["model"]["vmoe"]
    data_cfg    = cfg["data"]
    batch_size  = cfg["training"]["vmoe"]["batch_size"]
    num_workers = data_cfg["num_workers"]
    image_size  = data_cfg["image_size"]
    patch_dir   = Path(data_cfg["root"]) / data_cfg["patch_dir"]
    label_map   = get_label_map(cfg)
    n_classes   = get_num_classes(cfg)

    skip_set = set(skip_datasets or [])
    checkpoints: dict[str, str] = {}

    for ds_name in get_dataset_names(cfg):
        if ds_name in skip_set:
            logger.info("--- Dataset: %s — SKIPPED (pre-existing checkpoint provided) ---", ds_name)
            continue
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

        trainer = ClassifyVMoETrainer(
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
# Phase 2
# ---------------------------------------------------------------------------

def phase2_train_ensemble(cfg: dict, vmoe_checkpoints: dict[str, str]) -> str:
    """Train the MetaLearner (classification only). Returns best checkpoint path."""
    logger.info("=" * 60)
    logger.info("PHASE 2: Training stacking ensemble meta-learner (classification only)")
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

    dataset_names = get_dataset_names(cfg)

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

    ckpt_paths = [vmoe_checkpoints.get(n) for n in dataset_names]
    for name, ckpt in zip(dataset_names, ckpt_paths):
        if ckpt is None:
            logger.warning("No checkpoint for VMoE [%s]; will use random weights.", name)

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

    trainer = ClassifyEnsembleTrainer(
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
        description="VMoE + Stacking Ensemble — classification only (Phase 1 & 2)"
    )
    parser.add_argument(
        "--config", default="configs/config.yaml",
        help="Path to YAML config file.",
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
    parser.add_argument(
        "--skip-datasets", nargs="+", default=None,
        metavar="NAME",
        help="Dataset names to skip in Phase 1 (already trained). "
             "Supply their checkpoints via --vmoe-ckpts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = load_config(args.config)

    setup_logging(cfg["logging"]["dir"], name="train_cls_only")
    set_seed(cfg.get("seed", 42))

    vmoe_ckpts: dict[str, str] = {}
    if args.vmoe_ckpts:
        for item in args.vmoe_ckpts:
            name, path = item.split(":", 1)
            vmoe_ckpts[name] = path

    if args.phase in ("vmoe", "all"):
        new_ckpts = phase1_train_vmoe(cfg, skip_datasets=args.skip_datasets)
        vmoe_ckpts.update(new_ckpts)

    if args.phase in ("ensemble", "all"):
        phase2_train_ensemble(cfg, vmoe_ckpts)

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
