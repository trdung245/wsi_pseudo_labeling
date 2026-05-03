"""
Base training loop shared by all three training phases.

Provides:
  - Device selection
  - AMP (automatic mixed precision) context
  - Cosine LR schedule with linear warm-up
  - Gradient clipping
  - Checkpoint saving / loading
  - Early stopping
  - Per-epoch metric logging
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from utils.logging_utils import MetricLogger
from utils.metrics import aggregate_metrics
from data.dataset import IDX_TO_LABEL, NUM_CLASSES

logger = logging.getLogger(__name__)


def get_device(preference: str = "auto") -> torch.device:
    """Select the best available device."""
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


def cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int):
    """Linear warm-up followed by cosine annealing to 0."""
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / max(1, num_warmup_steps)
        progress = float(step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


class EarlyStopping:
    """Stop training when validation metric does not improve for `patience` epochs."""

    def __init__(self, patience: int = 7, mode: str = "max", delta: float = 1e-4) -> None:
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.best: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def step(self, metric: float) -> bool:
        """Return True if training should stop."""
        if self.best is None:
            self.best = metric
            return False
        improved = (
            metric > self.best + self.delta if self.mode == "max"
            else metric < self.best - self.delta
        )
        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class BaseTrainer:
    """Abstract base class for all three training phases.

    Subclasses must implement:
      - _train_step(batch) -> (loss_dict, logits, attn_pred, labels, attns)
      - _eval_step(batch)  -> (logits, attn_pred, labels, attns)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: dict,
        phase_name: str,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.phase_name = phase_name

        # Store the config key so unfreeze logic can look up the phase LR ratio.
        cfg["_phase_key"] = phase_name
        phase_cfg = cfg["training"][phase_name]
        self.epochs       = phase_cfg["epochs"]
        self.lr           = phase_cfg["learning_rate"]
        self.weight_decay = phase_cfg["weight_decay"]
        self.grad_clip    = phase_cfg.get("grad_clip", 1.0)
        self.use_amp      = phase_cfg.get("amp", False)
        self.patience     = phase_cfg.get("patience", 999)
        self.warmup_epochs = phase_cfg.get("warmup_epochs", 0)
        self.log_every    = cfg["logging"]["log_every_n_steps"]

        self.device = get_device(cfg.get("device", "auto"))

        # Derive class names from config (supports 3-class or 4-class)
        lbl_cfg = cfg.get("labels", {})
        if "label_to_idx" in lbl_cfg:
            idx_to_label = {v: k for k, v in lbl_cfg["label_to_idx"].items()}
        else:
            idx_to_label = IDX_TO_LABEL
        self.class_names: list[str] = [idx_to_label[i] for i in sorted(idx_to_label)]
        self.model.to(self.device)

        # Differential LR: if the model exposes param_groups(), use backbone_lr
        # (10× lower) for pretrained layers and head_lr for new layers.
        # Falls back to a single LR group for models without param_groups().
        backbone_lr_ratio = phase_cfg.get("backbone_lr_ratio", 0.1)
        backbone_lr = self.lr * backbone_lr_ratio
        if hasattr(model, "param_groups"):
            param_groups = model.param_groups(
                backbone_lr=backbone_lr, head_lr=self.lr
            )
            logger.info(
                "Differential LR: backbone=%.2e  heads=%.2e",
                backbone_lr, self.lr,
            )
        else:
            param_groups = filter(lambda p: p.requires_grad, model.parameters())

        self.optimizer = AdamW(
            param_groups,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Epoch at which to unfreeze the full backbone for end-to-end fine-tuning.
        # Set to 0 or omit to skip unfreeze entirely.
        self.unfreeze_epoch: int = phase_cfg.get("unfreeze_epoch", 0)

        steps_per_epoch = len(train_loader)
        total_steps = self.epochs * steps_per_epoch
        warmup_steps = self.warmup_epochs * steps_per_epoch

        self.scheduler = cosine_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps
        )
        self.scaler = GradScaler(enabled=self.use_amp and self.device.type == "cuda")

        ckpt_dir = Path(cfg["checkpoints"]["dir"]) / phase_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = ckpt_dir
        self.save_top_k = cfg["checkpoints"]["save_top_k"]

        log_dir = Path(cfg["logging"]["dir"]) / phase_name
        self.metric_logger = MetricLogger(log_dir)
        self.early_stopping = EarlyStopping(patience=self.patience)

        self._top_checkpoints: list[tuple[float, Path]] = []   # (score, path)
        self.global_step = 0

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def _train_step(self, batch: dict) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _eval_step(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> dict[str, float]:
        """Run the full training loop.

        Returns
        -------
        Best validation metrics dict.
        """
        best_metrics: dict[str, float] = {}

        for epoch in range(1, self.epochs + 1):
            # Gradual unfreeze: at the configured epoch, unfreeze the full
            # backbone and add its previously-frozen params to the optimizer.
            if self.unfreeze_epoch and epoch == self.unfreeze_epoch:
                if hasattr(self.model, "unfreeze_all"):
                    self.model.unfreeze_all()
                    # Re-add newly unfrozen backbone params at the lower LR
                    backbone_lr = self.lr * self.cfg["training"][self.cfg["_phase_key"]].get(
                        "backbone_lr_ratio", 0.1
                    )
                    newly_unfrozen = [
                        p for p in self.model.parameters()
                        if p.requires_grad and not any(
                            p is q for group in self.optimizer.param_groups
                            for q in group["params"]
                        )
                    ]
                    if newly_unfrozen:
                        self.optimizer.add_param_group(
                            {"params": newly_unfrozen, "lr": backbone_lr}
                        )
                    logger.info("Epoch %d: unfroze all backbone layers (lr=%.2e).", epoch, backbone_lr)

            train_metrics = self._run_epoch(epoch, "train")
            val_metrics   = self._run_epoch(epoch, "val")

            lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                "Epoch %d/%d | LR=%.2e | train_loss=%.4f | val_loss=%.4f | "
                "val_acc=%.4f | val_macro_f1=%.4f | σ_cls=%.3f | σ_reg=%.4f",
                epoch, self.epochs, lr,
                train_metrics.get("loss", float("nan")),
                val_metrics.get("loss", float("nan")),
                val_metrics.get("accuracy", float("nan")),
                val_metrics.get("macro_f1", float("nan")),
                val_metrics.get("sigma_cls", float("nan")),
                val_metrics.get("sigma_reg", float("nan")),
            )
            # Log per-class F1 every epoch so minority-class collapse is visible immediately.
            per_class = " | ".join(
                f"{c}: {val_metrics.get(f'f1_{c}', float('nan')):.3f}"
                for c in self.class_names
            )
            logger.info("  Per-class val F1 → %s", per_class)

            self.metric_logger.log_scalars(train_metrics, self.global_step, prefix=f"{self.phase_name}/train")
            self.metric_logger.log_scalars(val_metrics, self.global_step, prefix=f"{self.phase_name}/val")

            score = val_metrics.get("macro_f1", 0.0)
            self._maybe_save_checkpoint(epoch, score)

            if self.early_stopping.step(score):
                logger.info("Early stopping triggered after epoch %d.", epoch)
                break

            best_metrics = val_metrics

        self.metric_logger.close()
        return best_metrics

    # ------------------------------------------------------------------
    # Epoch runner
    # ------------------------------------------------------------------

    def _run_epoch(self, epoch: int, split: str) -> dict[str, float]:
        is_train = split == "train"
        self.model.train(is_train)
        # Switch any auxiliary modules (e.g. projection head for SupCon) that
        # the subclass may have attached as attributes.
        if hasattr(self, "proj_head"):
            self.proj_head.train(is_train)
        loader = self.train_loader if is_train else self.val_loader

        all_labels: list[int] = []
        all_logits: list[np.ndarray] = []
        all_attn_pred: list[float] = []
        all_attn_true: list[float] = []
        epoch_loss = 0.0

        for step, batch in enumerate(loader):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            if is_train:
                self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    loss_dict, logits, attn_pred, labels, attns = self._train_step(batch)

                loss = loss_dict["total"]
                self.scaler.scale(loss).backward()

                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    params_to_clip = list(self.model.parameters())
                    if hasattr(self, "proj_head"):
                        params_to_clip += list(self.proj_head.parameters())
                    nn.utils.clip_grad_norm_(params_to_clip, self.grad_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.global_step += 1

                if step % self.log_every == 0:
                    self.metric_logger.log_scalars(
                        {k: v.item() if isinstance(v, torch.Tensor) else v
                         for k, v in loss_dict.items()},
                        self.global_step, prefix=f"{self.phase_name}/train_step"
                    )
            else:
                with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    logits, attn_pred, labels, attns = self._eval_step(batch)
                    # val_criterion takes precedence (e.g. DistillationTrainer uses a
                    # simpler criterion at eval time since the teacher is not available).
                    _val_crit = getattr(self, "val_criterion", None) or getattr(self, "criterion", None)
                    if _val_crit is not None:
                        loss_dict = _val_crit(
                            logits=logits,
                            attn_score=attn_pred,
                            labels=labels,
                            attentions=attns,
                        )
                    else:
                        loss_dict = {}

            epoch_loss += loss_dict.get("total", torch.tensor(0.0)).item() if loss_dict else 0.0
            all_labels.extend(labels.cpu().numpy().tolist())
            all_logits.extend(logits.detach().cpu().numpy().tolist())
            all_attn_pred.extend(attn_pred.detach().cpu().squeeze(-1).numpy().tolist())
            all_attn_true.extend(attns.cpu().numpy().tolist())

        metrics = aggregate_metrics(
            all_labels, all_logits, all_attn_pred, all_attn_true,
            class_names=self.class_names,
        )
        metrics["loss"] = epoch_loss / max(len(loader), 1)

        # Include learnable σ values if the model exposes them
        if hasattr(self, "criterion"):
            crit = self.criterion
            if hasattr(crit, "log_var_cls"):
                import math
                metrics["sigma_cls"] = math.exp(crit.log_var_cls.item() / 2)
                metrics["sigma_reg"] = math.exp(crit.log_var_reg.item() / 2)
            elif hasattr(crit, "hard_loss"):
                import math
                metrics["sigma_cls"] = math.exp(crit.hard_loss.log_var_cls.item() / 2)
                metrics["sigma_reg"] = math.exp(crit.hard_loss.log_var_reg.item() / 2)

        return metrics

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    def _maybe_save_checkpoint(self, epoch: int, score: float) -> None:
        path = self.ckpt_dir / f"epoch_{epoch:03d}_f1_{score:.4f}.pth"
        torch.save({
            "epoch":       epoch,
            "model":       self.model.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "scheduler":   self.scheduler.state_dict(),
            "score":       score,
        }, path)

        self._top_checkpoints.append((score, path))
        self._top_checkpoints.sort(key=lambda x: x[0], reverse=True)

        # Remove checkpoints beyond save_top_k
        while len(self._top_checkpoints) > self.save_top_k:
            _, old_path = self._top_checkpoints.pop()
            if old_path.exists():
                old_path.unlink()

    def load_best_checkpoint(self) -> None:
        if not self._top_checkpoints:
            logger.warning("No checkpoints saved; skipping load.")
            return
        _, best_path = self._top_checkpoints[0]
        state = torch.load(best_path, map_location=self.device)
        self.model.load_state_dict(state["model"])
        logger.info("Loaded best checkpoint: %s (score=%.4f)", best_path, state["score"])
