"""
Phase 3: Knowledge Distillation Trainer

The stacking ensemble (teacher) is frozen.
The student PatchClassifier is trained using the combined distillation loss:
  L = α · KD_soft + β · attn_KD + (1-α) · hard_uncertainty_loss
"""

from __future__ import annotations

import torch

from losses import DistillationLoss
from models import PatchClassifier, VMoE, StackingEnsemble
from .base_trainer import BaseTrainer


class DistillationTrainer(BaseTrainer):
    """Trainer for Phase 3: student PatchClassifier learns from the ensemble teacher."""

    def __init__(
        self,
        student: PatchClassifier,
        teacher: StackingEnsemble,
        train_loader,
        val_loader,
        cfg: dict,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__(student, train_loader, val_loader, cfg, phase_name="distillation")

        self.teacher = teacher.to(self.device)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

        dist_cfg = cfg["training"]["distillation"]
        loss_cfg  = cfg.get("loss", {})

        self.criterion = DistillationLoss(
            temperature=dist_cfg.get("temperature", 4.0),
            alpha=dist_cfg.get("alpha", 0.6),
            num_classes=cfg["labels"]["num_classes"],
            init_log_var_cls=loss_cfg.get("init_log_var_cls", 0.0),
            init_log_var_reg=loss_cfg.get("init_log_var_reg", 0.0),
            use_focal=True,
            focal_gamma=2.0,
            class_weights=class_weights,
        ).to(self.device)

        self.optimizer.add_param_group({
            "params": list(self.criterion.parameters()),
            "lr": self.lr,
            "weight_decay": 0.0,
        })
        # The scheduler was created in BaseTrainer.__init__ before this group
        # was added.  Extend its internal lambda/base_lr lists so that
        # scheduler.step() produces one lr value per param group.
        self.scheduler.lr_lambdas.append(self.scheduler.lr_lambdas[-1])
        self.scheduler.base_lrs.append(self.lr)

        # During eval the teacher is not run, so use the hard-label component
        # (UncertaintyMultiTaskLoss) which has the base_trainer-compatible signature.
        self.val_criterion = self.criterion.hard_loss

    # ------------------------------------------------------------------

    def _train_step(self, batch):
        images     = batch["image"]
        labels     = batch["label"]
        attentions = batch["attention"]

        with torch.no_grad():
            teacher_out = self.teacher(images)

        student_out = self.model(images)

        loss_dict = self.criterion(
            student_logits=student_out["logits"],
            student_attn=student_out["attn_score"],
            teacher_logits=teacher_out["logits"],
            labels=labels,
            attentions=attentions,
        )
        return loss_dict, student_out["logits"], student_out["attn_score"], labels, attentions

    def _eval_step(self, batch):
        images     = batch["image"]
        labels     = batch["label"]
        attentions = batch["attention"]

        out = self.model(images)
        return out["logits"], out["attn_score"], labels, attentions
