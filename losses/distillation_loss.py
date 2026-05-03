"""
Knowledge Distillation Loss
============================
Combines:
  1. Soft KD loss    – KL divergence between student and teacher softened distributions
  2. Hard label loss – cross-entropy with ground-truth labels
  3. Uncertainty weighting applied to the combined multi-task signal

The total loss is:

  L = α · L_KD + (1-α) · L_hard
    subject to per-task uncertainty weighting

where:
  L_KD   = KL(σ(z_t/T) || σ(z_s/T))  · T²    (Hinton et al., 2015)
  L_hard = CrossEntropy(z_s, y)
  T      = temperature (controls softness of teacher distribution)
  α      = KD weight  (higher → more distillation)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .multitask_loss import UncertaintyMultiTaskLoss


class DistillationLoss(nn.Module):
    """Knowledge distillation loss with uncertainty-weighted multi-task objectives.

    Parameters
    ----------
    temperature:
        Softening temperature T for logit-based KD. Higher T → softer
        distributions, transferring more "dark knowledge".  Typical: 3–8.
    alpha:
        Weight for the soft KD loss (0=no KD, 1=only KD).
    num_classes:
        Number of output classes.
    init_log_var_cls / init_log_var_reg:
        Initial uncertainty parameters (passed to UncertaintyMultiTaskLoss).
    use_focal / focal_gamma / class_weights:
        Forwarded to UncertaintyMultiTaskLoss for the hard-label term.
    load_balance_coeff:
        VMoE auxiliary load-balancing loss coefficient.
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.6,
        num_classes: int = 4,
        init_log_var_cls: float = 0.0,
        init_log_var_reg: float = 0.0,
        use_focal: bool = False,
        focal_gamma: float = 2.0,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        assert 0.0 <= alpha <= 1.0, "alpha must be in [0, 1]"

        self.T = temperature
        self.alpha = alpha

        # Hard-label multi-task loss (shared uncertainty parameters)
        self.hard_loss = UncertaintyMultiTaskLoss(
            num_classes=num_classes,
            init_log_var_cls=init_log_var_cls,
            init_log_var_reg=init_log_var_reg,
            use_focal=use_focal,
            focal_gamma=focal_gamma,
            class_weights=class_weights,
            load_balance_coeff=0.0,
        )

    def forward(
        self,
        student_logits: torch.Tensor,      # (B, C)
        student_attn: torch.Tensor,         # (B, 1)
        teacher_logits: torch.Tensor,       # (B, C)   – from frozen teacher
        labels: torch.Tensor,               # (B,)  long
        attentions: torch.Tensor,           # (B,)  float  (ground-truth)
    ) -> dict[str, torch.Tensor]:
        """
        Returns
        -------
        dict with keys:
          total, kd_loss, hard_cls, hard_reg, sigma_cls, sigma_reg
        """
        T = self.T

        # 1. Soft KD loss: KL(teacher || student) with temperature scaling
        #    Multiply by T² to preserve gradient magnitudes when T > 1
        #    (Hinton et al., 2015 – "Distilling the Knowledge in a Neural Network")
        soft_teacher = F.log_softmax(teacher_logits.detach() / T, dim=-1)
        soft_student = F.log_softmax(student_logits / T, dim=-1)
        kd_loss = F.kl_div(
            soft_student, soft_teacher.exp(),
            reduction="batchmean",
        ) * (T ** 2)

        # 2. Hard-label multi-task loss (uncertainty weighted)
        hard = self.hard_loss(
            logits=student_logits,
            attn_score=student_attn,
            labels=labels,
            attentions=attentions,
        )

        # 3. Combine
        total = self.alpha * kd_loss + (1 - self.alpha) * hard["total"]

        return {
            "total":         total,
            "kd_loss":       kd_loss.detach(),
            "hard_cls":      hard["cls_loss"],
            "hard_reg":      hard["reg_loss"],
            "sigma_cls":     hard["sigma_cls"],
            "sigma_reg":     hard["sigma_reg"],
            "load_bal_loss": hard["load_bal_loss"],
        }
