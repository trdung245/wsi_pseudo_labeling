"""
Multi-task losses for WSI patch classification + attention regression.

Two loss classes are provided:

UncertaintyMultiTaskLoss  (Kendall et al., CVPR 2018)
-----------------------------------------------------
Learns per-task noise (σ₁, σ₂) to weight tasks automatically.
Risk: the optimizer can increase σ₁ to down-weight classification,
causing collapse toward the majority class (non-cancer).

CBFocalMultiTaskLoss  (recommended — default in vmoe_trainer.py)
----------------------------------------------------------------
Class-Balanced Focal Loss (Cui et al., CVPR 2019) for classification
+ fixed-weight Huber regression. No learnable per-task weights,
so there is no mechanism to silently down-weight the classification task.

Class-Balanced weighting
~~~~~~~~~~~~~~~~~~~~~~~~
  Effective number of samples for class i:
    E_i = (1 - β^{n_i}) / (1 - β)
  where n_i = raw sample count, β ∈ [0, 1) (default 0.9999).

  Class weight:
    w_i = 1 / E_i    (then normalised so Σw = num_classes)

  Compared to inverse-frequency (w_i = 1/n_i), CB weighting applies
  a diminishing-returns correction: adding the 1001st sample of a
  class provides less new information than the 2nd.  This gives the
  majority class a non-zero but appropriately reduced weight and avoids
  over-penalising the minority class.

Fixed regression weight
~~~~~~~~~~~~~~~~~~~~~~~
  The regression loss is scaled by `reg_weight` (default 0.1).
  This keeps the attention-score task as a useful auxiliary signal
  without letting it flood the classification gradients — the problem
  that motivated the learnable uncertainty in the first place.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CBFocalMultiTaskLoss(nn.Module):
    """Class-Balanced Focal Loss + fixed-weight Huber regression.

    Parameters
    ----------
    samples_per_class:
        List/array of raw training-set counts per class, in label-index order
        (e.g. [n_non_cancer, n_benign, n_atypical, n_malignant]).
        Used to compute the effective-number class weights.
    num_classes:
        Number of label classes.
    beta:
        CB smoothing hyperparameter (Cui et al.).  Default 0.9999 is the
        value recommended in the paper for most datasets.
    focal_gamma:
        Focusing parameter γ.  2.0 is the standard recommendation; it
        reduces the gradient contribution of easy well-classified examples
        by a factor of (1-p)^γ, forcing the model to focus on hard
        minority-class distinctions.
    reg_weight:
        Fixed scalar multiplier for the attention-regression Huber loss.
        0.1 keeps the regression task as a helpful auxiliary signal without
        dominating the classification gradients.
    load_balance_coeff:
        Multiplier for the VMoE auxiliary load-balancing loss.
    """

    def __init__(
        self,
        samples_per_class: list[int] | np.ndarray,
        num_classes: int = 4,
        beta: float = 0.9999,
        focal_gamma: float = 2.0,
        reg_weight: float = 0.1,
        load_balance_coeff: float = 0.01,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.focal_gamma = focal_gamma
        self.reg_weight = reg_weight
        self.load_balance_coeff = load_balance_coeff

        # Class-Balanced weights (Cui et al., CVPR 2019)
        samples = np.array(samples_per_class, dtype=np.float64)
        effective_num = 1.0 - np.power(beta, samples)
        weights = (1.0 - beta) / effective_num          # unnormalised CB weight
        weights = weights / weights.sum() * num_classes  # normalise: mean weight = 1
        self.register_buffer("class_weights", torch.tensor(weights, dtype=torch.float32))

    # ------------------------------------------------------------------

    def _cb_focal_loss(
        self,
        logits: torch.Tensor,   # (B, C)
        targets: torch.Tensor,  # (B,) long
    ) -> torch.Tensor:
        """CB-weighted focal loss."""
        ce = F.cross_entropy(
            logits, targets,
            weight=self.class_weights,
            reduction="none",
        )  # (B,)
        probs = torch.exp(-ce)                               # p_t (after CB weighting)
        focal = (1.0 - probs) ** self.focal_gamma * ce
        return focal.mean()

    @staticmethod
    def _regression_loss(
        pred: torch.Tensor,    # (B, 1) or (B,)
        target: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        pred = pred.squeeze(-1)
        return F.huber_loss(pred, target, delta=0.1)

    # ------------------------------------------------------------------

    def forward(
        self,
        logits: torch.Tensor,        # (B, C)
        attn_score: torch.Tensor,    # (B, 1)
        labels: torch.Tensor,        # (B,) long
        attentions: torch.Tensor,    # (B,) float
        load_bal_loss: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        l_cls = self._cb_focal_loss(logits, labels)
        l_reg = self._regression_loss(attn_score, attentions)

        total = l_cls + self.reg_weight * l_reg

        lb = torch.tensor(0.0, device=logits.device)
        if load_bal_loss is not None:
            lb = self.load_balance_coeff * load_bal_loss
            total = total + lb

        return {
            "total":         total,
            "cls_loss":      l_cls.detach(),
            "reg_loss":      l_reg.detach(),
            "weighted_cls":  l_cls.detach(),
            "weighted_reg":  (self.reg_weight * l_reg).detach(),
            "sigma_cls":     torch.tensor(float("nan")),   # not applicable
            "sigma_reg":     torch.tensor(float("nan")),   # not applicable
            "load_bal_loss": lb.detach(),
        }

    def extra_repr(self) -> str:
        return (
            f"focal_gamma={self.focal_gamma}, "
            f"reg_weight={self.reg_weight}, "
            f"load_balance_coeff={self.load_balance_coeff}"
        )


class UncertaintyMultiTaskLoss(nn.Module):
    """Kendall et al. (2018) uncertainty-weighted multi-task loss.

    Learnable parameters
    --------------------
    log_var_cls:  s₁ = log(σ₁²) for the classification task.
    log_var_reg:  s₂ = log(σ₂²) for the attention regression task.

    Both are initialised to 0 (σ = 1, equal weighting) and updated by the
    optimiser during training.

    Parameters
    ----------
    num_classes:
        Number of label classes.
    init_log_var_cls / init_log_var_reg:
        Initial values of the learnable log-variance parameters.
    use_focal:
        If True, replace cross-entropy with focal loss for the classification
        task (helpful for the class imbalance in pos_heavy / neg_heavy sets).
    focal_gamma:
        Focusing parameter γ for focal loss (ignored when use_focal=False).
    class_weights:
        Optional (num_classes,) tensor of per-class weights for CE/focal.
    load_balance_coeff:
        Multiplier for the VMoE auxiliary load-balancing loss.
    """

    def __init__(
        self,
        num_classes: int = 4,
        init_log_var_cls: float = 0.0,
        init_log_var_reg: float = 0.0,
        use_focal: bool = False,
        focal_gamma: float = 2.0,
        class_weights: torch.Tensor | None = None,
        load_balance_coeff: float = 0.01,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma
        self.load_balance_coeff = load_balance_coeff

        # Learnable log-variance parameters (one per task).
        self.log_var_cls = nn.Parameter(torch.tensor(init_log_var_cls))
        self.log_var_reg = nn.Parameter(torch.tensor(init_log_var_reg))

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    # ------------------------------------------------------------------
    # Task-specific losses
    # ------------------------------------------------------------------

    def _classification_loss(
        self,
        logits: torch.Tensor,   # (B, C)
        targets: torch.Tensor,  # (B,) long
    ) -> torch.Tensor:
        if self.use_focal:
            return self._focal_loss(logits, targets)
        return F.cross_entropy(logits, targets, weight=self.class_weights)

    def _focal_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Focal loss: FL(p_t) = -α_t (1 - p_t)^γ log(p_t)."""
        ce = F.cross_entropy(
            logits, targets,
            weight=self.class_weights,
            reduction="none",
        )  # (B,)
        probs = torch.exp(-ce)                         # p_t
        focal = (1 - probs) ** self.focal_gamma * ce
        return focal.mean()

    @staticmethod
    def _regression_loss(
        pred: torch.Tensor,    # (B, 1) or (B,)
        target: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        """Huber (smooth-L1) loss for attention score regression.

        Huber is preferred over MSE here because attention scores have a
        heavy-tailed distribution (a few patches carry most of the weight)
        and MSE would be dominated by outliers.
        """
        pred = pred.squeeze(-1)                        # (B,)
        return F.huber_loss(pred, target, delta=0.1)

    # ------------------------------------------------------------------
    # Combined uncertainty-weighted loss
    # ------------------------------------------------------------------

    def forward(
        self,
        logits: torch.Tensor,        # (B, C)
        attn_score: torch.Tensor,    # (B, 1)
        labels: torch.Tensor,        # (B,)  long
        attentions: torch.Tensor,    # (B,)  float
        load_bal_loss: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the full multi-task loss.

        Returns
        -------
        dict with keys:
          total          – back-propagated scalar
          cls_loss       – unweighted classification loss
          reg_loss       – unweighted regression loss
          weighted_cls   – uncertainty-weighted classification term
          weighted_reg   – uncertainty-weighted regression term
          sigma_cls      – current σ₁ (for logging)
          sigma_reg      – current σ₂ (for logging)
          load_bal_loss  – auxiliary VMoE loss (0 if not provided)
        """
        l_cls = self._classification_loss(logits, labels)
        l_reg = self._regression_loss(attn_score, attentions)

        # Uncertainty weighting  (Eq. 3 & 7 of Kendall et al.)
        # s = log(σ²), σ² = exp(s), 1/σ² = exp(-s), log(σ) = s/2
        s1 = self.log_var_cls   # classification log-variance
        s2 = self.log_var_reg   # regression log-variance

        weighted_cls = l_cls * torch.exp(-s1) + s1 / 2
        weighted_reg = l_reg * torch.exp(-s2) / 2 + s2 / 2

        total = weighted_cls + weighted_reg

        lb = torch.tensor(0.0, device=logits.device)
        if load_bal_loss is not None:
            lb = self.load_balance_coeff * load_bal_loss
            total = total + lb

        return {
            "total":          total,
            "cls_loss":       l_cls.detach(),
            "reg_loss":       l_reg.detach(),
            "weighted_cls":   weighted_cls.detach(),
            "weighted_reg":   weighted_reg.detach(),
            "sigma_cls":      torch.exp(s1 / 2).detach(),
            "sigma_reg":      torch.exp(s2 / 2).detach(),
            "load_bal_loss":  lb.detach(),
        }

    def extra_repr(self) -> str:
        return (
            f"use_focal={self.use_focal}, "
            f"focal_gamma={self.focal_gamma}, "
            f"load_balance_coeff={self.load_balance_coeff}"
        )


# =============================================================================
# Notes on alternative multi-task loss strategies
# =============================================================================
#
# 1. GradNorm (Chen et al., 2018)
#    - Dynamically adjusts loss weights based on the ratio of gradient norms.
#    - Target: all tasks should have similar gradient magnitudes relative to
#      their initial training loss.
#    - Advantage over Kendall: directly controls gradient flow; more robust
#      when task losses have very different scales.
#    - Drawback: requires an extra backward pass to compute per-task gradients.
#
# 2. PCGrad (Yu et al., NeurIPS 2020)
#    - Projects conflicting task gradients onto each other's normal plane.
#    - Helpful when classification and regression gradients point in opposite
#      directions (task conflict).
#    - Particularly relevant here: the attention score regression may conflict
#      with the classification objective for high-attention non-cancer patches.
#
# 3. DWA – Dynamic Weight Averaging (Liu et al., CVPR 2019)
#    - Adjusts weights based on the rate of change of each task's loss.
#    - Simple, no extra backward pass; just requires tracking loss history.
#    - Recommended as an easy drop-in alternative.
#
# 4. Recommended for this specific problem
#    - The uncertainty loss (Kendall) is a good default.
#    - If the attention regression loss dominates early training (it can,
#      because a few patches have very high attention scores), switch to
#      focal loss + a lower initial log_var_reg (e.g. -1.0) to down-weight
#      the regression task initially.
#    - If class imbalance is severe (neg_heavy CSV), enable use_focal=True
#      with class_weights computed from the training set's inverse frequency.
# =============================================================================
