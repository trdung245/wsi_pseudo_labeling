"""
Supervised Contrastive Loss
===========================
Paper: Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020
       https://arxiv.org/abs/2004.11362

Why this helps for visually ambiguous classes
---------------------------------------------
Cross-entropy only trains the final linear layer to separate classes.  For
benign patches — which share texture/morphology with non-cancer *and* atypical
— CE loss does not explicitly push the *feature embeddings* apart.  The model
can achieve a reasonable CE loss while benign and non-cancer embeddings remain
interleaved in feature space, which causes misclassification at inference.

Supervised Contrastive Loss operates on the embedding space directly:

  L_SupCon = -1/|P(i)| · Σ_{p ∈ P(i)} log[
                exp(z_i · z_p / τ)
                ─────────────────────────────────────
                Σ_{a ∈ A(i)} exp(z_i · z_a / τ)
             ]

where:
  z_i  = ℓ₂-normalised projection of sample i   (unit sphere)
  P(i) = indices of other samples with the same class as i  (positives)
  A(i) = all other samples in the batch           (negatives + positives)
  τ    = temperature (default 0.07)

Effect on the embedding space
------------------------------
  - Benign patches are pulled toward *other benign patches*.
  - Non-cancer, atypical, and malignant patches are explicitly used as
    negatives for benign, regardless of visual similarity.
  - The hyperspherical geometry means the model cannot "hedge" by placing
    ambiguous benign patches between clusters — it must commit them to a region.

Usage
-----
  loss = SupConLoss(temperature=0.07)
  # features: (B, proj_dim), ℓ₂-normalised
  # labels:   (B,) long
  l = loss(features, labels)

Projection head
---------------
SupCon is applied to a *projected* representation, not the raw feature vector.
A 2-layer MLP (feature_dim → hidden → proj_dim, with BN and ReLU) is standard.
The projection head is only used during training and discarded at inference.
This protects the feature representations used by the classification head from
being over-constrained by the contrastive objective.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).

    Parameters
    ----------
    temperature:
        τ — scaling of the dot products before softmax.  Lower values
        create sharper peaks, stronger class separation.  0.07 is standard.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        features: torch.Tensor,   # (B, D)  ℓ₂-normalised projections
        labels: torch.Tensor,     # (B,)    long class indices
    ) -> torch.Tensor:
        """Compute the supervised contrastive loss.

        Samples whose class has no other representative in the batch contribute
        zero loss (no positives available) — this is handled gracefully.
        """
        B = features.size(0)
        if B < 2:
            return features.sum() * 0.0     # no-op, keeps graph alive

        # ── Similarity matrix ───────────────────────────────────────────
        # features are already ℓ₂-normalised; dot product = cosine similarity
        sim = torch.matmul(features, features.T) / self.temperature  # (B, B)

        # ── Masks ───────────────────────────────────────────────────────
        device = features.device
        labels = labels.view(-1, 1)                              # (B, 1)
        pos_mask = labels.eq(labels.T).float()                   # (B, B) – 1 where same class
        self_mask = torch.eye(B, device=device).bool()
        pos_mask.masked_fill_(self_mask, 0.0)                    # exclude self

        # ── Log-sum-exp denominator (all non-self pairs) ─────────────────
        # Subtract max per row for numerical stability before exp
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        exp_sim = torch.exp(sim)                                 # (B, B)
        exp_sim_no_self = exp_sim.masked_fill(self_mask, 0.0)   # (B, B)
        log_denominator = torch.log(exp_sim_no_self.sum(dim=1, keepdim=True) + 1e-8)  # (B, 1)

        # ── Per-sample SupCon loss ───────────────────────────────────────
        # For each anchor i: mean over its positives of log(p_ip / Σ_a p_ia)
        log_prob = sim - log_denominator                         # (B, B)

        # Only average over valid positive pairs
        n_pos = pos_mask.sum(dim=1)                              # (B,)
        valid = n_pos > 0                                        # samples that have at least one positive

        loss_per_sample = -(pos_mask * log_prob).sum(dim=1)     # (B,)
        loss_per_sample[valid] = loss_per_sample[valid] / n_pos[valid]

        return loss_per_sample[valid].mean() if valid.any() else features.sum() * 0.0
