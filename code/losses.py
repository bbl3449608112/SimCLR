from __future__ import annotations

import torch
import torch.nn.functional as F


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """Compute the NT-Xent loss for two batches of SimCLR projections.

    z1 and z2 have shape [N, D]. The concatenated batch has 2N examples; each
    example's positive pair is the other augmented view of the same original
    image, and all other non-self examples are negatives.
    """
    if z1.shape != z2.shape:
        raise ValueError(f"z1 and z2 must have the same shape, got {z1.shape} and {z2.shape}")
    if z1.ndim != 2:
        raise ValueError(f"z1 and z2 must be rank-2 tensors, got {z1.ndim}")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    batch_size = z1.shape[0]
    projections = torch.cat([z1, z2], dim=0)
    projections = F.normalize(projections, dim=1)

    logits = projections @ projections.T
    logits = logits / temperature

    self_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(self_mask, float("-inf"))

    targets = torch.arange(2 * batch_size, device=logits.device)
    targets = (targets + batch_size) % (2 * batch_size)
    return F.cross_entropy(logits, targets)
