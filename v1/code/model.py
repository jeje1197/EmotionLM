"""ALS edge-classification model for v1.

This is a minimal logistic model that maps three features
[Semantic_Similarity, Emotional_Alignment, Time_Closeness]
into a single edge "link" probability via a linear layer
followed by a sigmoid.

It mirrors the original ALSModel used in training/model.py so that
v1 can be used as a standalone bundle.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ALSModel(nn.Module):
    def __init__(self, ignore_feature: int | None = None) -> None:
        super().__init__()
        # Order of inputs: [Semantic_Similarity, Emotional_Alignment, Time_Closeness]
        self.slp = nn.Linear(3, 1, bias=True)
        # If set to an index (0, 1, or 2), that feature is zeroed in forward.
        self.ignore_feature = ignore_feature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ignore_feature is not None:
            # Zero out the selected feature dimension across the batch.
            x = x.clone()
            x[:, self.ignore_feature] = 0.0
        return torch.sigmoid(self.slp(x))
