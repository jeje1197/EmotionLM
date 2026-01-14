"""ALS edge-classification model with Emotional Intensity (v2).

This model maps four features:
[Semantic_Similarity, Emotional_Alignment, Time_Closeness, Target_Intensity]
into a single edge "link" probability.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ALSModelV2(nn.Module):
    def __init__(self, ignore_feature: int | None = None) -> None:
        super().__init__()
        # Order: [Semantic, Emotional, Temporal, Intensity]
        self.slp = nn.Linear(4, 1, bias=True)
        self.ignore_feature = ignore_feature
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ignore_feature is not None:
            x = x.clone()
            x[:, self.ignore_feature] = 0.0
        return torch.sigmoid(self.slp(x))
