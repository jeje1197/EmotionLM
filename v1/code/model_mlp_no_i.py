"""ALS MLP model (v4_no_i) for intensity-blind retrieval.

Maps [Semantic_Similarity, Emotional_Alignment, Time_Closeness]
into a single edge "link" probability.
"""

import torch
import torch.nn as nn

class ALSModelMLPNoI(nn.Module):
    def __init__(self, hidden_dim: int = 8):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
