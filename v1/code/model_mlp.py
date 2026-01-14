"""ALS MLP model (v4) for non-linear feature interaction.

Maps [Semantic_Similarity, Emotional_Alignment, Time_Closeness, Target_Intensity]
through a hidden layer to capture complex retrieval logic.
"""

import torch
import torch.nn as nn

class ALSModelMLP(nn.Module):
    def __init__(self, hidden_dim: int = 8):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
