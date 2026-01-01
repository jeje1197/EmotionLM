import torch
import torch.nn as nn
import numpy as np


class ALSModel(nn.Module):
    def __init__(self, ignore_feature=None):
        super().__init__()

        self.slp = nn.Linear(3, 1, bias=True)
        self.ignore_feature = ignore_feature
    
    def forward(self, x):
        if self.ignore_feature:
            x[self.ignore_feature] = torch.tensor(0, dtype=torch.float32)
        return nn.functional.sigmoid(self.slp(x))