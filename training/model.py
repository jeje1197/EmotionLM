import torch.nn as nn


class ALSModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.slp = nn.Linear(3, 1, bias=True)
    
    def forward(self, x):
        return nn.functional.sigmoid(self.slp(x))