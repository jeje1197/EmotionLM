from training.model import ALSModel

import torch

if __name__ == "__main__":
    model = torch.load('data/models/als_099.pt', weights_only=False)

    print(model.slp.weight[0].tolist())