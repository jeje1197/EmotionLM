import torch
from training.model import ALSModel


def evaluate_and_report(model_path):
    model = ALSModel() 
    
    # Load data
    try:
        checkpoint = torch.load(model_path, weights_only=False)
        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        else:
            model = checkpoint
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()

    # Extract weights from the 'slp' layer
    weights = model.slp.weight[0].detach().tolist()
    bias = model.slp.bias.item()

    print(f"Weight SS (Semantic): {weights[0]:.4f}")
    print(f"Weight SE (Emotion):  {weights[1]:.4f}")
    print(f"Weight ST (Time):     {weights[2]:.4f}")
    print(f"Bias:                 {bias:.4f}")


if __name__ == "__main__":
    evaluate_and_report('data/models/als.pt')