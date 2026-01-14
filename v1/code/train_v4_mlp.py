#!/usr/bin/env python3
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model_mlp import ALSModelMLP

# Configuration
CONFIG = {
    "num_epochs": 200,
    "learning_rate": 0.005,
    "batch_size": 32,
    "val_split": 0.2,
    "random_seed": 42,
    "early_stopping_patience": 25,
    "hidden_dim": 8
}

def load_data(csv_path: Path):
    df = pd.read_csv(csv_path)
    features = ["Semantic_Similarity", "Emotional_Alignment", "Time_Closeness", "Target_Intensity"]
    X = torch.tensor(df[features].values, dtype=torch.float32)
    y = torch.tensor(df["Linked"].values, dtype=torch.float32).unsqueeze(1)
    return TensorDataset(X, y)

def train_model(name: str, dataset: TensorDataset):
    print(f"\n--- Training {name.upper()} MLP Model ---")
    
    val_size = int(len(dataset) * CONFIG["val_split"])
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(CONFIG["random_seed"])
    )
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"])
    
    model = ALSModelMLP(hidden_dim=CONFIG["hidden_dim"])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.BCELoss()
    
    best_val_loss = float('inf')
    best_weights = None
    patience_counter = 0
    history = []
    
    for epoch in range(CONFIG["num_epochs"]):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        
        train_loss /= len(train_ds)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                all_preds.extend((y_pred > 0.5).float().cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        
        val_loss /= len(val_ds)
        acc = accuracy_score(all_labels, all_preds)
        
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": acc
        })
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:03d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Acc: {acc:.4f}")
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
    model.load_state_dict(best_weights)
    model.eval()
    
    final_preds = []
    final_labels = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            y_pred = model(X_batch)
            final_preds.extend((y_pred > 0.5).float().cpu().numpy())
            final_labels.extend(y_batch.cpu().numpy())
            
    metrics = {
        "accuracy": accuracy_score(final_labels, final_preds),
        "precision": precision_score(final_labels, final_preds, zero_division=0),
        "recall": recall_score(final_labels, final_preds, zero_division=0),
        "f1": f1_score(final_labels, final_preds, zero_division=0)
    }
    
    print(f"Final Accuracy for {name}: {metrics['accuracy']:.4f}")
    return model, history, metrics

def main():
    root = Path("v1/data")
    causal_csv = root / "causal_training_dataset.csv"
    affective_csv = root / "affective_training_dataset.csv"
    
    results = {}
    
    # Task 1: Causal
    causal_ds = load_data(causal_csv)
    c_model, c_hist, c_metrics = train_model("causal_v4_mlp", causal_ds)
    results["causal_mlp"] = {"history": c_hist, "metrics": c_metrics}
    torch.save(c_model.state_dict(), "v1/artifacts/pretrained/als_causal_v4_mlp.pt")
    
    # Task 2: Affective
    affective_ds = load_data(affective_csv)
    a_model, a_hist, a_metrics = train_model("affective_v4_mlp", affective_ds)
    results["affective_mlp"] = {"history": a_hist, "metrics": a_metrics}
    torch.save(a_model.state_dict(), "v1/artifacts/pretrained/als_affective_v4_mlp.pt")
    
    # Task 3: Both
    combined_X = torch.cat([causal_ds.tensors[0], affective_ds.tensors[0]], dim=0)
    combined_y = torch.cat([causal_ds.tensors[1], affective_ds.tensors[1]], dim=0)
    both_ds = TensorDataset(combined_X, combined_y)
    
    b_model, b_hist, b_metrics = train_model("combined_v4_mlp", both_ds)
    results["combined_mlp"] = {"history": b_hist, "metrics": b_metrics}
    torch.save(b_model.state_dict(), "v1/artifacts/pretrained/als_combined_v4_mlp.pt")
    
    with open("v1/artifacts/pretrained/v4_mlp_metrics.json", "w") as f:
        json.dump({"config": CONFIG, "results": results}, f, indent=4)
    print("\nAll V4 MLP models trained and results saved.")

if __name__ == "__main__":
    main()
