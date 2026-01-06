#!/usr/bin/env python3
"""Evaluate a trained ALS model on a binary edge-classification CSV.

This script loads the ALS model weights and a validation CSV and reports
basic metrics (accuracy, precision, recall, F1) for the Linked label.
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from model import ALSModel


def load_dataset(csv_path: Path, feature_fields, label_field: str) -> TensorDataset:
    import pandas as pd

    df = pd.read_csv(csv_path)
    X = torch.tensor(df[feature_fields].values, dtype=torch.float32)
    y = torch.tensor(np.expand_dims(df[label_field].values, axis=-1), dtype=torch.float32)
    return TensorDataset(X, y)


def compute_metrics(y_true: torch.Tensor, y_pred: torch.Tensor, threshold: float) -> Dict[str, float]:
    y_hat = (y_pred >= threshold).float()

    tp = ((y_hat == 1) & (y_true == 1)).sum().item()
    tn = ((y_hat == 0) & (y_true == 0)).sum().item()
    fp = ((y_hat == 1) & (y_true == 0)).sum().item()
    fn = ((y_hat == 0) & (y_true == 1)).sum().item()

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def evaluate(model_path: Path, csv_path: Path, threshold: float) -> Dict[str, float]:
    feature_fields = ["Semantic_Similarity", "Emotional_Alignment", "Time_Closeness"]
    label_field = "Linked"

    dataset = load_dataset(csv_path, feature_fields, label_field)
    loader = DataLoader(dataset, batch_size=256)

    model = ALSModel(ignore_feature=None)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            all_preds.append(y_pred)
            all_labels.append(y_batch)

    y_pred_full = torch.cat(all_preds, dim=0)
    y_true_full = torch.cat(all_labels, dim=0)

    metrics = compute_metrics(y_true_full, y_pred_full, threshold)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ALS binary classifier on a CSV file.")
    parser.add_argument("--model", default="v1/artifacts/pretrained/als.pt", help="Path to trained ALS model .pt file.")
    parser.add_argument("--data", default="v1/data/edge_classification_dataset.csv", help="Validation CSV file path.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for Linked=1.")
    parser.add_argument("--out", default=None, help="Optional JSON file to save metrics.")
    args = parser.parse_args()

    model_path = Path(args.model)
    data_path = Path(args.data)

    metrics = evaluate(model_path, data_path, args.threshold)

    print("Evaluation metrics (threshold = {:.2f}):".format(args.threshold))
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics written to {out_path}")


if __name__ == "__main__":
    main()
