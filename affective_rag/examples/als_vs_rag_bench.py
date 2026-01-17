import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict

# Assuming the model and weights exist from the previous training run
MODEL_PATH = "v1/artifacts/pretrained/als_unified_linear.pt"
DATA_PATH = "v1/data/unified_memory_dataset.json"

class ALSModel(torch.nn.Module):
    def __init__(self, input_dim: int = 4):
        super().__init__()
        self.slp = torch.nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.slp(x))

def cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def calculate_temporal(d1, d2):
    delta = abs(d1 - d2)
    return 1.0 / (1.0 + np.log1p(delta))

def run_benchmark():
    # 1. Load Model
    model = ALSModel(input_dim=4)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    # 2. Get learned weights for reporting
    weights = model.slp.weight.detach().numpy()[0]
    bias = model.slp.bias.detach().numpy()[0]
    print(f"--- ALS Formula Coefficients ---")
    print(f"S: {weights[0]:.2f}, E: {weights[1]:.2f}, T: {weights[2]:.2f}, I: {weights[3]:.2f}, B: {bias:.2f}\n")

    # 3. Load Real Dataset
    with open(DATA_PATH, "r") as f:
        quads = json.load(f)

    results = []

    for q in quads:
        anchor = q["anchor"]
        candidates = [
            (q["positive"], "Positive (Goal)"),
            (q["temporal_distractor"], "Temporal Distractor"),
            (q["semantic_distractor"], "Semantic Distractor")
        ]

        # Score all candidates with both Methods
        scored_candidates = []
        for event, label in candidates:
            # Semantic RAG Score (Standard Cosine Similarity)
            rag_score = (cosine_similarity(anchor["semantic_vec"], event["semantic_vec"]) + 1) / 2
            
            # ALS Feature Vector
            s = (cosine_similarity(anchor["semantic_vec"], event["semantic_vec"]) + 1) / 2
            e = (cosine_similarity(anchor["emotional_vec"], event["emotional_vec"]) + 1) / 2
            t = calculate_temporal(anchor["d_days"], event["d_days"])
            i = event["emotional_intensity"]
            
            feat = torch.tensor([[s, e, t, i]], dtype=torch.float32)
            with torch.no_grad():
                als_score = model(feat).item()
            
            scored_candidates.append({
                "label": label,
                "rag_score": rag_score,
                "als_score": als_score,
                "category": q["category"],
                "intensity": i,
                "time_gap": abs(anchor["d_days"] - event["d_days"])
            })

        # Rank them
        by_rag = sorted(scored_candidates, key=lambda x: x["rag_score"], reverse=True)
        by_als = sorted(scored_candidates, key=lambda x: x["als_score"], reverse=True)

        results.append({
            "category": q["category"],
            "rag_winner": by_rag[0]["label"] == "Positive (Goal)",
            "als_winner": by_als[0]["label"] == "Positive (Goal)",
            "rag_top_label": by_rag[0]["label"],
            "als_top_label": by_als[0]["label"]
        })

    # 4. Report Comparative Stats
    df = pd.DataFrame(results)
    
    print("--- ACCURACY COMPARISON (Top-1 Winner is Positive) ---")
    comparison = df.groupby("category").agg({
        "rag_winner": "mean",
        "als_winner": "mean"
    })
    print(comparison)
    
    print("\n--- ERROR ANALYSIS: WHAT RAG MISSES ---")
    # Find cases where RAG failed but ALS won
    rag_failures = df[(df["rag_winner"] == False) & (df["als_winner"] == True)]
    print(f"Total 'Rescue' cases (ALS Win / RAG Loss): {len(rag_failures)}")
    if not rag_failures.empty:
        print(rag_failures.groupby("category").size())

if __name__ == "__main__":
    run_benchmark()
