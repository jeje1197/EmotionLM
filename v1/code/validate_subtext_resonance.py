import json
import torch
import numpy as np
import os
from pathlib import Path
from google import genai
from google.genai import types

# Paths
MODEL_PATH = "v1/artifacts/pretrained/als_unified_linear.pt"
GRAPH_PATH = "v1/data/narrative_benchmark_graph.json"

class ALSModel(torch.nn.Module):
    def __init__(self, input_dim: int = 4):
        super().__init__()
        self.slp = torch.nn.Linear(input_dim, 1)
    def forward(self, x):
        return torch.sigmoid(self.slp(x))

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def calculate_temporal(d1, d2):
    delta = abs(d1 - d2)
    return 1.0 / (1.0 + np.log1p(delta))

def validate_subtext_resonance():
    # 1. Setup
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key: raise ValueError("GEMINI_API_KEY not found.")
    client = genai.Client(api_key=api_key)

    model = ALSModel(input_dim=4)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    with open(GRAPH_PATH, "r") as f:
        data = json.load(f)
    nodes = {n["id"]: n for n in data["nodes"]}

    # 2. Pick the Crisis Node (S4)
    anchor = nodes["S4"]
    
    # 3. Execution: Run ALS Global Search (Spreading Activation)
    spreading_results = []
    for other_id, other in nodes.items():
        if other_id == "S4" or other_id.startswith("S"): continue # Ignore spine
        
        s = (cosine_similarity(anchor["semantic_vec"], other["semantic_vec"]) + 1) / 2
        e = (cosine_similarity(anchor["emotional_vec"], other["emotional_vec"]) + 1) / 2
        t = calculate_temporal(anchor["d_days"], other["d_days"])
        i_val = other["emotional_intensity"]
        
        feat = torch.tensor([[s, e, t, i_val]], dtype=torch.float32)
        with torch.no_grad():
            score = model(feat).item()
        spreading_results.append((other, score))

    spreading_results.sort(key=lambda x: x[1], reverse=True)
    top_echo = spreading_results[0][0]
    score = spreading_results[0][1]

    print("=== ALS SUBTEXT RESONANCE VALIDATION ===")
    print(f"ANHCHOR CRISIS: \"{anchor['event']}\"")
    print(f"RETRIEVED SUBTEXT: \"{top_echo['event']}\"")
    print(f"ALS SCORE: {score:.4f}\n")

    # 4. LLM-as-a-Judge: Resonance Scoring
    judge_prompt = f"""
As an expert literary critic and cognitive scientist, evaluate the 'Resonance' between an anchor event and a retrieved subtextual memory.

ANCHOR (Current Crisis): {anchor['event']}
SUBTEXT (Retrieved Memory): {top_echo['event']}

CRITERIA:
1. THEMATIC ALIGNMENT (1-5): Does the subtext share a deep emotional root (e.g., helplessness, trauma) with the anchor?
2. NARRATIVE DEPTH (1-5): Does providing this subtext for an LLM to generate from create character depth? (5 = unique subtext, 1 = redundant/literal fact).

Output ONLY JSON:
{{"thematic_score": int, "depth_score": int, "explanation": str}}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=judge_prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json")
    )
    
    result = json.loads(response.text)
    print(f"--- JUDGE VERDICT ---")
    print(f"Thematic Alignment: {result['thematic_score']}/5")
    print(f"Narrative Depth: {result['depth_score']}/5")
    print(f"Explanation: {result['explanation']}")

if __name__ == "__main__":
    validate_subtext_resonance()
