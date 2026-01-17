import os
import json
import time
import torch
import numpy as np
from pathlib import Path
from google import genai
from google.genai import types

# Constants
MODEL_WEIGHTS_PATH = "v1/artifacts/pretrained/als_unified_linear.pt"
DATASET_PATH = "v1/data/unified_memory_dataset.json"

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

class LLMJudgeBenchmark:
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found.")
        self.client = genai.Client(api_key=self.api_key)
        
        # Load ALS Model
        self.model = ALSModel(input_dim=4)
        self.model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location="cpu"))
        self.model.eval()
        
        # Load Dataset
        with open(DATASET_PATH, "r") as f:
            self.data = json.load(f)

    def get_context_rag(self, anchor, candidates, top_k=3):
        scored = []
        for c in candidates:
            score = (cosine_similarity(anchor["semantic_vec"], c["semantic_vec"]) + 1) / 2
            scored.append((c, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def get_context_als(self, anchor, candidates, top_k=3):
        scored = []
        for c in candidates:
            s = (cosine_similarity(anchor["semantic_vec"], c["semantic_vec"]) + 1) / 2
            e = (cosine_similarity(anchor["emotional_vec"], c["emotional_vec"]) + 1) / 2
            t = calculate_temporal(anchor["d_days"], c["d_days"])
            i = c["emotional_intensity"]
            
            feat = torch.tensor([[s, e, t, i]], dtype=torch.float32)
            with torch.no_grad():
                score = self.model(feat).item()
            scored.append((c, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def run_judge(self, anchor, rag_set, als_set):
        prompt = f"""
You are an expert narrative analyst and cognitive scientist.
I am comparing two retrieval algorithms that help an AI simulate human memory for a specific persona.

CURRENT SITUATION:
"{anchor['event']}"
Emotion: {anchor['emotional_state']} (Intensity: {anchor['emotional_intensity']})

MEMORY SET A:
{chr(10).join([f"- {n[0]['event']} (Emotion: {n[0]['emotional_state']})" for n in rag_set])}

MEMORY SET B:
{chr(10).join([f"- {n[0]['event']} (Emotion: {n[0]['emotional_state']})" for n in als_set])}

TASK:
Which set of memories (A or B) provides a more meaningful context for an LLM to accurately simulate this character's internal thoughts or predict their next reaction? 
- "A" often focuses on keyword overlap (Semantics).
- "B" focuses on a mix of timing and emotional intensity.

Output your choice and a brief rationale in JSON format.
{{
  "preference": "A" or "B",
  "rationale": "...",
  "key_difference": "..."
}}
"""
        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        return json.loads(response.text)

    def run_benchmark(self, limit=10):
        # Pick random distinct anchors
        anchors = random.sample(self.data, limit)
        all_nodes = []
        for q in self.data:
            all_nodes.extend([q["anchor"], q["positive"], q["temporal_distractor"], q["semantic_distractor"]])
        
        results = []
        for i, quad in enumerate(anchors):
            anchor = quad["anchor"]
            print(f"[{i+1}/{limit}] Judging: {anchor['event'][:50]}...")
            
            # Use all other nodes as our searchable "Memory Bank"
            candidates = [n for n in all_nodes if n["event"] != anchor["event"]]
            
            rag_set = self.get_context_rag(anchor, candidates)
            als_set = self.get_context_als(anchor, candidates)
            
            # Randomized labeling to avoid bias
            labels = ["A", "B"]
            random.shuffle(labels)
            sets = {labels[0]: rag_set, labels[1]: als_set}
            
            judge_res = self.run_judge(anchor, sets["A"], sets["B"])
            
            # Map choice back to algorithm
            winner = "RAG" if judge_res["preference"] == labels[0] else "ALS"
            results.append({
                "winner": winner,
                "category": quad["category"],
                "rationale": judge_res["rationale"]
            })
            time.sleep(1)

        print("\n--- LLM-AS-A-JUDGE RESULTS ---")
        df = pd.DataFrame(results)
        print(df["winner"].value_counts(normalize=True))
        
        print("\n--- RATIONALE SAMPLES ---")
        for res in results[:3]:
            print(f"[{res['winner']}] {res['rationale']}")

if __name__ == "__main__":
    import random
    import pandas as pd
    bench = LLMJudgeBenchmark()
    bench.run_benchmark(limit=10)
