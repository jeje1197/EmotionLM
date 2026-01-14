#!/usr/bin/env python3
import torch
import torch.nn as nn
import json
import numpy as np
from pathlib import Path
from model_mlp_no_i import ALSModelMLPNoI
from google import genai
from google.genai import types
import os
import math

GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")

class ALSEvaluatorNoI:
    def __init__(self, model_path: str):
        self.model = ALSModelMLPNoI(hidden_dim=8)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
        
    def calculate_temporal_proximity(self, d_days_a, d_days_b):
        delta = abs(d_days_a - d_days_b)
        import math
        return 1.0 / (1.0 + math.log1p(delta))

    def get_embeddings(self, text_list):
        if not self.client:
            return np.random.rand(len(text_list), 768)
        res = self.client.models.embed_content(
            model="text-embedding-004",
            contents=text_list,
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
        )
        return np.array([e.values for e in res.embeddings])

    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def evaluate_scenario(self, anchor, candidates):
        texts = [anchor['event']] + [c['event'] for c in candidates]
        emotions = [anchor['emotional_state']] + [c['emotional_state'] for c in candidates]
        
        text_embs = self.get_embeddings(texts)
        emotion_embs = self.get_embeddings(emotions)
        
        anchor_text_emb = text_embs[0]
        anchor_emotion_emb = emotion_embs[0]
        
        results = []
        for i, cand in enumerate(candidates):
            idx = i + 1
            s = self.cosine_similarity(anchor_text_emb, text_embs[idx])
            e = self.cosine_similarity(anchor_emotion_emb, emotion_embs[idx])
            t = self.calculate_temporal_proximity(anchor['d_days'], cand['d_days'])
            
            # NO INTENSITY FEATURE
            feat_tensor = torch.tensor([[s, e, t]], dtype=torch.float32)
            with torch.no_grad():
                score = self.model(feat_tensor).item()
            
            results.append({
                "event": cand['event'],
                "score": score,
                "label": cand.get('label', 'unknown')
            })
            
        results.sort(key=lambda x: x['score'], reverse=True)
        return results

def main():
    # The scenarios from earlier
    causal_low_intensity = {
        "anchor": {"event": "Turning the key in the ignition of an old car.", "d_days": 0.0, "emotional_state": "Routine focus, slight mechanical expectation.", "emotional_intensity": 0.2},
        "candidates": [
            {"event": "The engine coughs and then starts with a steady rumble.", "d_days": 0.00002, "emotional_state": "Satisfaction, mechanical relief.", "emotional_intensity": 0.25, "label": "TRUE_LINK_LOW_I"},
            {"event": "A neighbor's dog across the street starts barking at a squirrel.", "d_days": 0.00001, "emotional_state": "Neutral, ambient noise.", "emotional_intensity": 0.3, "label": "RANDOM_LOW_I"}
        ]
    }

    causal_laptop = {
        "anchor": {"event": "Opening the lid of a silver laptop on a desk.", "d_days": 0.0, "emotional_state": "Routine work readiness, neutral focus.", "emotional_intensity": 0.15},
        "candidates": [
            {"event": "The screen illuminates, showing a mountain landscape background.", "d_days": 0.00003, "emotional_state": "Neutral, focused.", "emotional_intensity": 0.1, "label": "TRUE_LINK_MUNDANE"},
            {"event": "A small bird chirps on a tree branch outside the window.", "d_days": 0.00002, "emotional_state": "Neutral, distracted.", "emotional_intensity": 0.1, "label": "RANDOM_MUNDANE"}
        ]
    }

    exam_failure = {
        "anchor": {"event": "Failing a major exam that was required for graduation.", "d_days": 0.0, "emotional_state": "Devastation, shame, crushing disappointment.", "emotional_intensity": 0.95},
        "candidates": [
            {"event": "Walking to the registrar's office to discuss potential retake options.", "d_days": 1.0, "emotional_state": "Dread, heavy anxiety.", "emotional_intensity": 0.8, "label": "TRUE_CAUSAL_HARD"},
            {"event": "Stopping to tie a shoelace on a quiet library floor.", "d_days": 0.05, "emotional_state": "Neutral, brief annoyance.", "emotional_intensity": 0.1, "label": "HARD_NEGATIVE_DISTRACTOR"},
            {"event": "A stranger wearing a blue hat walks past in the hallway.", "d_days": 0.0001, "emotional_state": "Indifferent.", "emotional_intensity": 0.05, "label": "RANDOM_NOISE"}
        ]
    }

    print("--- EVALUATING INTENSITY-BLIND MLP ---")
    eval_model = ALSEvaluatorNoI("v1/artifacts/pretrained/als_combined_no_i.pt")

    print("\n[CAUSAL ROUTINE (Ignition)]")
    res1 = eval_model.evaluate_scenario(causal_low_intensity['anchor'], causal_low_intensity['candidates'])
    for r in res1: print(f"Score: {r['score']:.4f} | Label: {r['label']} | Event: {r['event'][:50]}...")

    print("\n[CAUSAL ROUTINE (Laptop)]")
    res2 = eval_model.evaluate_scenario(causal_laptop['anchor'], causal_laptop['candidates'])
    for r in res2: print(f"Score: {r['score']:.4f} | Label: {r['label']} | Event: {r['event'][:50]}...")

    print("\n[HIGH-INTENSITY EXAM (Can it still filter?)]")
    res3 = eval_model.evaluate_scenario(exam_failure['anchor'], exam_failure['candidates'])
    for r in res3: print(f"Score: {r['score']:.4f} | Label: {r['label']} | Event: {r['event'][:50]}...")

if __name__ == "__main__":
    main()
