#!/usr/bin/env python3
import torch
import torch.nn as nn
import json
import numpy as np
from pathlib import Path
from model_v2 import ALSModelV2
from google import genai
from google.genai import types
import os

# NOTE: This script assumes Gemini API key is available in environment
# OR it will use mock embeddings if not available for demonstration.
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")

class ALSEvaluator:
    def __init__(self, model_path: str):
        self.model = ALSModelV2()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
        
    def calculate_temporal_proximity(self, d_days_a, d_days_b):
        delta = abs(d_days_a - d_days_b)
        # Using log1p(x) for ln(1+x) stability
        import math
        return 1.0 / (1.0 + math.log1p(delta))

    def get_embeddings(self, text_list):
        if not self.client:
            # Fallback to random embeddings for structure testing if client is missing
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
        """
        anchor: dict with {event, d_days, emotional_state, emotional_intensity}
        candidates: list of similar dicts
        """
        # Batch embed
        texts = [anchor['event']] + [c['event'] for c in candidates]
        emotions = [anchor['emotional_state']] + [c['emotional_state'] for c in candidates]
        
        text_embs = self.get_embeddings(texts)
        emotion_embs = self.get_embeddings(emotions)
        
        anchor_text_emb = text_embs[0]
        anchor_emotion_emb = emotion_embs[0]
        
        results = []
        for i, cand in enumerate(candidates):
            idx = i + 1
            # Features
            s = self.cosine_similarity(anchor_text_emb, text_embs[idx])
            e = self.cosine_similarity(anchor_emotion_emb, emotion_embs[idx])
            t = self.calculate_temporal_proximity(anchor['d_days'], cand['d_days'])
            intensity = cand['emotional_intensity']
            
            # Predict
            feat_tensor = torch.tensor([[s, e, t, intensity]], dtype=torch.float32)
            with torch.no_grad():
                score = self.model(feat_tensor).item()
            
            results.append({
                "event": cand['event'],
                "score": score,
                "features": {"S": s, "E": e, "T": t, "I": intensity},
                "label": cand.get('label', 'unknown')
            })
            
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        return results

def main():
    # 1. Unbiased Causal Test Scenario
    # Anchor: A car engine makes a loud grinding noise.
    # Candidates:
    # A. (Link) Car breaks down on the highway 10 minutes later. (High Intensity, Close Time)
    # B. (No Link) Listened to a podcast about history. (Unrelated, similar time)
    # C. (No Link) Checked the weather forecast. (Low intensity, unrelated)
    
    causal_test = {
        "anchor": {
            "event": "A car engine starts making a high-pitched grinding metal noise.",
            "d_days": 0.0,
            "emotional_state": "Sharp anxiety, hyper-focus on sound.",
            "emotional_intensity": 0.7
        },
        "candidates": [
            {
                "event": "The car engine seizes and smoke pours out of the hood.",
                "d_days": 0.007, # 10 mins
                "emotional_state": "Panic, frustration, helplessness.",
                "emotional_intensity": 0.9,
                "label": "TRUE_LINK"
            },
            {
                "event": "Finished a podcast episode about the Roman Empire.",
                "d_days": 0.008, # 11 mins
                "emotional_state": "Mild intellectual curiosity.",
                "emotional_intensity": 0.2,
                "label": "RANDOM"
            },
            {
                "event": "Noticed a nice sunset in the rearview mirror.",
                "d_days": 0.005, # 7 mins
                "emotional_state": "Brief aesthetic appreciation.",
                "emotional_intensity": 0.3,
                "label": "RANDOM"
            }
        ]
    }

    # 2. Unbiased Affective Test Scenario
    # Anchor: Failing a big exam.
    # Candidates:
    # A. (Link) Getting a rejection letter from a job 1 month later. (High Intensity, similar emotional state)
    # B. (No Link) Buying a cup of coffee. (Low intensity)
    
    affective_test = {
        "anchor": {
            "event": "Received a failing grade on a final medical board exam.",
            "d_days": 0.0,
            "emotional_state": "Crushing weight in chest, numbness, fear for the future.",
            "emotional_intensity": 0.95
        },
        "candidates": [
            {
                "event": "Received a formal rejection letter from a residency program.",
                "d_days": 30.0, # 1 month later
                "emotional_state": "Deep hollow ache, familiar sense of inadequacy.",
                "emotional_intensity": 0.9,
                "label": "TRUE_LINK"
            },
            {
                "event": "Ordered a latte at a quiet cafe.",
                "d_days": 31.0,
                "emotional_state": "Routine calm, slightly distracted.",
                "emotional_intensity": 0.1,
                "label": "RANDOM"
            }
        ]
    }

    # 3. Challenging Causal: Multi-day delay vs immediate distraction
    # Anchor: Preparing a complex dinner.
    # A. (Link) Guest thank you note (3 days later, mid intensity).
    # B. (No Link) Dropped a glass (10 mins later, high intensity).
    causal_hard = {
        "anchor": {
            "event": "Spending four hours preparing a five-course meal for a high-profile guest.",
            "d_days": 0.0,
            "emotional_state": "Focused exhaustion, quiet pride, performance anxiety.",
            "emotional_intensity": 0.65
        },
        "candidates": [
            {
                "event": "Received a handwritten note praising the dinner's specific flavors.",
                "d_days": 3.0,
                "emotional_state": "Warm glow of validation and success.",
                "emotional_intensity": 0.75,
                "label": "TRUE_LINK_DELAYED"
            },
            {
                "event": "A wine glass shatters on the kitchen tile, requiring immediate cleanup.",
                "d_days": 0.01, # 15 mins
                "emotional_state": "Sharp spike of irritation and frustration.",
                "emotional_intensity": 0.8,
                "label": "DISTRACTOR_IMMEDIATE"
            }
        ]
    }

    # 4. Challenging Affective: Semantic Trap
    # Anchor: Childhood lost in a mall.
    # A. (Link) Feeling isolated in a big city years later.
    # B. (No Link) Buying shoes in a mall (High semantic overlap).
    affective_hard = {
        "anchor": {
            "event": "Being a 5-year-old child and losing sight of my parents in a crowded shopping mall.",
            "d_days": 0.0,
            "emotional_state": "Panic, absolute isolation, a feeling of being small and invisible.",
            "emotional_intensity": 0.95
        },
        "candidates": [
            {
                "event": "Standing on a busy street corner in Tokyo, realizing I don't know the way back to the hotel.",
                "d_days": 7000.0, # ~20 years later
                "emotional_state": "Adult version of the same primal isolation; being small in a vast crowd.",
                "emotional_intensity": 0.85,
                "label": "TRUE_AFFECTIVE_LINK"
            },
            {
                "event": "Browsing for running shoes in a modern suburban shopping mall.",
                "d_days": 7000.1,
                "emotional_state": "Mild boredom, routine shopping focus.",
                "emotional_intensity": 0.2,
                "label": "SEMANTIC_TRAP"
            }
        ]
    }

    print("--- EVALUATING UNBIASED SCENARIOS ---")
    
    eval_causal = ALSEvaluator("v1/artifacts/pretrained/als_causal_v3.pt")
    eval_affective = ALSEvaluator("v1/artifacts/pretrained/als_affective_v3.pt")

    print("\n[CAUSAL TEST: EASY]")
    res1 = eval_causal.evaluate_scenario(causal_test['anchor'], causal_test['candidates'])
    for r in res1:
        print(f"Score: {r['score']:.4f} | Label: {r['label']} | Event: {r['event'][:50]}...")

    print("\n[AFFECTIVE TEST: EASY]")
    res2 = eval_affective.evaluate_scenario(affective_test['anchor'], affective_test['candidates'])
    for r in res2:
        print(f"Score: {r['score']:.4f} | Label: {r['label']} | Event: {r['event'][:50]}...")

    print("\n[CAUSAL TEST: HARD (Delay vs Immediate Distractor)]")
    res3 = eval_causal.evaluate_scenario(causal_hard['anchor'], causal_hard['candidates'])
    for r in res3:
        print(f"Score: {r['score']:.4f} | Label: {r['label']} | Event: {r['event'][:50]}...")

    print("\n[AFFECTIVE TEST: HARD (Semantic Trap)]")
    res4 = eval_affective.evaluate_scenario(affective_hard['anchor'], affective_hard['candidates'])
    for r in res4:
        print(f"Score: {r['score']:.4f} | Label: {r['label']} | Event: {r['event'][:50]}...")

    # Save results to JSON
    output_data = {
        "causal_easy": {"anchor": causal_test["anchor"], "results": res1},
        "affective_easy": {"anchor": affective_test["anchor"], "results": res2},
        "causal_hard": {"anchor": causal_hard["anchor"], "results": res3},
        "affective_hard": {"anchor": affective_hard["anchor"], "results": res4}
    }
    
    output_path = "v1/artifacts/llm_judge/unbiased_test_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
