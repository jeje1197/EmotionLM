#!/usr/bin/env python3
import torch
import torch.nn as nn
import json
import numpy as np
from pathlib import Path
from model_mlp import ALSModelMLP
from google import genai
from google.genai import types
import os
import math

GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")

class ALSEvaluatorMLP:
    def __init__(self, model_path: str):
        self.model = ALSModelMLP(hidden_dim=8)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
        
    def calculate_temporal_proximity(self, d_days_a, d_days_b):
        delta = abs(d_days_a - d_days_b)
        # Match the log-transform used in training
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
            intensity = cand['emotional_intensity']
            
            feat_tensor = torch.tensor([[s, e, t, intensity]], dtype=torch.float32)
            with torch.no_grad():
                score = self.model(feat_tensor).item()
            
            results.append({
                "event": cand['event'],
                "score": score,
                "features": {"S": float(s), "E": float(e), "T": float(t), "I": float(intensity)},
                "label": cand.get('label', 'unknown')
            })
            
        results.sort(key=lambda x: x['score'], reverse=True)
        return results

def main():
    # These are the same test cases as v3
    causal_test = {
        "anchor": {"event": "A car engine starts making a high-pitched grinding metal noise.", "d_days": 0.0, "emotional_state": "Sharp anxiety, hyper-focus on sound.", "emotional_intensity": 0.7},
        "candidates": [
            {"event": "The car engine seizes and smoke pours out of the hood.", "d_days": 0.007, "emotional_state": "Panic, frustration, helplessness.", "emotional_intensity": 0.9, "label": "TRUE_LINK"},
            {"event": "Finished a podcast episode about the Roman Empire.", "d_days": 0.008, "emotional_state": "Mild intellectual curiosity.", "emotional_intensity": 0.2, "label": "RANDOM"},
            {"event": "Noticed a nice sunset in the rearview mirror.", "d_days": 0.005, "emotional_state": "Brief aesthetic appreciation.", "emotional_intensity": 0.3, "label": "RANDOM"}
        ]
    }

    affective_test = {
        "anchor": {"event": "Received a failing grade on a final medical board exam.", "d_days": 0.0, "emotional_state": "Crushing weight in chest, numbness, fear for the future.", "emotional_intensity": 0.95},
        "candidates": [
            {"event": "Received a formal rejection letter from a residency program.", "d_days": 30.0, "emotional_state": "Deep hollow ache, familiar sense of inadequacy.", "emotional_intensity": 0.9, "label": "TRUE_LINK"},
            {"event": "Ordered a latte at a quiet cafe.", "d_days": 31.0, "emotional_state": "Routine calm, slightly distracted.", "emotional_intensity": 0.1, "label": "RANDOM"}
        ]
    }

    causal_hard = {
        "anchor": {"event": "Spending four hours preparing a five-course meal for a high-profile guest.", "d_days": 0.0, "emotional_state": "Focused exhaustion, quiet pride, performance anxiety.", "emotional_intensity": 0.65},
        "candidates": [
            {"event": "Received a handwritten note praising the dinner's specific flavors.", "d_days": 3.0, "emotional_state": "Warm glow of validation and success.", "emotional_intensity": 0.75, "label": "TRUE_LINK_DELAYED"},
            {"event": "A wine glass shatters on the kitchen tile, requiring immediate cleanup.", "d_days": 0.01, "emotional_state": "Sharp spike of irritation and frustration.", "emotional_intensity": 0.8, "label": "DISTRACTOR_IMMEDIATE"}
        ]
    }

    affective_hard = {
        "anchor": {"event": "Being a 5-year-old child and losing sight of my parents in a crowded shopping mall.", "d_days": 0.0, "emotional_state": "Panic, absolute isolation, a feeling of being small and invisible.", "emotional_intensity": 0.95},
        "candidates": [
            {"event": "Standing on a busy street corner in Tokyo, realizing I don't know the way back to the hotel.", "d_days": 7000.0, "emotional_state": "Adult version of the same primal isolation; being small in a vast crowd.", "emotional_intensity": 0.85, "label": "TRUE_AFFECTIVE_LINK"},
            {"event": "Browsing for running shoes in a modern suburban shopping mall.", "d_days": 7000.1, "emotional_state": "Mild boredom, routine shopping focus.", "emotional_intensity": 0.2, "label": "SEMANTIC_TRAP"}
        ]
    }

    # 5. Low-Intensity Causal: Routine Task
    # Anchor: Turning the key in the ignition.
    # A. (Link) Dashboard lights flicker to life (Immediate, Low Intensity).
    # B. (No Link) Neighbor's dog barks (Immediate, slightly higher intensity).
    causal_low_intensity = {
        "anchor": {
            "event": "Turning the key in the ignition of an old car.",
            "d_days": 0.0,
            "emotional_state": "Routine focus, slight mechanical expectation.",
            "emotional_intensity": 0.2
        },
        "candidates": [
            {
                "event": "The engine coughs and then starts with a steady rumble.",
                "d_days": 0.00002, # ~2 seconds
                "emotional_state": "Mild satisfaction that the car started.",
                "emotional_intensity": 0.25,
                "label": "TRUE_LINK_LOW_I"
            },
            {
                "event": "A neighbor's dog across the street starts barking at a squirrel.",
                "d_days": 0.00001, # ~1 second
                "emotional_state": "Fleeting awareness of the sound.",
                "emotional_intensity": 0.3,
                "label": "RANDOM_LOW_I"
            }
        ]
    }

    # 6. More Mundane Tasks: Procedure vs. Noise
    # Anchor: Opening a laptop.
    # Link: Seeing the login screen background (Low Intensity, Immediate).
    # Noise: A bird chirps outside.
    causal_laptop = {
        "anchor": {
            "event": "Opening the lid of a silver laptop on a desk.",
            "d_days": 0.0,
            "emotional_state": "Routine work readiness, neutral focus.",
            "emotional_intensity": 0.15
        },
        "candidates": [
            {
                "event": "The screen illuminates, showing a mountain landscape background.",
                "d_days": 0.00003, # 3 seconds
                "emotional_state": "Neutral recognition of the screen being on.",
                "emotional_intensity": 0.1,
                "label": "TRUE_LINK_MUNDANE"
            },
            {
                "event": "A small bird chirps on a tree branch outside the window.",
                "d_days": 0.00002,
                "emotional_state": "Passing auditory perception, no focus.",
                "emotional_intensity": 0.1,
                "label": "RANDOM_MUNDANE"
            }
        ]
    }

    # Anchor: Putting toast in the toaster.
    # Link: The lever pops up and bread is toasted (Mid Intensity, 2 min delay).
    # Noise: Noticed a slightly dirty plate in the sink.
    causal_toast = {
        "anchor": {
            "event": "Slicing bread and pushing down the lever on the toaster.",
            "d_days": 0.0,
            "emotional_state": "Hunger, basic morning routine focus.",
            "emotional_intensity": 0.25
        },
        "candidates": [
            {
                "event": "The toaster pops up with two browned slices of sourdough.",
                "d_days": 0.0014, # ~2 minutes
                "emotional_state": "Slight satisfaction, ready to eat.",
                "emotional_intensity": 0.35,
                "label": "TRUE_LINK_MUNDANE"
            },
            {
                "event": "Noticed a small coffee stain on the bottom of a nearby plate.",
                "d_days": 0.0007, # 1 minute
                "emotional_state": "Fleeting, low-level annoyance.",
                "emotional_intensity": 0.2,
                "label": "RANDOM_MUNDANE"
            }
        ]
    }

    print("--- EVALUATING V4 MLP MODELS (Hidden Dim 8) ---")
    
    eval_causal = ALSEvaluatorMLP("v1/artifacts/pretrained/als_causal_v4_mlp.pt")
    eval_affective = ALSEvaluatorMLP("v1/artifacts/pretrained/als_affective_v4_mlp.pt")
    eval_combined = ALSEvaluatorMLP("v1/artifacts/pretrained/als_combined_v4_mlp.pt")

    scenarios = [
        ("CAUSAL EASY (Engine)", causal_test, eval_causal, eval_affective, eval_combined),
        ("AFFECTIVE EASY (Exam)", affective_test, eval_causal, eval_affective, eval_combined),
        ("CAUSAL HARD (Dinner)", causal_hard, eval_causal, eval_affective, eval_combined),
        ("AFFECTIVE HARD (Mall)", affective_hard, eval_causal, eval_affective, eval_combined),
        ("CAUSAL ROUTINE (Ignition)", causal_low_intensity, eval_causal, eval_affective, eval_combined),
        ("CAUSAL ROUTINE (Laptop)", causal_laptop, eval_causal, eval_affective, eval_combined),
        ("CAUSAL ROUTINE (Toaster)", causal_toast, eval_causal, eval_affective, eval_combined)
    ]

    all_results = {}

    for name, test, m_cau, m_aff, m_com in scenarios:
        print(f"\n[{name}]")
        res_cau = m_cau.evaluate_scenario(test['anchor'], test['candidates'])
        res_aff = m_aff.evaluate_scenario(test['anchor'], test['candidates'])
        res_com = m_com.evaluate_scenario(test['anchor'], test['candidates'])
        
        # We'll use the candidate event string as the key to align scores
        events = [c['event'] for c in test['candidates']]
        
        print(f"{'Candidate Event':<40} | {'Causal':<7} | {'Affect':<7} | {'Combin':<7}")
        print("-" * 75)
        
        for event_name in events:
            s_cau = next(r['score'] for r in res_cau if r['event'] == event_name)
            s_aff = next(r['score'] for r in res_aff if r['event'] == event_name)
            s_com = next(r['score'] for r in res_com if r['event'] == event_name)
            print(f"{event_name[:40]:<40} | {s_cau:.4f}  | {s_aff:.4f}  | {s_com:.4f}")

    # Output simplified summary to JSON
    output_path = "v1/artifacts/llm_judge/v4_mlp_comparison_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
