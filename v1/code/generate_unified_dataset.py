import os
import json
import time
import random
import argparse
import math
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# Constants
EMBEDDING_MODEL = "text-embedding-004"
CHUNK_SIZE = 100

class QuadEvent(BaseModel):
    event: str = Field(description="Natural language description of the event")
    d_days: float = Field(description="Days difference relative to anchor event (anchor d_days=0.0)")
    emotional_state: str = Field(description="Descriptive emotional state")
    emotional_intensity: float = Field(description="Intensity of the emotion from 0.0 to 1.0")
    semantic_vec: Optional[List[float]] = None
    emotional_vec: Optional[List[float]] = None

class MemoryQuad(BaseModel):
    anchor: QuadEvent = Field(description="The source/anchor memory")
    positive: QuadEvent = Field(description="The true linked memory")
    temporal_distractor: QuadEvent = Field(description="Negative sample: Unrelated event at the same time as positive")
    semantic_distractor: QuadEvent = Field(description="Negative sample: Related keyword event at a different time")
    category: str = Field(description="The type of link logic used (causal or affective)")

class QuadBatch(BaseModel):
    quads: List[MemoryQuad]

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    a = np.array(v1)
    b = np.array(v2)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.5
    sim = dot / (norm_a * norm_b)
    return (float(sim) + 1.0) / 2.0

def calculate_temporal_proximity(d_days_a: float, d_days_b: float) -> float:
    """Standardized Log-Compressed Reciprocal for the paper."""
    delta_days = abs(d_days_a - d_days_b)
    return 1.0 / (1.0 + math.log1p(delta_days))

class UnifiedDatasetGenerator:
    def __init__(self, output_dir: str = "./v1/data"):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Environment variable 'GEMINI_API_KEY' not set.")
        
        self.client = genai.Client(api_key=api_key)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def get_personas(self) -> List[str]:
        return [
            "An air traffic controller during a thunderstorm.",
            "A grandparent teaching a toddler to plant seeds.",
            "A software engineer debugging a site outage.",
            "A hobbyist astronomer tracking a comet.",
            "A surgical resident in a high-volume ER.",
            "A first-time homebuyer stripping old wallpaper.",
            "A forensic accountant tracing a money trail.",
            "A socialite host realizing they've run out of ice.",
            "A commercial pilot navigating turbulence.",
            "A long-distance runner hitting mile 20 of a race.",
            "A librarian archiving rare, fragile manuscripts.",
            "A teenager learning to bake their first cake alone.",
            "A structural engineer inspecting a bridge at dawn.",
            "A tired parent trying to finish a laundry cycle at midnight.",
            "A concept artist sketching in a busy city park."
        ]

    def _api_call(self, category: str, count: int, persona: str) -> List[Dict]:
        if category == "causal":
            # Focus on Routine/Logistical Logic (Low Intensity)
            logic = (
                "Goal: Routine/Logistical Continuity. "
                "Context: Mundane daily actions with VERY LOW Emotional Intensity (0.05-0.25). "
                "Positive: The immediate logical next step or direct consequence. "
                "d_days: Immediate (0.0001 to 0.01 days). "
                "Temporal Distractor: Unrelated mundane event at the same time as positive. "
                "Semantic Distractor: Shared noun but unrelated task 100+ days later."
            )
        else:
            # Focus on Flashbulb/Affective Logic (High Intensity)
            logic = (
                "Goal: Affective Association (Emotional Wormhole). "
                "Context: High-stakes, intense events with HIGH Emotional Intensity (0.7-1.0). "
                "Positive: A past memory that SHARES the same specific emotion/bodily sensation. "
                "d_days: Distant past (30 to 2000 days ago). "
                "Temporal Distractor: Unrelated neutral event at that same distant time. "
                "Semantic Distractor: Shared object but different emotion 10-60 days later."
            )

        sys_instr = (
            f"Persona: {persona}\n"
            f"Task: Create {count} episodic memory quads for a retrieval benchmarking dataset.\n"
            f"Category Logic: {logic}\n"
            "Guidelines:\n"
            "- AVOID repeating nouns from anchor in positive for affective links unless necessary.\n"
            "- Ensure emotional_intensity strictly follows the category logic (0.05-0.25 for causal, 0.7-1.0 for affective).\n"
            "- The 'positive' event must be the THEORETICALLY correct retrieval target for this persona."
        )

        config = types.GenerateContentConfig(
            system_instruction=sys_instr,
            response_mime_type="application/json",
            response_schema=QuadBatch.model_json_schema(),
            thinking_config=types.ThinkingConfig(thinking_budget=0) 
        )

        response = self.client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=f"Generate {count} {category} memory quads.", 
            config=config
        )
        data = json.loads(response.text)
        quads = data.get("quads", data)
        for q in quads:
            q["category"] = category
        return quads

    def generate(self, count_per_cat: int) -> List[Dict]:
        personas = self.get_personas()
        all_quads = []
        batch_size = 5
        
        for category in ["causal", "affective"]:
            print(f"\n--- Generating {category.upper()} quads ---")
            num_batches = (count_per_cat + batch_size - 1) // batch_size
            for i in range(num_batches):
                persona = personas[i % len(personas)]
                print(f"Batch {i+1}/{num_batches} | Persona: {persona[:40]}...")
                try:
                    batch = self._api_call(category, batch_size, persona)
                    all_quads.extend(batch)
                except Exception as e:
                    print(f"Error: {e}")
                    continue
                time.sleep(2.0)
        
        return all_quads

    def embed_and_save_json(self, quads_raw: List[Dict], filename: str):
        print("\n--- Starting Embedding Phase ---")
        quads = [MemoryQuad(**q) for q in quads_raw]
        
        semantic_texts = []
        emotional_texts = []
        for q in quads:
            for event in [q.anchor, q.positive, q.temporal_distractor, q.semantic_distractor]:
                semantic_texts.append(event.event)
                emotional_texts.append(event.emotional_state)

        def get_embeddings(texts: List[str], task: str):
            results = []
            for i in range(0, len(texts), CHUNK_SIZE):
                chunk = texts[i:i+CHUNK_SIZE]
                print(f"Embedding {task} chunk {i//CHUNK_SIZE + 1}...")
                res = self.client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=chunk,
                    config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
                )
                results.extend([e.values for e in res.embeddings])
            return results

        sem_vecs = get_embeddings(semantic_texts, "Semantic")
        emo_vecs = get_embeddings(emotional_texts, "Emotional")

        idx = 0
        for q in quads:
            for event in [q.anchor, q.positive, q.temporal_distractor, q.semantic_distractor]:
                event.semantic_vec = sem_vecs[idx]
                event.emotional_vec = emo_vecs[idx]
                idx += 1

        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            json.dump([q.model_dump() for q in quads], f, indent=4)
        print(f"Unified JSON saved to {output_path}")
        return quads

    def export_csv(self, quads: List[MemoryQuad], filename: str):
        rows = []
        for q in quads:
            anchor = q.anchor
            targets = [
                (q.positive, 1, "positive"),
                (q.temporal_distractor, 0, "temporal_neg"),
                (q.semantic_distractor, 0, "semantic_neg")
            ]
            
            for target, label, pair_type in targets:
                s = cosine_similarity(anchor.semantic_vec, target.semantic_vec)
                e = cosine_similarity(anchor.emotional_vec, target.emotional_vec)
                t = calculate_temporal_proximity(anchor.d_days, target.d_days)
                i = target.emotional_intensity
                
                rows.append({
                    "Semantic_Similarity": s,
                    "Emotional_Alignment": e,
                    "Time_Closeness": t,
                    "Target_Intensity": i,
                    "Linked": label,
                    "Pair_Type": pair_type,
                    "Category": q.category
                })
        
        df = pd.DataFrame(rows)
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)
        print(f"Training CSV saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100, help="Quads per category")
    args = parser.parse_args()

    gen = UnifiedDatasetGenerator()
    
    # 1. Generate
    raw_quads = gen.generate(args.count)
    
    # 2. Embed & Save JSON (Unified)
    embedded_quads = gen.embed_and_save_json(raw_quads, "unified_memory_dataset.json")
    
    # 3. Export CSV (For Training)
    gen.export_csv(embedded_quads, "unified_training_dataset.csv")

if __name__ == "__main__":
    main()
