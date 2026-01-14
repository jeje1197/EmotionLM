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

# Constants for Stage 2
EMBEDDING_MODEL = "text-embedding-004"
CHUNK_SIZE = 100

class QuadEvent(BaseModel):
    event: str = Field(description="Natural language description of the event")
    d_days: float = Field(description="Days difference relative to anchor event (anchor d_days=0.0)")
    emotional_state: str = Field(description="Descriptive emotional state (e.g., 'A sharp spike of professional anxiety mixed with intense focus')")
    emotional_intensity: float = Field(description="Intensity of the emotion from 0.0 to 1.0")
    semantic_vec: Optional[List[float]] = None
    emotional_vec: Optional[List[float]] = None

class MemoryQuad(BaseModel):
    anchor: QuadEvent = Field(description="The source/anchor memory")
    positive: QuadEvent = Field(description="The true linked memory (causal consequence or affective match)")
    temporal_distractor: QuadEvent = Field(description="Negative sample: Unrelated event at the same time as positive")
    semantic_distractor: QuadEvent = Field(description="Negative sample: Related keyword event at a different time")

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
    """Calculate proximity using a log-compressed reciprocal to handle long time gaps."""
    delta_days = abs(d_days_a - d_days_b)
    # Using log1p(x) for ln(1+x) stability
    return 1.0 / (1.0 + math.log1p(delta_days))

class NarrativeEpisodicGenerator:
    def __init__(self, output_dir: str = "./v1/data"):
        # Uses standard GEMINI_API_KEY environment variable
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Environment variable 'GEMINI_API_KEY' not set.")
        
        self.client = genai.Client(api_key=api_key)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def get_personas(self) -> List[str]:
        # Balanced spectrum: Technical/Professional vs. Domestic/Social
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

    def _generate_with_retry(self, category: str, count: int, persona: str, max_retries=2):
        """Custom backoff retry mechanism (standard library only)."""
        attempts = 0
        while attempts <= max_retries:
            try:
                return self._api_call(category, count, persona)
            except Exception as e:
                attempts += 1
                if attempts > max_retries:
                    print(f"Final failure for {persona}: {e}")
                    raise e
                wait_time = (2 ** (attempts + 1)) + random.uniform(0, 1)
                print(f"Attempt {attempts} failed. Retrying in {wait_time:.2f}s...")
                time.sleep(wait_time)

    def _api_call(self, category: str, count: int, persona: str) -> List[Dict]:
        if category == "causal":
            # Causal now spans from immediate reactions to multi-day outcomes
            logic = (
                "Positive: Logical consequence or intentional goal fulfillment. "
                "d_days: Mix of immediate (0.001-0.005), short-term (0.04-0.5), and multi-day outcomes (1.0-7.0). "
                "Temporal Distractor: Unrelated event at ALMOST THE SAME d_days as the Positive event. "
                "Semantic Distractor: Shared nouns/objects but 4-6 months later (d=120-180)."
            )
        else:
            logic = (
                "Positive: Emotional state match or mood-congruent memory. Primarily linked by internal feeling/intensity rather than time. "
                "d_days: Diverse range including recent (2-14 days), medium (30-90 days), and long-term (365-2000 days). "
                "Temporal Distractor: Unrelated neutral event at ALMOST THE SAME d_days as the Positive event. "
                "Semantic Distractor: Shared objects/nouns but different context and time (d=60-150)."
            )

        sys_instr = (
            f"Persona: {persona}\n"
            f"Task: Create {count} episodic memory quads.\n"
            f"Logic: {logic}\n"
            "Guidelines:\n"
            "- No timestamps needed. Use d_days relative to anchor (anchor d_days=0.0).\n"
            "- emotional_state should focus and name the internal feeling and bodily sensation (e.g., 'A hollow, sinking feeling in the chest' or 'An electric surge of triumphant energy').\n"
            "- AVOID repeating details from the event text in the emotional_state field to ensure the emotional embedding is distinct from the semantic embedding.\n"
            "- emotional_intensity should be a float between 0.0 and 1.0 reflecting the impact of the event."
        )

        # Configuration including thinking_budget=0 for Gemini 2.5 Flash
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
        # Handle different response formats from the pydantic-schema API
        raw_json = json.loads(response.text)
        if isinstance(raw_json, dict) and "quads" in raw_json:
            return raw_json["quads"]
        return raw_json

    def run(self, target_per_cat: int, causal_out: Optional[str] = None, affective_out: Optional[str] = None):
        personas = self.get_personas()
        batch_size = 5
        
        configs = [
            ("causal", causal_out or f"causal_quads_2.5_flash.json"),
            ("affective", affective_out or f"affective_quads_2.5_flash.json")
        ]
        
        for category, filename in configs:
            category_dataset = []
            print(f"\n>>> Generating {category.upper()} Data")
            num_batches = target_per_cat // batch_size
            for i in range(num_batches):
                persona = personas[i % len(personas)]
                try:
                    batch = self._generate_with_retry(category, batch_size, persona)
                    category_dataset.extend(batch)
                    print(f"Total Successes: {len(category_dataset)} {category} quads.")
                except Exception:
                    continue
                
                output_path = Path(filename) if filename.endswith(".json") else self.output_dir / filename
                if not output_path.is_absolute():
                    output_path = self.output_dir / output_path.name
                
                with open(output_path, "w") as f:
                    json.dump(category_dataset, f, indent=4)
                time.sleep(1.0)

    def embed_quads(self, category: str, input_path: Optional[str] = None, output_path: Optional[str] = None):
        """Part 2: Generate semantic and emotional embeddings."""
        in_p = Path(input_path) if input_path else self.output_dir / f"{category}_quads_2.5_flash.json"
        out_p = Path(output_path) if output_path else self.output_dir / f"{category}_quads_embedded.json"
        
        if not in_p.exists():
            print(f"Skipping {category}: input file {in_p} not found.")
            return

        with open(in_p, "r") as f:
            quads_raw = json.load(f)
        
        quads = [MemoryQuad(**q) for q in quads_raw]
        
        # Collect all texts to embed
        semantic_texts = []
        emotional_texts = []
        
        for q in quads:
            for event in [q.anchor, q.positive, q.temporal_distractor, q.semantic_distractor]:
                semantic_texts.append(event.event)
                emotional_texts.append(event.emotional_state)
        
        print(f"Generating {len(semantic_texts)} semantic and {len(emotional_texts)} emotional embeddings for {category}...")
        
        def batch_embed(texts: List[str], task_type: str):
            all_vecs = []
            for i in range(0, len(texts), CHUNK_SIZE):
                chunk = texts[i:i + CHUNK_SIZE]
                res = self.client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=chunk,
                    config=types.EmbedContentConfig(task_type=task_type)
                )
                all_vecs.extend([e.values for e in res.embeddings])
            return all_vecs

        sem_vecs = batch_embed(semantic_texts, "SEMANTIC_SIMILARITY")
        emo_vecs = batch_embed(emotional_texts, "SEMANTIC_SIMILARITY")
        
        # Re-assign vectors
        vec_idx = 0
        for q in quads:
            for event in [q.anchor, q.positive, q.temporal_distractor, q.semantic_distractor]:
                event.semantic_vec = sem_vecs[vec_idx]
                event.emotional_vec = emo_vecs[vec_idx]
                vec_idx += 1
        
        with open(out_p, "w") as f:
            json.dump([q.model_dump() for q in quads], f, indent=4)
        print(f"Embedded data saved to {out_p}")

    def create_csv(self, category: str, input_path: Optional[str] = None, output_path: Optional[str] = None):
        """Part 3: Flatten quads into training pairs CSV."""
        in_p = Path(input_path) if input_path else self.output_dir / f"{category}_quads_embedded.json"
        out_p = Path(output_path) if output_path else self.output_dir / f"{category}_training_dataset.csv"
        
        if not in_p.exists():
            print(f"Skipping {category}: embedded file {in_p} not found.")
            return

        with open(in_p, "r") as f:
            quads_raw = json.load(f)
        
        quads = [MemoryQuad(**q) for q in quads_raw]
        rows = []
        
        for q in quads:
            anchor = q.anchor
            # Pairs: (Positive, 1), (Temporal, 0), (Semantic, 0)
            targets = [
                (q.positive, 1, "positive"),
                (q.temporal_distractor, 0, "temporal_neg"),
                (q.semantic_distractor, 0, "semantic_neg")
            ]
            
            for target, label, pair_type in targets:
                sem_sim = cosine_similarity(anchor.semantic_vec, target.semantic_vec)
                emo_sim = cosine_similarity(anchor.emotional_vec, target.emotional_vec)
                temp_prox = calculate_temporal_proximity(anchor.d_days, target.d_days)
                
                rows.append({
                    "Semantic_Similarity": sem_sim,
                    "Emotional_Alignment": emo_sim,
                    "Time_Closeness": temp_prox,
                    "Target_Intensity": target.emotional_intensity,
                    "Linked": label,
                    "Pair_Type": pair_type,
                    "Category": category
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(out_p, index=False)
        print(f"CSV dataset saved to {out_p}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["generate", "embed", "csv", "all"], default="all")
    parser.add_argument("--n", type=int, default=50, help="Quads per category")
    parser.add_argument("--output-dir", type=str, default="./v1/data", help="Root directory for output")
    
    # Specific file path overrides
    parser.add_argument("--causal-json", type=str, help="Path for causal raw JSON")
    parser.add_argument("--affective-json", type=str, help="Path for affective raw JSON")
    parser.add_argument("--causal-embedded", type=str, help="Path for causal embedded JSON")
    parser.add_argument("--affective-embedded", type=str, help="Path for affective embedded JSON")
    parser.add_argument("--causal-csv", type=str, help="Path for causal CSV output")
    parser.add_argument("--affective-csv", type=str, help="Path for affective CSV output")

    args = parser.parse_args()
    
    try:
        gen = NarrativeEpisodicGenerator(output_dir=args.output_dir)
        
        if args.stage in ["generate", "all"]:
            gen.run(args.n, causal_out=args.causal_json, affective_out=args.affective_json)
            
        if args.stage in ["embed", "all"]:
            gen.embed_quads("causal", input_path=args.causal_json, output_path=args.causal_embedded)
            gen.embed_quads("affective", input_path=args.affective_json, output_path=args.affective_embedded)
                
        if args.stage in ["csv", "all"]:
            gen.create_csv("causal", input_path=args.causal_embedded, output_path=args.causal_csv)
            gen.create_csv("affective", input_path=args.affective_embedded, output_path=args.affective_csv)
            
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")