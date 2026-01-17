import os
import json
import time
import math
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# Constants
EMBEDDING_MODEL = "text-embedding-004"
CHUNK_SIZE = 100

class NarrativeNode(BaseModel):
    id: str
    event: str
    d_days: float  # Relative to "Today" (Node 20 is d=0)
    emotional_state: str
    emotional_intensity: float
    semantic_vec: Optional[List[float]] = None
    emotional_vec: Optional[List[float]] = None

class NarrativeBenchmark(BaseModel):
    spine: List[NarrativeNode]
    echoes: List[NarrativeNode]

class NarrativeGenerator:
    def __init__(self, output_dir: str = "./v1/data"):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found.")
        self.client = genai.Client(api_key=api_key)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_narrative_structure(self) -> NarrativeBenchmark:
        persona = "A surgical resident in a high-volume ER."
        
        sys_instr = f"""
Persona: {persona}
Task: Generate a concise 'Golden Narrative' benchmark for a research paper.

STRUCTURE:
1. SPINE (7 Nodes): 
   - Nodes 1-3: ER Prep. Scrubbing in, checking charts. Mundane. Intensity: 0.05-0.15. d_days: -0.5 to -0.3.
   - Node 4: THE CRISIS. 'Code Red'. Massive hemorrhage in OR 3. Intensity: 1.0. d_days: -0.2.
   - Nodes 5-7: Aftermath. Closing the patient, debriefing, terminal paperwork. Intensity: 0.3-0.5. d_days: -0.1 to 0.0.

2. ECHOES (2 Nodes):
   - Distant past memories (years ago).
   - Echo 1: Father's emergency surgery. Helplessness in the waiting room. Intensity: 0.9. d_days: -1500.
   - Echo 2: A childhood accident. The smell of antiseptic for the first time. Intensity: 0.7. d_days: -3000.

Output in JSON format matching the schema.
"""
        print(">>> Generating Short-Story Narrative via Gemini...")
        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Create the surgical resident golden narrative benchmark.",
            config=types.GenerateContentConfig(
                system_instruction=sys_instr,
                response_mime_type="application/json",
                response_schema=NarrativeBenchmark.model_json_schema(),
            )
        )
        
        data = json.loads(response.text)
        return NarrativeBenchmark(**data)

    def embed_and_save(self, benchmark: NarrativeBenchmark, filename: str):
        print(">>> Embedding Narrative Nodes...")
        all_nodes = benchmark.spine + benchmark.echoes
        
        semantic_texts = [n.event for n in all_nodes]
        emotional_texts = [n.emotional_state for n in all_nodes]

        def get_embeddings(texts: List[str], task: str):
            results = []
            for i in range(0, len(texts), CHUNK_SIZE):
                chunk = texts[i:i+CHUNK_SIZE]
                print(f"   Embedding {task} chunk...")
                res = self.client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=chunk,
                    config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
                )
                results.extend([e.values for e in res.embeddings])
            return results

        sem_vecs = get_embeddings(semantic_texts, "Semantic")
        emo_vecs = get_embeddings(emotional_texts, "Emotional")

        for i, node in enumerate(all_nodes):
            node.semantic_vec = sem_vecs[i]
            node.emotional_vec = emo_vecs[i]

        output_path = self.output_dir / filename
        
        # Format as a graph-ready JSON
        graph_data = {
            "nodes": [n.model_dump() for n in all_nodes],
            "edges": []
        }
        # Add linear spine edges
        for i in range(len(benchmark.spine) - 1):
            graph_data["edges"].append({
                "source": benchmark.spine[i].id,
                "target": benchmark.spine[i+1].id,
                "type": "causal"
            })

        with open(output_path, "w") as f:
            json.dump(graph_data, f, indent=4)
        print(f"\nNarrative Benchmark Graph saved to {output_path}")
        return output_path

if __name__ == "__main__":
    gen = NarrativeGenerator()
    structure = gen.generate_narrative_structure()
    gen.embed_and_save(structure, "narrative_benchmark_graph.json")
