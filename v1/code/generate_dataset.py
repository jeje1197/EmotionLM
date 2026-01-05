#!/usr/bin/env python3
"""Unified dataset generation script following the original notebook pipeline.

Stages (select via --stage):

1) graph   - Generate the diverse 5x100 memory graph from scratch using Gemini (LLM)
2) embed   - Generate semantic & emotional embeddings for each node (Gemini embeddings)
3) csv     - Create the edge classification CSV for training (Time_Closeness, Semantic_Similarity, Emotional_Alignment, Linked)
4) all     - Run graph -> embed -> csv in sequence

This is a CLI version of research/scripts/Dataset_Generation.ipynb, adapted to run locally
using GEMINI_API_KEY from the environment instead of Colab/Drive.
"""

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from google import genai
from google.genai import types


# ---------------------------------------------------------------------------
# Shared configuration (mirrors notebook, but paths are local)
# ---------------------------------------------------------------------------

NUM_STORIES = 5
NODES_PER_STORY = 100
TOTAL_NODES = NUM_STORIES * NODES_PER_STORY
MODEL_NAME = "gemini-2.5-flash"
BRANCHING_FACTOR = 4
CONTEXT_BUFFER_SIZE = 5
TIMING_INTERVAL = 50
CHECKPOINT_INTERVAL = 100
RECENT_LINK_WINDOW_MULTIPLIER = 2
RECENT_LINK_CANDIDATES = CONTEXT_BUFFER_SIZE * RECENT_LINK_WINDOW_MULTIPLIER

EMBEDDING_MODEL = "text-embedding-004"
VECTOR_DIMENSION = 768
NEGATIVE_SAMPLE_RATIO = 2.0
NEGATIVE_SAMPLE_ATTEMPTS = 5
EMBEDDING_CHUNK_SIZE = 100

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data"
RAW_GRAPH_FILENAME_BASE = "memory_graph_raw.json"
EMBEDDED_GRAPH_FILENAME = "memory_graph_embedded.json"
FINAL_DATASET_FILENAME = "edge_classification_dataset.csv"


STORY_PROMPTS = [
    {
        "story_id": "A",
        "persona_name": "Alex",
        "core_theme": "Navigating a career promotion that requires relocating away from a best friend.",
        "start_time": "2025-01-10T15:00:00Z",
        "seed_event": (
            "I just received a promotion offer, but it requires relocating away from my best friend. "
            "I feel a mix of elation about the career step and deep sadness about the personal loss."
        ),
        "seed_tags": ["career achievement", "personal sacrifice", "major decision", "relocation"],
        "seed_emotion": "Conflict (Joy & Sadness)",
    },
    {
        "story_id": "B",
        "persona_name": "Maya",
        "core_theme": (
            "Starting a new, highly-competitive PhD program while struggling with imposter syndrome "
            "and a demanding research supervisor."
        ),
        "start_time": "2024-09-01T09:30:00Z",
        "seed_event": (
            "The official welcome to the PhD program felt overwhelming. I'm excited by the research, "
            "but I constantly doubt if I'm smart enough to be here. My supervisor's first email was very curt."
        ),
        "seed_tags": ["academic pressure", "imposter syndrome", "new environment", "self-doubt"],
        "seed_emotion": "Anxiety",
    },
    {
        "story_id": "C",
        "persona_name": "Liam",
        "core_theme": (
            "Maintaining a long-distance relationship with a partner in a different time zone and planning "
            "a major move to close the distance."
        ),
        "start_time": "2026-03-20T11:00:00Z",
        "seed_event": (
            "Had a late-night video call with my partner, celebrating a minor visa approval. It reminds me how "
            "hard the distance is, but how worth it the future planning feels. We're setting a date for me to visit."
        ),
        "seed_tags": ["long-distance romance", "future planning", "visa milestone", "communication"],
        "seed_emotion": "Hopeful",
    },
    {
        "story_id": "D",
        "persona_name": "Sarah",
        "core_theme": (
            "Caring for an aging parent while balancing a full-time, emotionally draining job in nursing, leading to "
            "burnout and strain on her romantic relationship."
        ),
        "start_time": "2025-05-15T18:00:00Z",
        "seed_event": (
            "I had to rush my parent to the ER again, which made me late for work. My partner was frustrated that our "
            "dinner plans were ruined. I feel like I'm failing everyone."
        ),
        "seed_tags": ["caregiving", "burnout", "relationship strain", "guilt"],
        "seed_emotion": "Exhaustion",
    },
    {
        "story_id": "E",
        "persona_name": "Ben",
        "core_theme": (
            "Training for a marathon while managing a major home renovation that keeps hitting unexpected structural "
            "problems and delaying their move-in date."
        ),
        "start_time": "2024-11-25T07:00:00Z",
        "seed_event": (
            "Completed my longest run yet—20 miles! Feeling strong physically, but the contractor just sent a photo of "
            "a termite problem in the attic, setting the renovation back by a month."
        ),
        "seed_tags": ["fitness goal", "home renovation", "setback", "physical challenge"],
        "seed_emotion": "Frustration (physical peak, logistical low)",
    },
]


class MemoryNode(BaseModel):
    """Schema for a single event/memory node in the graph, now with story_id."""

    event_id: int = Field(description="Sequential integer ID within its story (0-99).")
    global_id: int = Field(description="Sequential integer ID across the entire graph (0-499).")
    story_id: str = Field(description="Story identifier (A, B, C, D, or E).")
    timestamp: str = Field(description="ISO 8601 UTC timestamp.")
    event_text: str = Field(description="Natural language description of the event.")
    semantic_tags: List[str] = Field(description="2-4 semantic keywords describing the event.")
    emotional_state: str = Field(description="Dominant emotional state.")
    semantic_vec: List[float] = Field(description="Semantic embedding vector.")
    emotional_vec: List[float] = Field(description="Emotional embedding vector.")

    id: int = Field(alias="id", default=None, description="Duplicate of global_id for graph tools.")

    def __init__(self, **data: Any):  # type: ignore[override]
        super().__init__(**data)
        if "id" not in data or data["id"] is None:
            self.id = self.global_id


client = None  # will be initialized from GEMINI_API_KEY
OUTPUT_DIR = DEFAULT_OUTPUT_DIR


def get_gemini_client() -> genai.Client:
    global client
    if client is not None:
        return client
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
    client = genai.Client(api_key=api_key)
    return client


# ---------------------------------------------------------------------------
# Utility helpers shared across stages
# ---------------------------------------------------------------------------


def adaptive_wait(base_wait: float = 0.5, max_rand: float = 1.0) -> None:
    """Small randomized delay to soften rate limits."""

    delay = base_wait + random.random() * max_rand
    time.sleep(delay)


def load_partial_graph(filename: str) -> Tuple[nx.DiGraph, int, Dict[str, int]]:
    """Loads existing graph data to resume generation (optional checkpointing)."""

    graph_path = OUTPUT_DIR / filename
    story_completion = {s["story_id"]: 0 for s in STORY_PROMPTS}

    if graph_path.exists():
        try:
            with open(graph_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            G = nx.node_link_graph(data)

            if G.nodes:
                for node_data in G.nodes.values():
                    story_id = node_data.get("story_id")
                    event_id = node_data.get("event_id")
                    if story_id and event_id is not None:
                        story_completion[story_id] = max(story_completion.get(story_id, 0), event_id)

            global_event_id = max(G.nodes) + 1 if G.nodes else 0

            print(f"\n--- RESUMING GENERATION (FILE: {filename}) ---")
            print(f"Loaded {G.number_of_nodes()} existing nodes. Starting Global ID at {global_event_id}.")
            print(f"Story Progress (highest local event_id): {story_completion}")
            return G, global_event_id, story_completion

        except json.JSONDecodeError:
            print("WARNING: Existing graph file corrupted. Restarting from scratch.")
            return nx.DiGraph(), 0, story_completion

    return nx.DiGraph(), 0, story_completion


def serialize_data(graph: nx.DiGraph, query_pool: List[str]) -> None:
    """Saves the RAW graph and optional query pool under OUTPUT_DIR."""

    graph_data = nx.node_link_data(graph)
    graph_path = OUTPUT_DIR / RAW_GRAPH_FILENAME_BASE
    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=4)

    if query_pool:
        query_path = OUTPUT_DIR / "02_DIVERSE_training_queries_5x100_FINAL.json"
        with open(query_path, "w", encoding="utf-8") as f:
            json.dump(query_pool, f, indent=4)
        print(f"Query pool saved to: {query_path}")

    print(f"RAW Graph data (NO Embeddings) saved to: {graph_path}")


def generate_complex_queries(count: int) -> List[str]:
    return [
        f"Query {i} about semantic conflicts and emotional states across multiple stories."
        for i in range(count)
    ]


def generate_contextual_node(
    global_id: int,
    local_event_id: int,
    story_config: Dict[str, Any],
    context_buffer: List[MemoryNode],
) -> Tuple[Union[MemoryNode, None], float]:
    """Generates a new MemoryNode for a specific story using Gemini."""

    client_local = get_gemini_client()

    story_id = story_config["story_id"]
    persona_name = story_config["persona_name"]
    last_timestamp_str = context_buffer[-1].timestamp if context_buffer else story_config["start_time"]

    context_history = "\n".join(
        [
            f"ID {node.event_id} ({node.timestamp}): {node.event_text} [Emotion: {node.emotional_state}]"
            for node in context_buffer
        ]
    )

    system_prompt = f"""
You are an event generator for a complex graph-based memory system.
Your task is to generate the next chronological event (Local Event ID {local_event_id}) in the life of the user, '{persona_name}'.

# Persona and Narrative (Story {story_id}):
The core theme is: '{story_config["core_theme"]}'. The events must stay focused on this core theme, evolving realistically, and spanning a wide time range over 100 events.

# Strict Output Requirements:
1. Format: Output must be a single, VALID JSON object adhering to the Pydantic schema.
2. Chronology: The 'timestamp' MUST be a valid ISO 8601 UTC string and MUST be chronologically later than the last event's timestamp: {last_timestamp_str}. Advance the time realistically, ensuring temporal diversity (both minutes and months).
3. Identifiers: 'event_id' must be {local_event_id}. 'global_id' must be {global_id}. 'story_id' must be '{story_id}'.
4. Placeholders: 'semantic_vec' and 'emotional_vec' must each be an empty list: [].

# Context History (Last {len(context_buffer)} Events for {persona_name}):
{context_history}
"""

    user_prompt = (
        f"Given the history for {persona_name} (Story {story_id}), generate Event ID {local_event_id}. "
        f"The previous timestamp was {last_timestamp_str}. Advance the story related to the main theme."
    )

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        response_mime_type="application/json",
        response_schema=MemoryNode.model_json_schema(),
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            start_api_call = time.time()
            response = client_local.models.generate_content(
                model=MODEL_NAME,
                contents=user_prompt,
                config=config,
            )
            elapsed_time = time.time() - start_api_call

            raw_json = response.text
            node_dict = json.loads(raw_json)
            new_node = MemoryNode(**node_dict)

            last_dt = datetime.fromisoformat(last_timestamp_str.replace("Z", "+00:00"))
            new_dt = datetime.fromisoformat(new_node.timestamp.replace("Z", "+00:00"))
            if new_dt <= last_dt:
                raise ValueError("Chronology Error: new timestamp is not later than previous.")

            if (
                new_node.event_id != local_event_id
                or new_node.global_id != global_id
                or new_node.story_id != story_id
            ):
                raise ValueError("ID/Story mismatch in generated node.")

            adaptive_wait()
            return new_node, elapsed_time

        except (ValidationError, ValueError, json.JSONDecodeError) as e:
            print(
                f"Attempt {attempt + 1}/{max_retries} failed for Story {story_id}, "
                f"ID {local_event_id}. Error: {type(e).__name__} - {e}"
            )
            if attempt == max_retries - 1:
                return None, 0.0

        except Exception as e:  # API/Network errors
            sleep_time = (2**attempt) + random.uniform(0, 1)
            if attempt < max_retries - 1:
                print(
                    f"API Error for Story {story_id}, ID {local_event_id}. "
                    f"Retrying in {sleep_time:.2f}s... ({e})"
                )
                time.sleep(sleep_time)
            else:
                print(
                    f"FINAL API FAILURE for Story {story_id}, ID {local_event_id}. "
                    f"Giving up after {max_retries} attempts."
                )
                return None, 0.0


def generate_and_build_full_graph() -> nx.DiGraph:
    """Generates all nodes and links across all stories with optional checkpointing."""

    G, current_global_id, story_progress = load_partial_graph(RAW_GRAPH_FILENAME_BASE)
    llm_call_times: List[float] = []

    print(f"\n--- Starting LLM Generation of {TOTAL_NODES} Nodes (5 stories) ---")
    start_llm_time = time.time()

    for story_config in STORY_PROMPTS:
        story_id = story_config["story_id"]
        persona_name = story_config["persona_name"]

        start_local_id = story_progress.get(story_id, 0)
        if start_local_id >= NODES_PER_STORY:
            print(
                f"\n--- SKIP: Story {story_id} ({persona_name}) is already complete "
                f"({NODES_PER_STORY} nodes)."
            )
            continue

        print(
            f"\n### BEGIN STORY {story_id}: {persona_name} "
            f"(Starting at Local ID {start_local_id}, Global ID {current_global_id}) ###"
        )

        context_buffer: List[MemoryNode] = []

        if start_local_id == 0:
            print(f" -> Inserting initial narrative event (Node 0) for {persona_name}...")
            initial_event = MemoryNode(
                event_id=0,
                global_id=current_global_id,
                story_id=story_id,
                timestamp=story_config["start_time"],
                event_text=story_config["seed_event"],
                semantic_tags=story_config["seed_tags"],
                emotional_state=story_config["seed_emotion"],
                semantic_vec=[],
                emotional_vec=[],
            )
            G.add_node(initial_event.global_id, **initial_event.model_dump())
            context_buffer.append(initial_event)
            current_global_id += 1
            start_local_id = 1
        else:
            current_story_nodes = [
                data for _, data in G.nodes(data=True) if data.get("story_id") == story_id
            ]
            current_story_nodes.sort(key=lambda x: x.get("global_id", 0))
            context_nodes = current_story_nodes[-CONTEXT_BUFFER_SIZE:]
            context_buffer.extend([MemoryNode(**data) for data in context_nodes])
            current_global_id = max(G.nodes) + 1 if G.nodes else 0

        for local_event_id in range(start_local_id, NODES_PER_STORY):
            node_data, node_time = generate_contextual_node(
                global_id=current_global_id,
                local_event_id=local_event_id,
                story_config=story_config,
                context_buffer=context_buffer,
            )

            if node_data:
                llm_call_times.append(node_time)
                G.add_node(node_data.global_id, **node_data.model_dump())

                num_links = random.randint(1, BRANCHING_FACTOR)
                target_candidates = [
                    data["global_id"]
                    for _, data in G.nodes(data=True)
                    if data.get("story_id") == story_id and data.get("event_id") < local_event_id
                ]

                if len(target_candidates) > RECENT_LINK_CANDIDATES:
                    source_candidates = target_candidates[-RECENT_LINK_CANDIDATES:]
                else:
                    source_candidates = target_candidates

                if source_candidates:
                    source_nodes = random.sample(
                        source_candidates, min(num_links, len(source_candidates))
                    )
                    for source_global_id in source_nodes:
                        G.add_edge(
                            source_global_id,
                            node_data.global_id,
                            weight=random.random(),
                        )

                context_buffer.append(node_data)
                context_buffer = context_buffer[-CONTEXT_BUFFER_SIZE:]
                current_global_id += 1
            else:
                print(
                    f"Skipping Node ID {local_event_id} due to LLM failure. "
                    f"Stopping generation for Story {story_id}."
                )
                break

            if llm_call_times and current_global_id % TIMING_INTERVAL == 0:
                if len(llm_call_times) >= TIMING_INTERVAL:
                    avg_time_per_node = sum(llm_call_times[-TIMING_INTERVAL:]) / TIMING_INTERVAL
                else:
                    avg_time_per_node = sum(llm_call_times) / len(llm_call_times)

                nodes_remaining = TOTAL_NODES - current_global_id
                time_remaining_minutes = (nodes_remaining * avg_time_per_node) / 60
                print(
                    f"    Generated {G.number_of_nodes()}/{TOTAL_NODES} nodes. (Story {story_id})"
                )
                print(
                    f"    Avg time per node (last {TIMING_INTERVAL}): {avg_time_per_node:.2f}s"
                )
                print(f"    Est. Time Remaining: {time_remaining_minutes:.1f} min")

            if current_global_id > 0 and (
                current_global_id % CHECKPOINT_INTERVAL == 0
                or current_global_id == TOTAL_NODES
            ):
                serialize_data(G, [])
                print(
                    f"\n    CHECKPOINT: Graph saved at Global Node {current_global_id - 1}."
                )

    end_llm_time = time.time()
    print(
        f"\nLLM Generation Complete. Total Nodes: {G.number_of_nodes()}/{TOTAL_NODES}. "
        f"Time: {end_llm_time - start_llm_time:.2f}s"
    )

    return G


# ---------------------------------------------------------------------------
# Phase 2: Embeddings and dataset creation (edge classification)
# ---------------------------------------------------------------------------


def load_graph(path: Union[str, Path]) -> nx.DiGraph:
    """Load a graph from a path or filename.

    If a full/relative filesystem path is given and exists, it is used as-is.
    Otherwise the value is treated as a filename inside OUTPUT_DIR.
    """

    graph_path = Path(path)
    if not graph_path.is_absolute() and not graph_path.exists():
        graph_path = OUTPUT_DIR / graph_path

    if not graph_path.exists():
        raise FileNotFoundError(f"Input graph file not found at: {graph_path}")

    with open(graph_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    try:
        node_count = len(data["nodes"])
    except KeyError:
        node_count = len(data.get("node-data", []))
    print(f"Loaded graph from {filename}. Total nodes: {node_count}.")
    return nx.node_link_graph(data)


def save_graph(graph: nx.DiGraph, filename: str) -> None:
    graph_data = nx.node_link_data(graph)
    graph_path = OUTPUT_DIR / filename
    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=4)
    print(f"Graph saved to: {graph_path}")


def save_dataframe(df: pd.DataFrame, filename: str) -> None:
    df_path = OUTPUT_DIR / filename
    df.to_csv(df_path, index=False)
    print(f"Final dataset saved to: {df_path}")


def calculate_temporal_closeness(timestamp_source: str, timestamp_target: str) -> float:
    dt_source = datetime.fromisoformat(timestamp_source.replace("Z", "+00:00"))
    dt_target = datetime.fromisoformat(timestamp_target.replace("Z", "+00:00"))
    time_diff_seconds = abs((dt_target - dt_source).total_seconds())
    return 1.0 / (time_diff_seconds + 1.0)


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    a = np.array(vec_a)
    b = np.array(vec_b)
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def calculate_feature_vector(
    node_a_data: Dict[str, Any], node_b_data: Dict[str, Any]
) -> Dict[str, float]:
    time_closeness = calculate_temporal_closeness(
        node_a_data["timestamp"], node_b_data["timestamp"]
    )
    semantic_similarity = cosine_similarity(
        node_a_data["semantic_vec"], node_b_data["semantic_vec"]
    )
    emotional_alignment = cosine_similarity(
        node_a_data["emotional_vec"], node_b_data["emotional_vec"]
    )
    return {
        "Time_Closeness": time_closeness,
        "Semantic_Similarity": semantic_similarity,
        "Emotional_Alignment": emotional_alignment,
    }


def chunk_list(lst: List[Any], n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def generate_embeddings(G: nx.DiGraph) -> nx.DiGraph:
    """Generate semantic and emotional embeddings for each node using Gemini embeddings."""

    client_local = get_gemini_client()

    batch_input_texts: List[str] = []
    vector_map: List[Dict[str, Any]] = []

    for node_id, data in G.nodes(data=True):
        if not data.get("semantic_vec") or len(data["semantic_vec"]) != VECTOR_DIMENSION:
            semantic_prompt = f"Event: {data['event_text']}. Tags: {', '.join(data['semantic_tags'])}"
            batch_input_texts.append(semantic_prompt)
            vector_map.append({"node_id": node_id, "type": "semantic"})

            emotional_prompt = data["emotional_state"]
            batch_input_texts.append(emotional_prompt)
            vector_map.append({"node_id": node_id, "type": "emotional"})

    if not batch_input_texts:
        print("Node enrichment skipped: all nodes already contain embeddings.")
        return G

    total_inputs = len(batch_input_texts)
    num_batches = int(math.ceil(total_inputs / EMBEDDING_CHUNK_SIZE))
    print(
        f"\n--- Starting micro-batch embedding generation for {total_inputs} "
        f"vectors ({total_inputs/2:.0f} nodes) ---"
    )
    start_time = time.time()

    all_embeddings: List[np.ndarray] = []

    for i, chunk in enumerate(chunk_list(batch_input_texts, EMBEDDING_CHUNK_SIZE)):
        print(f"Processing micro-batch {i + 1} of {num_batches} (size={len(chunk)})...")
        try:
            batch_result = client_local.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=chunk,
                config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
            )
            all_embeddings.extend([np.array(e.values) for e in batch_result.embeddings])
        except Exception as e:
            print(f"FATAL MICRO-BATCH ERROR in chunk {i + 1}: {e}. Stopping.")
            return G

    if len(all_embeddings) != total_inputs:
        raise RuntimeError(
            f"Total embeddings returned ({len(all_embeddings)}) does not match expected ({total_inputs})."
        )

    for i, vector in enumerate(all_embeddings):
        map_entry = vector_map[i]
        node_id = map_entry["node_id"]
        vec_type = map_entry["type"]
        vector_list = vector.tolist()
        if vec_type == "semantic":
            G.nodes[node_id]["semantic_vec"] = vector_list
        elif vec_type == "emotional":
            G.nodes[node_id]["emotional_vec"] = vector_list

    elapsed = time.time() - start_time
    print(
        f"\nAll {total_inputs/2:.0f} nodes enriched across {num_batches} batches in {elapsed:.2f}s. "
        f"Avg per node: {elapsed / (total_inputs/2):.2f}s"
    )
    return G


def create_training_dataset(G: nx.DiGraph, ratio: float) -> pd.DataFrame:
    """Create final edge classification dataset (Linked=1 and Linked=0 samples)."""

    all_samples: List[Dict[str, Any]] = []
    story_nodes: Dict[str, List[int]] = {}

    for node_id, data in G.nodes(data=True):
        story_id = data.get("story_id")
        if story_id:
            story_nodes.setdefault(story_id, []).append(node_id)

    print(
        f"\n--- Starting sample generation (ratio Linked=0/Linked=1 = {ratio:.1f}) ---"
    )

    positive_count = 0
    for source_id, target_id in G.edges():
        source_data = G.nodes[source_id]
        target_data = G.nodes[target_id]
        if source_data["story_id"] != target_data["story_id"]:
            continue
        features = calculate_feature_vector(source_data, target_data)
        features.update(
            {
                "Linked": 1,
                "story_id": source_data["story_id"],
                "Source_ID": source_id,
                "Target_ID": target_id,
            }
        )
        all_samples.append(features)
        positive_count += 1

    print(f"Generated {positive_count} positive samples (Linked=1).")

    target_negative_count = int(positive_count * ratio)
    negative_count = 0
    print(f"Target negative samples (Linked=0): {target_negative_count}")

    while negative_count < target_negative_count:
        story_id = random.choice(list(story_nodes.keys()))
        current_story_nodes = story_nodes[story_id]
        if len(current_story_nodes) < 2:
            continue

        for _ in range(NEGATIVE_SAMPLE_ATTEMPTS):
            source_id, target_id = random.sample(current_story_nodes, 2)
            if G.nodes[source_id]["global_id"] >= G.nodes[target_id]["global_id"]:
                source_id, target_id = target_id, source_id
            if not G.has_edge(source_id, target_id):
                source_data = G.nodes[source_id]
                target_data = G.nodes[target_id]
                features = calculate_feature_vector(source_data, target_data)
                features.update(
                    {
                        "Linked": 0,
                        "story_id": story_id,
                        "Source_ID": source_id,
                        "Target_ID": target_id,
                    }
                )
                all_samples.append(features)
                negative_count += 1
                if negative_count % 500 == 0:
                    print(
                        f"   -> Generated {negative_count}/{target_negative_count} negative samples..."
                    )
                break

    print(f"Final negative samples generated: {negative_count}")
    print(f"TOTAL samples generated: {positive_count + negative_count}")

    df = pd.DataFrame(all_samples)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    global OUTPUT_DIR

    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the full dataset pipeline: generate memory graph, "
            "embeddings, and edge-classification CSV."
        )
    )
    parser.add_argument(
        "--stage",
        choices=["graph", "embed", "csv", "all"],
        default="all",
        help="Which stage of the pipeline to run.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where graph and dataset files are stored (default: ./data)",
    )
    parser.add_argument(
        "--graph-in",
        default=None,
        help=(
            "Input graph JSON file. For --stage embed, defaults to memory_graph_raw.json "
            "under --output-dir. For --stage csv, defaults to memory_graph_embedded.json "
            "under --output-dir."
        ),
    )
    parser.add_argument(
        "--csv-out",
        default=None,
        help=(
            "Output CSV file for edge classification (default: edge_classification_dataset.csv "
            "under --output-dir)."
        ),
    )
    args = parser.parse_args()

    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    default_raw_graph = RAW_GRAPH_FILENAME_BASE
    default_embedded_graph = EMBEDDED_GRAPH_FILENAME
    default_csv = FINAL_DATASET_FILENAME

    if args.stage in {"graph", "all"}:
        print("=== Stage 1: Generating memory graph ===")
        G = generate_and_build_full_graph()
        query_pool = generate_complex_queries(TOTAL_NODES)
        serialize_data(G, query_pool)
    else:
        G = None

    if args.stage in {"embed", "all"}:
        print("=== Stage 2: Generating embeddings ===")
        if G is None:
            graph_in = args.graph_in or default_raw_graph
            G = load_graph(graph_in)
        G_embedded = generate_embeddings(G)
        save_graph(G_embedded, EMBEDDED_GRAPH_FILENAME)
    else:
        G_embedded = None

    if args.stage in {"csv", "all"}:
        print("=== Stage 3: Creating edge classification CSV ===")
        if G_embedded is None:
            graph_in = args.graph_in or default_embedded_graph
            G_embedded = load_graph(graph_in)
        final_df = create_training_dataset(G_embedded, NEGATIVE_SAMPLE_RATIO)
        csv_out = args.csv_out or default_csv
        save_dataframe(final_df, csv_out)
        print("Dataset creation pipeline complete.")


if __name__ == "__main__":
    main()
