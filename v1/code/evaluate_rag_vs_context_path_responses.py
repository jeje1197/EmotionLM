#!/usr/bin/env python3
"""LLM-as-a-judge evaluation for RAG vs Context Path Retrieval (v1 experiment).

For each query in a JSON file, this script:
- Restricts retrieval to memories from the same story.
- Builds two contexts:
    1) RAG: top-k memories by embedding similarity.
    2) Context Path: a local path of connected memories around the most relevant node.
- Generates two answers using the same LLM (Gemini) but different contexts.
- Saves a responses CSV and then runs the LLM-as-a-judge pipeline to score
    both methods according to the rubric in llm_as_a_judge_config.json.
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import chromadb  # type: ignore
from chromadb.config import Settings  # type: ignore
import networkx as nx
import numpy as np
import pandas as pd
from google import genai  # type: ignore
from google.genai import types  # type: ignore
from google.genai import errors as genai_errors  # type: ignore

from generate_dataset import (  # type: ignore
    EMBEDDING_MODEL,
    MODEL_NAME,
    VECTOR_DIMENSION,
    get_gemini_client,
)


V1_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRAPH_PATH = V1_ROOT / "data" / "memory_graph_embedded.json"
DEFAULT_QUERIES_PATH = V1_ROOT / "data" / "queries.json"
DEFAULT_OUTPUT_CSV = V1_ROOT / "artifacts" / "predictions" / "rag_vs_context_path_responses.csv"


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _call_gemini_with_backoff(
    client: genai.Client,
    *,
    model: str,
    contents: str,
    config: types.GenerateContentConfig | None = None,
    max_retries: int = 5,
    base_wait_seconds: float = 30.0,
):
    """Call Gemini with simple backoff on transient quota/availability errors.

    Behavior is the same as a direct generate_content call, except that
    certain 4xx/5xx errors (quota exceeded, model overloaded) trigger a
    short sleep and retry instead of failing immediately.
    """

    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except genai_errors.APIError as e:  # type: ignore[reportAttributeAccessIssue]
            message = str(e)

            # If this looks like a hard daily free-tier limit, don't keep retrying.
            if "GenerateRequestsPerDayPerProjectPerModel-FreeTier" in message:
                raise RuntimeError(
                    "Gemini free-tier daily request limit reached for this model. "
                    "Wait until the quota resets or switch to a paid tier / different model."
                ) from e

            transient = (
                "RESOURCE_EXHAUSTED" in message
                or "quota" in message.lower()
                or "UNAVAILABLE" in message
                or "overloaded" in message.lower()
            )
            if not transient or attempt == max_retries - 1:
                raise

            wait = base_wait_seconds
            print(
                f"[Gemini] Transient error (attempt {attempt + 1}/{max_retries}): {message}. "
                f"Sleeping for {wait:.0f}s before retrying..."
            )
            time.sleep(wait)
            continue


def build_judge_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
    return genai.Client(api_key=api_key)


def make_judge_prompt(example: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    rubric = cfg["rubric"]
    template = cfg["prompt_template"]
    context = example.get("context", "")
    reference = example.get("reference", "")
    prediction = example.get("prediction", "")

     # Build a human-readable list of dimensions and a strict key list.
    dim_entries = rubric.get("dimensions", [])
    dimensions_lines: List[str] = []
    dimension_names: List[str] = []
    for d in dim_entries:
        name = str(d.get("name"))
        desc = str(d.get("description", ""))
        dimension_names.append(name)
        dimensions_lines.append(f"- {name}: {desc}")

    dimensions_block = "\n".join(dimensions_lines)
    dimension_keys = ", ".join(dimension_names)

    return template.format(
        scale_min=rubric["scale_min"],
        scale_max=rubric["scale_max"],
        dimensions=dimensions_block,
        dimension_keys=dimension_keys,
        context=context,
        reference=reference,
        prediction=prediction,
    )


def score_example(client: genai.Client, example: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    model_name = cfg.get("model", "gemini-2.5-flash")
    temperature = float(cfg.get("temperature", 0.2))

    system_instruction = cfg.get(
        "system_instruction", "You are an impartial, consistent evaluator of model outputs."
    )
    prompt = make_judge_prompt(example, cfg)

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json",
        temperature=temperature,
    )
    response = _call_gemini_with_backoff(
        client,
        model=model_name,
        contents=prompt,
        config=config,
    )

    try:
        scores = json.loads(response.text)
    except Exception:
        scores = {"raw_response": response.text}

    return scores


def load_graph(path: Path) -> nx.DiGraph:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return nx.node_link_graph(data)


def load_queries(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        queries = json.load(f)
    if not isinstance(queries, list):
        raise ValueError("Queries JSON must be a list of objects.")
    return queries


def build_story_index(G: nx.DiGraph) -> Dict[str, Dict[str, Any]]:
    """Build a Chroma collection per story over semantic_vec embeddings."""

    client = chromadb.Client(Settings(anonymized_telemetry=False))
    story_index: Dict[str, Dict[str, Any]] = {}

    for node_id, data in G.nodes(data=True):
        story_id = data.get("story_id")
        if not story_id:
            continue
        vec = np.array(data.get("semantic_vec", []), dtype=float)
        if vec.size != VECTOR_DIMENSION:
            continue

        entry = story_index.get(story_id)
        if entry is None:
            collection = client.create_collection(name=f"story_{story_id}")
            entry = {"collection": collection}
            story_index[story_id] = entry

        collection = entry["collection"]
        collection.add(
            ids=[str(node_id)],
            embeddings=[vec.astype("float32").tolist()],
            metadatas=[{"story_id": story_id}],
        )

    return story_index


def embed_texts(texts: List[str]) -> np.ndarray:
    client = get_gemini_client()
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
    )
    return np.stack([np.array(e.values, dtype=float) for e in result.embeddings], axis=0)


def build_context_from_nodes(G: nx.DiGraph, node_ids: List[int]) -> str:
    if not node_ids:
        return ""

    nodes_with_time: List[Tuple[int, str]] = []
    for nid in node_ids:
        data = G.nodes[nid]
        ts = data.get("timestamp", "")
        nodes_with_time.append((nid, ts))

    nodes_with_time.sort(key=lambda x: x[1])

    parts: List[str] = []
    for nid, ts in nodes_with_time:
        data = G.nodes[nid]
        text = data.get("event_text", "")
        # Do not surface explicit emotion labels; rely on narrative context only.
        parts.append(f"[{ts}] {text}")

    return "\n".join(parts)


def retrieve_rag_context(
    G: nx.DiGraph,
    story_index: Dict[str, Dict[str, Any]],
    story_id: str,
    query_text: str,
    top_k: int = 5,
) -> Tuple[str, List[int]]:
    entry = story_index.get(story_id)
    if not entry:
        return "", []
    collection = entry["collection"]

    query_vec = embed_texts([query_text])[0].astype("float32").tolist()
    results = collection.query(query_embeddings=[query_vec], n_results=top_k)
    ids = results.get("ids", [[]])[0]
    selected_node_ids = [int(i) for i in ids]

    context = build_context_from_nodes(G, selected_node_ids)
    return context, selected_node_ids


def retrieve_context_path(
    G: nx.DiGraph,
    story_index: Dict[str, Dict[str, Any]],
    story_id: str,
    query_text: str,
    max_nodes: int = 5,
    seed_top_k: int = 1,
    max_depth: int | None = None,
) -> Tuple[str, List[int]]:
    """Context Path Traversal: nearest neighbor + bounded BFS over the graph.

    - Seed selection: top-k semantic neighbors (within the same story) from Chroma.
    - Traversal: BFS from all seeds over the memory graph, treating edges as
      undirected, restricted to the same story, until either max_nodes is reached
      or max_depth (if set) is exceeded.
    """

    entry = story_index.get(story_id)
    if not entry:
        return "", []
    collection = entry["collection"]

    query_vec = embed_texts([query_text])[0].astype("float32").tolist()
    results = collection.query(query_embeddings=[query_vec], n_results=max(seed_top_k, 1))
    ids = results.get("ids", [[]])[0]
    if not ids:
        return "", []

    # Initialize BFS from up to seed_top_k seeds.
    visited: set[int] = set()
    queue: List[tuple[int, int]] = []  # (node_id, depth)

    for raw_id in ids[:seed_top_k]:
        nid = int(raw_id)
        if G.nodes[nid].get("story_id") != story_id:
            continue
        if nid in visited:
            continue
        visited.add(nid)
        queue.append((nid, 0))
        if len(visited) >= max_nodes:
            break

    while queue and len(visited) < max_nodes:
        current, depth = queue.pop(0)
        if max_depth is not None and depth >= max_depth:
            continue

        neighbors = list(G.successors(current)) + list(G.predecessors(current))
        for nbr in neighbors:
            if len(visited) >= max_nodes:
                break
            if nbr in visited:
                continue
            if G.nodes[nbr].get("story_id") != story_id:
                continue
            visited.add(nbr)
            queue.append((nbr, depth + 1))

    node_ids = list(visited)
    context = build_context_from_nodes(G, node_ids)
    return context, node_ids


def generate_answer(client: genai.Client, query_text: str, context: str, method: str) -> str:
    if not context:
        context = "(No relevant memories retrieved for this query.)"

    system_instruction = (
        "You are an emotionally aware assistant that answers questions about a user's life "
        "based on a set of memory snippets. Be faithful to the provided context and do not "
        "invent detailed events that are not supported by it."
    )

    prompt = (
        f"Retrieval method: {method}.\n\n"
        f"Context memories:\n{context}\n\n"
        f"User query:\n{query_text}\n\n"
        "Answer the user's query using the context above. If the answer is not supported by "
        "the context, clearly say that you cannot answer based on the available information."
    )

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
    )
    response = _call_gemini_with_backoff(
        client,
        model=MODEL_NAME,
        contents=prompt,
        config=config,
    )

    return response.text


def run_experiment(
    graph_path: Path,
    queries_path: Path,
    output_csv: Path,
    top_k_rag: int = 5,
    max_nodes_path: int = 5,
    seed_top_k: int = 1,
    max_depth: int | None = None,
) -> None:
    G = load_graph(graph_path)
    queries = load_queries(queries_path)
    story_index = build_story_index(G)

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    client = get_gemini_client()

    print(f"Loaded graph from {graph_path} with {G.number_of_nodes()} nodes.")
    print(f"Loaded {len(queries)} queries from {queries_path}.")
    print(
        f"Running generation with top_k_rag={top_k_rag}, max_nodes_path={max_nodes_path}, "
        f"seed_top_k={seed_top_k}, max_depth={max_depth}."
    )

    rows: List[Dict[str, Any]] = []
    for idx, q in enumerate(queries, start=1):
        qid = q.get("id")
        story_id = q.get("story_id")
        query_text = q.get("query")
        if not qid or not story_id or not query_text:
            continue

        print(f"[Gen {idx}/{len(queries)}] Query {qid} (story {story_id}) - building contexts...")

        rag_context, rag_nodes = retrieve_rag_context(G, story_index, story_id, query_text, top_k=top_k_rag)
        path_context, path_nodes = retrieve_context_path(
            G,
            story_index,
            story_id,
            query_text,
            max_nodes=max_nodes_path,
            seed_top_k=seed_top_k,
            max_depth=max_depth,
        )

        print(
            f"[Gen {idx}/{len(queries)}] Query {qid} - generating answers (RAG & Context Path)..."
        )

        rag_answer = generate_answer(client, query_text, rag_context, method="rag")
        path_answer = generate_answer(client, query_text, path_context, method="context_path")

        rows.append(
            {
                "id": f"{qid}_rag",
                "base_id": qid,
                "story_id": story_id,
                "method": "rag",
                "query": query_text,
                "context": rag_context,
                "prediction": rag_answer,
                "reference": "",
            }
        )

        rows.append(
            {
                "id": f"{qid}_context_path",
                "base_id": qid,
                "story_id": story_id,
                "method": "context_path",
                "query": query_text,
                "context": path_context,
                "prediction": path_answer,
                "reference": "",
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}")


def run_judging(cfg_path: Path) -> None:
    cfg = load_config(cfg_path)

    predictions_csv = Path(cfg["predictions_csv"])
    id_col = cfg["id_column"]
    context_col = cfg["context_column"]
    pred_col = cfg["prediction_column"]
    ref_col = cfg["reference_column"]
    output_path = Path(cfg["output_path"])

    df = pd.read_csv(predictions_csv)

    max_examples = int(cfg.get("max_examples", len(df)))
    df = df.head(max_examples)

    client = build_judge_client()

    print(f"Running LLM-as-a-judge on {len(df)} predictions from {predictions_csv}...")

    results: List[Dict[str, Any]] = []
    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        example = {
            "id": row[id_col],
            "context": row.get(context_col, ""),
            "prediction": row[pred_col],
            "reference": row.get(ref_col, ""),
        }
        print(f"[Judge {idx}/{len(df)}] Scoring {example['id']}...")
        scores = score_example(client, example, cfg)
        results.append({"id": example["id"], "scores": scores})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"LLM-as-a-judge results written to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run RAG vs Context Path Retrieval experiment and LLM-as-a-judge "
            "evaluation over the v1 memory graph."
        )
    )
    parser.add_argument(
        "--graph",
        type=Path,
        default=None,
        help="Path to embedded memory graph JSON (overrides config).",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=None,
        help="Path to queries JSON file (overrides config).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write predictions CSV (overrides config).",
    )
    parser.add_argument(
        "--judge-config",
        type=Path,
        default=V1_ROOT / "code" / "llm_as_a_judge_config.json",
        help="Path to LLM-as-a-judge configuration JSON.",
    )
    parser.add_argument(
        "--top-k-rag",
        type=int,
        default=None,
        help="Number of top memories to retrieve for RAG (overrides config).",
    )
    parser.add_argument(
        "--max-nodes-path",
        type=int,
        default=None,
        help="Maximum number of nodes in the context path (overrides config).",
    )
    parser.add_argument(
        "--cpt-top-k-seeds",
        type=int,
        default=None,
        help="Number of semantic seeds for Context Path Traversal (overrides config).",
    )
    parser.add_argument(
        "--cpt-max-depth",
        type=int,
        default=None,
        help="Maximum BFS depth for Context Path Traversal (overrides config).",
    )

    args = parser.parse_args()

    cfg = load_config(args.judge_config)

    graph_path = args.graph or Path(cfg.get("graph_path", DEFAULT_GRAPH_PATH))
    queries_path = args.queries or Path(cfg.get("queries_path", DEFAULT_QUERIES_PATH))
    output_csv = args.output or Path(cfg.get("predictions_csv", DEFAULT_OUTPUT_CSV))

    top_k_rag = args.top_k_rag if args.top_k_rag is not None else int(cfg.get("top_k_rag", 5))
    max_nodes_path = (
        args.max_nodes_path if args.max_nodes_path is not None else int(cfg.get("max_nodes_path", 5))
    )
    seed_top_k = (
        args.cpt_top_k_seeds
        if args.cpt_top_k_seeds is not None
        else int(cfg.get("cpt_top_k_seeds", 1))
    )
    max_depth_cfg = cfg.get("cpt_max_depth")
    if args.cpt_max_depth is not None:
        max_depth: int | None = args.cpt_max_depth
    elif max_depth_cfg is not None:
        max_depth = int(max_depth_cfg)
    else:
        max_depth = None

    run_experiment(
        graph_path=graph_path,
        queries_path=queries_path,
        output_csv=output_csv,
        top_k_rag=top_k_rag,
        max_nodes_path=max_nodes_path,
        seed_top_k=seed_top_k,
        max_depth=max_depth,
    )

    run_judging(args.judge_config)


if __name__ == "__main__":
    main()
