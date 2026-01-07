"""Context Path Traversal implementation.

- Seeds are obtained from a provided vector lookup callable (flexible: text, id, vector).
- A scoring function can be provided to replace the default ALS scoring.
- Returns ranked context paths (list of node dicts + path score).
"""

from typing import Any, Callable, List, Tuple, Sequence, Optional, Dict
from pydantic import BaseModel
from .memory_store import MemoryGraph
from .als import calculate_als_score


class CPTConfig(BaseModel):
    seed_nodes: int = 3
    max_depth: int = 3
    spreading_activation: bool = False
    # Maximum number of neighbor candidates to consider at each hop
    max_neighbors: int = 10
    # Optional minimum ALS/score threshold; neighbors below this are ignored
    min_als_score: Optional[float] = None
    # Cache event data during traversal to reduce redundant lookups
    cache_event_data: bool = True


class ContextPath(BaseModel):
    nodes: List[dict]
    score: float


class CPTResult(BaseModel):
    paths: List[ContextPath]


def execute_context_path_traversal(
    memory_store: MemoryGraph,
    query: Any,
    vector_store_lookup_function: Callable[[Any], Sequence[Tuple[str, float]]],
    cpt_config: CPTConfig = CPTConfig(),
    score_fn: Optional[Callable[[dict, dict], float]] = None,
    event_data_provider: Optional[Callable[[str], dict]] = None,
    logger: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> CPTResult:
    """Perform Context Path Traversal.

    Args:
        memory_store: MemoryGraph instance (stores structure: nodes and edges).
        query: The retrieval query (text, id, or vector) passed to the vector lookup callable.
        vector_store_lookup_function: Callable that accepts `query` and returns an iterable
            of (node_id, score) tuples ordered by relevance.
        cpt_config: CPTConfig controls seed count and depth.
        score_fn: Optional function (event_a, event_b) -> float. Defaults to ALS.
        event_data_provider: Optional callable (event_id) -> dict that fetches event data
            from an external source. If None, reads from memory_store.get_event().

    Returns:
        CPTResult with one greedy ContextPath per seed (subject to constraints).
    """
    if score_fn is None:
        score_fn = calculate_als_score

    # Helper to fetch event data: use provider if given, else read from graph
    event_cache: dict[str, dict] = {} if cpt_config.cache_event_data else None

    def get_event_data(event_id: str) -> dict:
        if event_cache is not None:
            if event_id not in event_cache:
                if event_data_provider is not None:
                    event_cache[event_id] = event_data_provider(event_id)
                else:
                    event_cache[event_id] = memory_store.get_event(event_id)
            return event_cache[event_id]
        # No caching
        if event_data_provider is not None:
            return event_data_provider(event_id)
        return memory_store.get_event(event_id)

    # Obtain seed nodes from vector DB/lookup
    seed_hits = list(vector_store_lookup_function(query))[: cpt_config.seed_nodes]
    seed_scores = {nid: float(score) for nid, score in seed_hits}

    if logger:
        logger({
            "event": "Context Path Retrieval",
            "type": "synchronous",
            "query": str(query),
            "config": {
                "max_depth": cpt_config.max_depth,
                "seed_nodes": cpt_config.seed_nodes,
                "max_neighbors": cpt_config.max_neighbors,
                "min_als_score": cpt_config.min_als_score,
                "cache_event_data": cpt_config.cache_event_data,
            },
            "seed_nodes": [(nid, score) for nid, score in seed_hits],
        })

    paths: List[ContextPath] = []
    globally_visited: set[str] = set()

    # Greedy one-path-per-seed traversal
    for seed_id, seed_score in seed_scores.items():
        if not memory_store.has_node(seed_id):
            continue
        # Skip seeds that were already part of a previous path
        if seed_id in globally_visited:
            continue

        if logger:
            logger({
                "event": "Building Context Path",
                "path_number": len(paths) + 1,
                "seed_node": seed_id,
                "seed_score": seed_score,
            })

        path_ids: List[str] = [seed_id]
        current_id = seed_id
        depth = 0
        acc_score = float(seed_score)
        globally_visited.add(seed_id)

        while depth < cpt_config.max_depth:
            neighbors = list(memory_store.neighbors(current_id))
            candidates: List[Tuple[str, float]] = []

            # Fetch current node data once for this step
            try:
                current_data = get_event_data(current_id)
            except KeyError:
                break

            for nbr in neighbors:
                # avoid cycles within this path
                if nbr in path_ids:
                    continue
                try:
                    nbr_data = get_event_data(nbr)
                except KeyError:
                    continue
                score = float(score_fn(current_data, nbr_data))
                if cpt_config.min_als_score is not None and score < cpt_config.min_als_score:
                    continue
                candidates.append((nbr, score))

            # Sort by score descending and consider at most max_neighbors candidates
            candidates.sort(key=lambda x: x[1], reverse=True)
            top_candidates = candidates[: max(cpt_config.max_neighbors, 1)]

            if logger:
                logger({
                    "event": "Traversal Step",
                    "path_number": len(paths) + 1,
                    "current_node": current_id,
                    "current_path": path_ids,
                    "depth": depth,
                    "computation": "als",
                    "neighbors": [(nid, score) for nid, score in candidates],
                    "top_candidates": [(nid, score) for nid, score in top_candidates],
                })

            if not candidates:
                break

            best_id, best_score = top_candidates[0]

            if logger:
                logger({
                    "event": "Node Selection",
                    "path_number": len(paths) + 1,
                    "selected_nodes": [best_id],
                    "selected_score": best_score,
                })

            path_ids.append(best_id)
            acc_score += best_score
            globally_visited.add(best_id)
            current_id = best_id
            depth += 1

        # Build path object for this seed
        nodes = [get_event_data(nid) for nid in path_ids]
        paths.append(ContextPath(nodes=nodes, score=acc_score))
        
        if logger:
            path_texts = [node.get("text", node.get("content", "<no text>")) for node in nodes]
            logger({
                "event": "Full Context Path",
                "path_number": len(paths),
                "node_ids": path_ids,
                "score": acc_score,
                "context_contributed": path_texts,
            })

    # Sort final paths by score descending
    ranked = sorted(paths, key=lambda p: p.score, reverse=True)
    return CPTResult(paths=ranked)
