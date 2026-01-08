"""Context Path Traversal implementation.

- Seeds are obtained from a provided vector lookup callable (flexible: text, id, vector).
- A scoring function can be provided to replace the default ALS scoring.
- Returns ranked context paths (list of node dicts + path score).
"""

from typing import Any, Callable, List, Tuple, Sequence, Optional, Dict
from pydantic import BaseModel
from .memory_store import MemoryGraph
from .als import calculate_als_score, ALSConfig
import numpy as _np


class CPTConfig(BaseModel):
    seed_nodes: int = 3
    max_depth: int = 3
    spreading_activation: bool = False
    # Maximum number of neighbor candidates to consider at each hop
    max_neighbors: int = 10
    # How many top neighbors to expand per partial path. Default 1 = greedy.
    branching_factor: int = 1
    # Optional minimum ALS/score threshold; neighbors below this are ignored
    min_als_score: Optional[float] = None
    # General top-k for neighbor selection (applied before branching_factor)
    top_k: Optional[int] = None
    # Nucleus sampling (top-p) for neighbors. Keep smallest prefix with cumulative score >= top_p
    top_p: Optional[float] = None
    # If True, sample neighbors stochastically using sampling_temperature
    random_sampling: bool = False
    sampling_temperature: Optional[float] = None
    # Cache event data during traversal to reduce redundant lookups
    cache_event_data: bool = True
    # Spreading activation (semantic-only) options (override general top_k/top_p/sampling)
    activation_top_k: Optional[int] = 5
    activation_top_p: Optional[float] = None
    activation_random: Optional[bool] = None
    activation_sampling_temperature: Optional[float] = None
    # Optional ALSConfig override when scoring activation hits specifically.
    activation_als_config: Optional["ALSConfig"] = None
    # Minimum score returned by the vector lookup to consider activation candidate
    activation_min_score: Optional[float] = None
    # If True, treat activation results as additional neighbors; otherwise
    # keep them separate (but still considered during candidate scoring)
    activation_as_neighbors: bool = True
    
    # Multiplier applied to scores computed for activation-derived candidates
    # Use <1.0 to decay activation scores so fewer pass `min_als_score`.
    activation_score_multiplier: float = 1.0


class ContextPath(BaseModel):
    nodes: List[dict]
    score: float


class CPTResult(BaseModel):
    paths: List[ContextPath]


def apply_topk_topp_sampling(
    candidates: List[Tuple[str, float]],
    top_k: Optional[int],
    top_p: Optional[float],
    random_sampling: bool,
    temperature: Optional[float],
) -> List[Tuple[str, float]]:
    """Apply top-k, top-p, and optional sampling to a list of scored candidates."""
    if not candidates:
        return []
    
    scores = _np.array([s for _, s in candidates], dtype=float)
    
    # Apply top-p (nucleus sampling)
    if top_p is not None:
        if scores.sum() <= 0:
            probs = _np.ones_like(scores) / len(scores)
        else:
            probs = scores / float(scores.sum())
        order = probs.argsort()[::-1]
        cum = 0.0
        selected_idx = []
        for idx in order:
            cum += probs[idx]
            selected_idx.append(int(idx))
            if cum >= float(top_p):
                break
        candidates = [candidates[i] for i in selected_idx]
        scores = _np.array([s for _, s in candidates], dtype=float)
    
    # Apply random sampling with temperature
    if random_sampling and temperature and temperature > 0:
        temp = float(temperature)
        stab = scores - scores.max()
        exp = _np.exp(stab / temp)
        probs = exp / exp.sum()
        rng = _np.random.default_rng()
        k = min(len(candidates), top_k if top_k else len(candidates))
        try:
            chosen = rng.choice(len(candidates), size=k, replace=False, p=probs)
        except Exception:
            chosen = _np.arange(min(k, len(candidates)))
        candidates = [candidates[int(i)] for i in chosen]
    else:
        # Deterministic: sort by score and keep top-k
        candidates.sort(key=lambda x: x[1], reverse=True)
        if top_k is not None:
            candidates = candidates[:top_k]
    
    return candidates


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

    # Helper to call the vector lookup with an optional top_k parameter.
    # Some adapters accept (query, top_k) while simpler ones accept only (query,).
    def call_vector_lookup(q: Any, top_k: int) -> List[Tuple[str, float]]:
        try:
            return list(vector_store_lookup_function(q, top_k))
        except TypeError:
            return list(vector_store_lookup_function(q))

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

        # Branching/traversal logic: either greedy (branching_factor==1)
        # or expand up to `branching_factor` top candidates for each active partial path.
        branching = max(1, int(cpt_config.branching_factor))
        # start with a single active partial path for this seed
        active_partial_paths = [
            {
                "path_ids": [seed_id],
                "current_id": seed_id,
                "acc_score": float(seed_score),
                "visited": {seed_id},
            }
        ]
        # mark the seed as globally visited (prevents reuse as a seed), but allow
        # node reuse across branches for richer exploration
        globally_visited.add(seed_id)

        depth = 0
        while depth < cpt_config.max_depth and active_partial_paths:
            new_active: List[dict] = []
            for part in active_partial_paths:
                path_ids = part["path_ids"]
                current_id = part["current_id"]
                acc_score = part["acc_score"]
                visited = set(part.get("visited", set(path_ids)))

                # Fetch current node data once for this step
                try:
                    current_data = get_event_data(current_id)
                except KeyError:
                    continue

                # compute neighbors and activation hits (same logic as before)
                neighbors = list(memory_store.neighbors(current_id))
                candidates: List[Tuple[str, float]] = []

                activation_hits: List[Tuple[str, float]] = []
                if cpt_config.spreading_activation:
                    current_text = current_data.get("text") or current_data.get("content")
                    if current_text:
                        try:
                            raw_hits = call_vector_lookup(current_text, cpt_config.activation_top_k)
                        except Exception:
                            raw_hits = []

                        hits_list = [(nid, float(s)) for nid, s in raw_hits]
                        if hits_list:
                            hits_list = [(nid, s) for nid, s in hits_list if nid != current_id and nid not in path_ids]
                            if cpt_config.activation_min_score is not None:
                                hits_list = [(nid, s) for nid, s in hits_list if s >= cpt_config.activation_min_score]
                            if hits_list:
                                # Use activation-specific settings or fall back to general
                                act_topk = cpt_config.activation_top_k if cpt_config.activation_top_k is not None else cpt_config.top_k
                                act_topp = cpt_config.activation_top_p if cpt_config.activation_top_p is not None else cpt_config.top_p
                                act_rand = cpt_config.activation_random if cpt_config.activation_random is not None else cpt_config.random_sampling
                                act_temp = cpt_config.activation_sampling_temperature if cpt_config.activation_sampling_temperature is not None else cpt_config.sampling_temperature
                                hits_list = apply_topk_topp_sampling(hits_list, act_topk, act_topp, act_rand, act_temp)
                        activation_hits = [(nid, float(s)) for nid, s in hits_list]

                if cpt_config.activation_as_neighbors and activation_hits:
                    for nid, _ in activation_hits:
                        if nid not in neighbors:
                            neighbors.append(nid)

                # Score graph neighbors
                activation_nids = {nid for nid, _ in activation_hits}
                for nbr in neighbors:
                    if nbr in path_ids:
                        continue
                    try:
                        nbr_data = get_event_data(nbr)
                    except KeyError:
                        continue
                    score = float(score_fn(current_data, nbr_data))
                    # If this neighbor came from activation, apply multiplier/decay
                    if nbr in activation_nids and cpt_config.activation_score_multiplier is not None:
                        try:
                            score = float(score) * float(cpt_config.activation_score_multiplier)
                        except Exception:
                            pass
                    if cpt_config.min_als_score is not None and score < cpt_config.min_als_score:
                        continue
                    candidates.append((nbr, score))

                # Score activation-only candidates if not merged
                if not cpt_config.activation_as_neighbors and activation_hits:
                    for nid, _ in activation_hits:
                        if nid in path_ids:
                            continue
                        try:
                            nbr_data = get_event_data(nid)
                        except KeyError:
                            continue
                        if cpt_config.activation_als_config is not None and score_fn is calculate_als_score:
                            score = float(calculate_als_score(current_data, nbr_data, cpt_config.activation_als_config))
                        else:
                            score = float(score_fn(current_data, nbr_data))
                        # apply activation multiplier/decay
                        if cpt_config.activation_score_multiplier is not None:
                            try:
                                score = float(score) * float(cpt_config.activation_score_multiplier)
                            except Exception:
                                pass
                        if cpt_config.min_als_score is not None and score < cpt_config.min_als_score:
                            continue
                        candidates.append((nid, score))

                # Apply general top-k/top-p/sampling to all candidates
                if candidates:
                    candidates = apply_topk_topp_sampling(
                        candidates,
                        cpt_config.top_k,
                        cpt_config.top_p,
                        cpt_config.random_sampling,
                        cpt_config.sampling_temperature,
                    )
                candidates.sort(key=lambda x: x[1], reverse=True)
                top_candidates = candidates[: max(cpt_config.max_neighbors, 1)]

                # (no explored-path collection in this mode)

                # Expand up to branching neighbors for this partial path
                if not top_candidates:
                    # no expansions; keep the partial path as completed
                    new_active.append(part)
                else:
                    for nid, nscore in top_candidates[:branching]:
                        if nid in visited:
                            continue
                        new_path_ids = list(path_ids) + [nid]
                        new_acc = acc_score + float(nscore)
                        new_visited = set(visited)
                        new_visited.add(nid)
                        new_active.append({
                            "path_ids": new_path_ids,
                            "current_id": nid,
                            "acc_score": new_acc,
                            "visited": new_visited,
                        })

            # Prepare for next depth
            active_partial_paths = new_active
            depth += 1

        # After expansion, any remaining active_partial_paths are treated as completed paths
        for part in active_partial_paths:
            final_nodes = [get_event_data(nid) for nid in part["path_ids"]]
            p = ContextPath(nodes=final_nodes, score=part["acc_score"])
            paths.append(p)
            if logger:
                path_texts = [node.get("text", node.get("content", "<no text>")) for node in final_nodes]
                logger({
                    "event": "Full Context Path",
                    "path_number": len(paths),
                    "node_ids": part["path_ids"],
                    "score": part["acc_score"],
                    "context_contributed": path_texts,
                })

    # Sort final paths by score descending
    ranked = sorted(paths, key=lambda p: p.score, reverse=True)
    return CPTResult(paths=ranked)
