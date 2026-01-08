"""
Demonstration of unified top-k/top-p/sampling applied to all neighbors (graph + activation).

This shows how the general `top_k`, `top_p`, and `random_sampling` config applies to both
graph edge neighbors and spreading activation hits.

Pattern: External storage provides all event data via callbacks.
"""

from affective_rag import MemoryGraph, CPTConfig
import numpy as np

# External storage (simulated vector DB)
EVENT_STORE = {
    "e1": {
        "id": "e1",
        "text": "I felt very happy today",
        "semantic": np.array([1.0, 0.0, 0.0]),
        "emotional": np.array([0.9, 0.1, 0.0]),
        "timestamp": "2024-01-01T10:00:00",
    },
    "e2": {
        "id": "e2",
        "text": "My birthday party was fun",
        "semantic": np.array([0.0, 1.0, 0.0]),
        "emotional": np.array([0.8, 0.2, 0.0]),
        "timestamp": "2024-01-02T10:00:00",
    },
    "e3": {
        "id": "e3",
        "text": "I felt sad later",
        "semantic": np.array([0.0, 0.0, 1.0]),
        "emotional": np.array([0.1, 0.8, 0.1]),
        "timestamp": "2024-01-03T10:00:00",
    },
    "e4": {
        "id": "e4",
        "text": "The birthday cake was delicious and brought me joy",
        "semantic": np.array([0.5, 0.5, 0.0]),
        "emotional": np.array([0.9, 0.05, 0.05]),
        "timestamp": "2024-01-02T14:00:00",
    },
}


def fake_vector_lookup(query: str, top_k: int = None) -> list:
    """
    Mock vector lookup that searches external storage by semantic similarity.
    """
    keyword_map = {
        "happy": [("e1", 0.95), ("e4", 0.85)],
        "joy": [("e1", 0.9), ("e4", 0.88)],
        "birthday": [("e2", 0.92), ("e4", 0.8)],
        "cake": [("e4", 0.95), ("e2", 0.7)],
        "sad": [("e3", 0.9)],
        "graduation": [("e2", 0.85)],
        "celebrate": [("e2", 0.88), ("e4", 0.82)],
    }
    
    results = []
    for word, candidates in keyword_map.items():
        if word in query.lower():
            results.extend(candidates)
    
    seen = {}
    for nid, score in results:
        if nid not in seen or score > seen[nid]:
            seen[nid] = score
    
    results = [(nid, score) for nid, score in seen.items()]
    results.sort(key=lambda x: x[1], reverse=True)
    
    if top_k:
        results = results[:top_k]
    
    return results


def event_data_provider(event_id: str) -> dict:
    """Fetch event data from external storage by ID."""
    return EVENT_STORE[event_id]


def main():
    # Create memory graph - structure only
    mg = MemoryGraph()
    
    # Add nodes with minimal data
    for event_id in EVENT_STORE.keys():
        mg.add_event(event_id, {"id": event_id})
    
    # Create graph edges (e1 → e2 → e3)
    mg.add_edge("e1", "e2")
    mg.add_edge("e2", "e3")
    # Note: e4 is NOT connected via edges, only discoverable via activation
    
    print("\n=== Graph Structure ===")
    print("Edges: e1 → e2 → e3")
    print("Isolated: e4 (semantically similar to e1/e2, discoverable via spreading activation)")
    
    # Demo 1: No sampling (deterministic)
    print("\n=== Demo 1: Deterministic (no top-k/top-p) ===")
    config1 = CPTConfig(
        seed_nodes=1,  # Number of seeds to retrieve
        max_depth=3,
        spreading_activation=True,
        activation_top_k=3,
    )
    result1 = mg.retrieve(
        "happy birthday",
        vector_lookup=fake_vector_lookup,
        cpt_config=config1,
        event_data_provider=event_data_provider,
    )
    print(f"Paths found: {len(result1.paths)}")
    for path in result1.paths:
        node_ids = [n.get("id") for n in path.nodes]
        print(f"  {node_ids} (score: {path.score:.2f})")
    
    # Demo 2: General top-k=2 (apply to all neighbors)
    print("\n=== Demo 2: General top_k=2 (applies to all neighbors) ===")
    config2 = CPTConfig(
        seed_nodes=1,
        max_depth=3,
        top_k=2,  # Limit all neighbor selection to top 2
        spreading_activation=True,
        activation_top_k=None,  # Use general top_k
    )
    result2 = mg.retrieve(
        "happy birthday",
        vector_lookup=fake_vector_lookup,
        cpt_config=config2,
        event_data_provider=event_data_provider,
    )
    print(f"Paths found: {len(result2.paths)}")
    for path in result2.paths:
        node_ids = [n.get("id") for n in path.nodes]
        print(f"  {node_ids} (score: {path.score:.2f})")
    
    # Demo 3: General top-p=0.9 (nucleus sampling)
    print("\n=== Demo 3: General top_p=0.9 (nucleus sampling for all) ===")
    config3 = CPTConfig(
        seed_nodes=1,
        max_depth=3,
        top_p=0.9,  # Nucleus sampling for all neighbors
        spreading_activation=True,
        activation_top_p=None,  # Use general top_p
    )
    result3 = mg.retrieve(
        "happy birthday",
        vector_lookup=fake_vector_lookup,
        cpt_config=config3,
        event_data_provider=event_data_provider,
    )
    print(f"Paths found: {len(result3.paths)}")
    for path in result3.paths:
        node_ids = [n.get("id") for n in path.nodes]
        print(f"  {node_ids} (score: {path.score:.2f})")
    
    # Demo 4: Random sampling with temperature
    print("\n=== Demo 4: Random sampling (temperature=0.5) ===")
    np.random.seed(42)  # For reproducibility
    config4 = CPTConfig(
        seed_nodes=1,
        max_depth=3,
        random_sampling=True,
        sampling_temperature=0.5,
        spreading_activation=True,
        activation_random=None,  # Use general random_sampling
    )
    result4 = mg.retrieve(
        "happy birthday",
        vector_lookup=fake_vector_lookup,
        cpt_config=config4,
        event_data_provider=event_data_provider,
    )
    print(f"Paths found: {len(result4.paths)}")
    for path in result4.paths:
        node_ids = [n.get("id") for n in path.nodes]
        print(f"  {node_ids} (score: {path.score:.2f})")
    
    # Demo 5: Activation-specific override
    print("\n=== Demo 5: Activation-specific override (activation_top_k=1 overrides general top_k=3) ===")
    config5 = CPTConfig(
        seed_nodes=1,
        max_depth=3,
        top_k=3,  # General: allow 3 neighbors
        spreading_activation=True,
        activation_top_k=1,  # But only 1 activation hit
    )
    result5 = mg.retrieve(
        "happy birthday",
        vector_lookup=fake_vector_lookup,
        cpt_config=config5,
        event_data_provider=event_data_provider,
    )
    print(f"Paths found: {len(result5.paths)}")
    for path in result5.paths:
        node_ids = [n.get("id") for n in path.nodes]
        print(f"  {node_ids} (score: {path.score:.2f})")
    
    print("\n=== Key Points ===")
    print("✓ top_k/top_p/random_sampling apply to ALL neighbors (graph edges + activation)")
    print("✓ activation_top_k/activation_top_p/activation_random override general settings for activation only")
    print("✓ This enables flexible neighbor selection while maintaining activation-specific behavior when needed")


if __name__ == "__main__":
    main()
