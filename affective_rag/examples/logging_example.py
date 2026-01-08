"""
Example demonstrating the logging feature for Context Path Traversal observability.

Pattern: External storage + callbacks (production-ready).
"""
import json
from affective_rag import MemoryGraph, CPTConfig

# External storage
EVENT_STORE = {
    "e1": {
        "id": "e1",
        "text": "Alice arrives at the party",
        "timestamp": "2024-01-01T18:00:00Z",
        "emotion": "excited",
        "semantic_vec": [0.1, 0.2, 0.3],
        "emotional_vec": [0.8, 0.1, 0.1],
    },
    "e2": {
        "id": "e2",
        "text": "Alice argues with Bob",
        "timestamp": "2024-01-01T19:30:00Z",
        "emotion": "angry",
        "semantic_vec": [0.15, 0.25, 0.28],
        "emotional_vec": [0.1, 0.1, 0.8],
    },
    "e3": {
        "id": "e3",
        "text": "They reconcile and laugh together",
        "timestamp": "2024-01-01T20:00:00Z",
        "emotion": "happy",
        "semantic_vec": [0.12, 0.22, 0.32],
        "emotional_vec": [0.9, 0.05, 0.05],
    },
    "e4": {
        "id": "e4",
        "text": "Later they met at a cafe and apologized to each other",
        "timestamp": "2024-01-02T12:00:00Z",
        "emotion": "relieved",
        "semantic_vec": [0.11, 0.21, 0.31],
        "emotional_vec": [0.85, 0.1, 0.05],
    },
}


def pretty_logger(log_data: dict):
    """Pretty print log data as formatted JSON."""
    try:
        compact = json.dumps(log_data, separators=(",", ":"), default=str, ensure_ascii=False)
        print(compact)
    except Exception:
        print(repr(log_data))

def event_data_provider(event_id: str) -> dict:
    return EVENT_STORE[event_id]


def main():
    # Create a simple graph - structure only
    graph = MemoryGraph()
    
    for event_id in EVENT_STORE.keys():
        graph.add_event(event_id, {"id": event_id})
    
    # Add edges
    graph.add_edge("e1", "e2")
    graph.add_edge("e2", "e3")
    
    # Enable logging with pretty printer
    graph.set_logging(enabled=True, logger_provider=pretty_logger)
    
    # Fake vector lookup for demo
    def fake_vector_lookup(query, top_k: int = 5):
        # Very small keyword-based simulator to show semantic spreading activation
        q = str(query).lower()
        if "reconc" in q or "laugh" in q or "apolog" in q:
            return [("e3", 0.9), ("e4", 0.85), ("e2", 0.2), ("e1", 0.1)][:top_k]
        if "argu" in q or "bob" in q:
            return [("e2", 0.95), ("e3", 0.4), ("e4", 0.25), ("e1", 0.2)][:top_k]
        if "arriv" in q or "party" in q:
            return [("e1", 0.95), ("e2", 0.4), ("e3", 0.2), ("e4", 0.1)][:top_k]
        # Fallback
        return [("e1", 0.6), ("e2", 0.5), ("e3", 0.4), ("e4", 0.3)][:top_k]
    
    # Run retrieval - this will generate detailed logs
    result = graph.retrieve(
        query="What happened at the party?",
        vector_lookup=fake_vector_lookup,
        cpt_config=CPTConfig(max_depth=3, spreading_activation=True, activation_top_k=2, activation_min_score=0.1),
        event_data_provider=event_data_provider,
    )
    
    print("\n--- FINAL RESULTS ---")
    print(f"Discovered {len(result.paths)} context path(s):\n")
    
    for i, path in enumerate(result.paths, 1):
        print(f"Path {i}: score={path.score:.4f}")
        for node in path.nodes:
            print(f"  - {node.get('text', '<no text>')}")


if __name__ == "__main__":
    main()
