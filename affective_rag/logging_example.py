"""
Example demonstrating the logging feature for Context Path Traversal observability.
"""
import json
from affective_rag import MemoryGraph, CPTConfig


def pretty_logger(log_data: dict):
    """Pretty print log data as formatted JSON."""
    print(json.dumps(log_data, indent=2, default=str))


def main():
    # Create a simple graph
    graph = MemoryGraph()
    
    # Add events with emotion metadata
    graph.add_event("e1", {
        "id": "e1",
        "text": "Alice arrives at the party",
        "timestamp": "2024-01-01T18:00:00Z",
        "emotion": "excited",
        "semantic_vec": [0.1, 0.2, 0.3],
        "emotional_vec": [0.8, 0.1, 0.1],
    })
    
    graph.add_event("e2", {
        "id": "e2",
        "text": "Alice argues with Bob",
        "timestamp": "2024-01-01T19:30:00Z",
        "emotion": "angry",
        "semantic_vec": [0.15, 0.25, 0.28],
        "emotional_vec": [0.1, 0.1, 0.8],
    })
    
    graph.add_event("e3", {
        "id": "e3",
        "text": "They reconcile and laugh together",
        "timestamp": "2024-01-01T20:00:00Z",
        "emotion": "happy",
        "semantic_vec": [0.12, 0.22, 0.32],
        "emotional_vec": [0.9, 0.05, 0.05],
    })
    
    # Add edges
    graph.add_edge("e1", "e2")
    graph.add_edge("e2", "e3")
    
    # Enable logging with pretty printer
    graph.set_logging(enabled=True, logger_provider=pretty_logger)
    
    # Fake vector lookup for demo
    def fake_vector_lookup(query):
        return [("e1", 0.95), ("e2", 0.75), ("e3", 0.60)]
    
    # Run retrieval - this will generate detailed logs
    result = graph.retrieve(
        query="What happened at the party?",
        vector_lookup=fake_vector_lookup,
        cpt_config=CPTConfig(max_depth=3),
    )
    
    print("\n--- FINAL RESULTS ---")
    print(f"Discovered {len(result.paths)} context path(s):\n")
    
    for i, path in enumerate(result.paths, 1):
        print(f"Path {i}: score={path.score:.4f}")
        for node in path.nodes:
            print(f"  - {node.get('text', '<no text>')}")


if __name__ == "__main__":
    main()
