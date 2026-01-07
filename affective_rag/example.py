"""Simple example demonstrating Affective-RAG usage.

This script shows how to:
- build a small memory graph with `add_event` and `add_edge`
- plug in a trivial vector store lookup callable
- run Context Path Traversal and inspect the resulting paths

Run from the project root:

    python -m affective_rag.example
"""

from __future__ import annotations

from typing import Any, List, Tuple

from . import MemoryGraph


def fake_vector_store_lookup(_query: Any) -> List[Tuple[str, float]]:
    """Very simple lookup function used for the example.

    In a real system, this would:
    - embed the query, or
    - accept a pre-computed query vector, and
    - query a vector DB / index

    Here we just pretend that e1 is the best seed and e2 is second-best.
    """

    return [("e1", 1.0), ("e2", 0.8)]


def main() -> None:
    # Build a MemoryGraph and populate it with events
    graph = MemoryGraph()

    graph.add_event(
        "e1",
        {
            "text": "Alice arrives at the party",
            "timestamp": "2024-01-01T19:00:00Z",
            "emotion": {"label": "Joy", "valence": 0.9, "arousal": 0.6},
        },
    )

    graph.add_event(
        "e2",
        {
            "text": "Alice argues with Bob",
            "timestamp": "2024-01-01T20:00:00Z",
            "emotion": {"label": "Conflict", "valence": -0.6, "arousal": 0.8},
        },
    )

    graph.add_event(
        "e3",
        {
            "text": "They reconcile and laugh together",
            "timestamp": "2024-01-01T21:00:00Z",
            "emotion": {"label": "Bittersweet", "valence": 0.5, "arousal": 0.4},
        },
    )

    # Simple temporal chain e1 -> e2 -> e3
    graph.add_edge("e1", "e2")
    graph.add_edge("e2", "e3")

    # Retrieve episodic context paths using a natural-language query
    context = graph.retrieve(
        "Why did Alice reconcile with Bob?",
        vector_lookup=fake_vector_store_lookup,
        depth=3,
    )

    print("Discovered context paths:\n")
    for i, path in enumerate(context.paths, start=1):
        ids = [node.get("id") for node in path.nodes]
        texts = [node.get("text") for node in path.nodes]
        print(f"Path {i}: ids={ids}, score={path.score:.4f}")
        for t in texts:
            print(f"  - {t}")
        print()


if __name__ == "__main__":
    main()
