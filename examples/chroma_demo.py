"""ChromaDB integration demo for affective_rag.

This demo shows two usage patterns:

1. Simple mode: Store event data in MemoryGraph (good for demos/tests)
2. Production mode: Store only structure in MemoryGraph, fetch data from ChromaDB
   via event_data_provider (avoids duplication)

Requirements:
    pip install chromadb

Run from the project root:
    python -m examples.chroma_demo
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from affective_rag import MemoryGraph, CPTConfig


def build_demo_graph_simple() -> MemoryGraph:
    """Simple mode: store all data in MemoryGraph."""
    graph = MemoryGraph()

    graph.add_event(
        "e1",
        {
            "text": "Discussed project milestones with the team.",
            "timestamp": "2025-12-27T10:00:00Z",
            "emotion": {"valence": 0.8, "arousal": 0.4},
        },
    )

    graph.add_event(
        "e2",
        {
            "text": "Received positive feedback from stakeholders.",
            "timestamp": "2025-12-27T11:00:00Z",
            "emotion": {"valence": 0.9, "arousal": 0.5},
        },
    )

    graph.add_event(
        "e3",
        {
            "text": "Identified a few risks in the schedule.",
            "timestamp": "2025-12-27T12:00:00Z",
            "emotion": {"valence": 0.2, "arousal": 0.6},
        },
    )

    graph.add_edge("e1", "e2")
    graph.add_edge("e2", "e3")

    return graph


def build_demo_graph_structure_only() -> MemoryGraph:
    """Production mode: store only structure (IDs + edges) in MemoryGraph."""
    graph = MemoryGraph()

    # Just add nodes with IDs, no data
    graph.add_event("e1", {})
    graph.add_event("e2", {})
    graph.add_event("e3", {})

    graph.add_edge("e1", "e2")
    graph.add_edge("e2", "e3")

    return graph


def build_chroma_collection_and_metadata():
    """Create an in-memory Chroma collection and event metadata store.

    Returns:
        - collection: ChromaDB collection for vector search
        - metadata: dict mapping event_id -> full event data
    """

    try:
        import chromadb
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "chromadb is not installed. Install with `pip install chromadb` to run this example."
        ) from exc

    # Event data that would normally live in your database
    event_metadata: Dict[str, dict] = {
        "e1": {
            "id": "e1",
            "text": "Discussed project milestones with the team.",
            "timestamp": "2025-12-27T10:00:00Z",
            "emotion": {"valence": 0.8, "arousal": 0.4},
        },
        "e2": {
            "id": "e2",
            "text": "Received positive feedback from stakeholders.",
            "timestamp": "2025-12-27T11:00:00Z",
            "emotion": {"valence": 0.9, "arousal": 0.5},
        },
        "e3": {
            "id": "e3",
            "text": "Identified a few risks in the schedule.",
            "timestamp": "2025-12-27T12:00:00Z",
            "emotion": {"valence": 0.2, "arousal": 0.6},
        },
    }

    client = chromadb.Client()
    collection = client.get_or_create_collection("affective_memories")

    ids: List[str] = list(event_metadata.keys())
    docs: List[str] = [event_metadata[i]["text"] for i in ids]

    collection.add(ids=ids, documents=docs)

    return collection, event_metadata


def make_chroma_lookup(collection) -> Any:
    """Build a `vector_lookup` adapter around a Chroma collection.

    The adapter accepts a text query and returns a list of (id, score)
    tuples ordered by descending similarity.
    """

    def lookup(query: Any, top_k: int = 5) -> List[Tuple[str, float]]:
        if not isinstance(query, str):
            query = str(query)

        # We let Chroma handle text -> embedding internally.
        res = collection.query(query_texts=[query], n_results=top_k)

        ids = res.get("ids", [[]])[0]
        # Chroma usually returns distances; convert to a similarity-like score.
        distances = res.get("distances") or [[]]
        distances = distances[0] if distances else []

        scores: List[Tuple[str, float]] = []
        if distances and len(distances) == len(ids):
            for i, d in zip(ids, distances):
                # Smaller distance -> higher score
                scores.append((str(i), float(1.0 / (1.0 + float(d)))))
        else:
            # Fallback: just return ids with dummy scores
            scores = [(str(i), 1.0) for i in ids]

        return scores

    return lookup


def main() -> None:
    print("=" * 60)
    print("Mode 1: Simple (data stored in MemoryGraph)")
    print("=" * 60)

    graph_simple = build_demo_graph_simple()
    collection, _ = build_chroma_collection_and_metadata()
    vector_lookup = make_chroma_lookup(collection)

    query = "What was the feedback on the project?"
    result = graph_simple.retrieve(
        query, 
        vector_lookup=vector_lookup, 
        cpt_config=CPTConfig(max_depth=3)
    )

    print("\nContext paths:\n")
    for i, path in enumerate(result.paths, start=1):
        ids = [node.get("id") for node in path.nodes]
        texts = [node.get("text") for node in path.nodes]
        print(f"Path {i}: ids={ids}, score={path.score:.4f}")
        for t in texts:
            print(f"  - {t}")
        print()

    print("=" * 60)
    print("Mode 2: Production (data in ChromaDB, structure in MemoryGraph)")
    print("=" * 60)

    graph_structure = build_demo_graph_structure_only()
    collection2, event_metadata = build_chroma_collection_and_metadata()
    vector_lookup2 = make_chroma_lookup(collection2)

    # Provide an event_data_provider that fetches from our metadata store
    # (in production, this would query your database)
    def get_event_data(event_id: str) -> dict:
        return event_metadata[event_id]

    result2 = graph_structure.retrieve(
        query,
        vector_lookup=vector_lookup2,
        event_data_provider=get_event_data,
        cpt_config=CPTConfig(max_depth=3),
    )

    print("\nContext paths (fetched from external metadata store):\n")
    for i, path in enumerate(result2.paths, start=1):
        ids = [node.get("id") for node in path.nodes]
        texts = [node.get("text") for node in path.nodes]
        print(f"Path {i}: ids={ids}, score={path.score:.4f}")
        for t in texts:
            print(f"  - {t}")
        print()


if __name__ == "__main__":
    main()
