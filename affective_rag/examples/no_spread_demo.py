"""Demo: spreading activation disabled — `e4` should NOT appear in the path.

Pattern: External storage provides data via callbacks.
"""
from affective_rag import MemoryGraph, CPTConfig

EVENT_STORE = {
    "e1": {"id":"e1","text":"Alice arrives at the party","timestamp":"2024-01-01T18:00:00Z","semantic_vec":[0.1,0.2,0.3]},
    "e2": {"id":"e2","text":"Alice argues with Bob","timestamp":"2024-01-01T19:30:00Z","semantic_vec":[0.15,0.25,0.28]},
    "e3": {"id":"e3","text":"They reconcile and laugh together","timestamp":"2024-01-01T20:00:00Z","semantic_vec":[0.12,0.22,0.32]},
    "e4": {"id":"e4","text":"Later they met at a cafe and apologized to each other","timestamp":"2024-01-02T12:00:00Z","semantic_vec":[0.11,0.21,0.31]},
}

def fake_vector_lookup(q, top_k: int = 5):
    q = str(q).lower()
    if "reconc" in q or "laugh" in q or "apolog" in q:
        return [("e3", 0.9), ("e4", 0.85), ("e2", 0.2)][:top_k]
    if "argu" in q or "bob" in q:
        return [("e2", 0.95), ("e3", 0.4), ("e4", 0.25)][:top_k]
    if "arriv" in q or "party" in q:
        return [("e1", 0.95), ("e2", 0.4), ("e3", 0.2)][:top_k]
    return [("e1",0.6),("e2",0.5),("e3",0.4),("e4",0.3)][:top_k]

def event_data_provider(event_id: str) -> dict:
    return EVENT_STORE[event_id]


def main():
    graph = MemoryGraph()

    for event_id in EVENT_STORE.keys():
        graph.add_event(event_id, {"id": event_id})

    graph.add_edge("e1","e2")
    graph.add_edge("e2","e3")

    result = graph.retrieve(
        query="What happened at the party?",
        vector_lookup=fake_vector_lookup,
        cpt_config=CPTConfig(max_depth=3, spreading_activation=False),
        event_data_provider=event_data_provider,
    )

    print("no_spread_demo: discovered paths:")
    for p in result.paths:
        ids = [n.get("id") or n.get("id") for n in p.nodes]
        print(ids)


if __name__ == "__main__":
    main()
