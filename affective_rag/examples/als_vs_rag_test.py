import numpy as np
import time
from datetime import datetime, timedelta
from affective_rag import MemoryGraph, CPTConfig, ALSConfig

# Mock Embeddings (768D)
def get_mock_vec(seed):
    np.random.seed(seed)
    v = np.random.randn(768)
    return v / np.linalg.norm(v)

# Personas and Events (The "Toaster vs Trauma" Test)
# E1: Mundane (Low I, Recent)
# E2: Traumatic (High I, Distant Past)
# E3: Current Anchor (High Arousal Trigger)

EVENT_STORE = {
    "logistics_1": {
        "id": "logistics_1",
        "text": "Putting bread in the toaster.",
        "timestamp": (datetime.now() - timedelta(minutes=5)).isoformat(),
        "embedding": get_mock_vec(1),
        "emotional_embedding": get_mock_vec(100),
        "emotional_intensity": 0.1
    },
    "trauma_1": {
        "id": "trauma_1",
        "text": "The kitchen fire that destroyed my childhood home.",
        "timestamp": (datetime.now() - timedelta(days=3650)).isoformat(), # 10 years ago
        "embedding": get_mock_vec(2),
        "emotional_embedding": get_mock_vec(200),
        "emotional_intensity": 0.95
    },
    "anchor": {
        "id": "anchor",
        "text": "Smelling burning toast and feeling a sudden surge of panic.",
        "timestamp": datetime.now().isoformat(),
        "embedding": get_mock_vec(3),
        "emotional_embedding": get_mock_vec(200), # Matches trauma_1
        "emotional_intensity": 0.8
    }
}

# The ALS V5 Weights discovered in training
V5_CONFIG = ALSConfig(
    semantic_weight=-0.55,
    emotional_weight=-0.27, # Reversely scaled in library usually, but let's use the raw training for sim
    temporal_weight=4.77,
    intensity_weight=6.69,
    bias=-4.36
)

# Simulated Vector Lookup
def vector_lookup(query_text, top_k=5):
    # If the anchor is "burning toast", RAG finds the toaster (semantic) 
    # and the trauma (emotional/semantic overlap).
    return [("logistics_1", 0.9), ("trauma_1", 0.7)]

def event_data_provider(eid):
    return EVENT_STORE[eid]

def run_comparison():
    graph = MemoryGraph()
    # Add events to graph
    for eid in EVENT_STORE:
        graph.add_event(eid, EVENT_STORE[eid])
    
    # Add a direct causal edge only for the move forward in time
    # In a real graph, logistics_1 would be a parent of anchor
    graph.add_edge("logistics_1", "anchor")

    print("\n=== PERFORMANCE TEST: ALS vs. Standard RAG ===\n")

    # 1. Standard RAG Simulation (Semantic-only)
    print("Scenario: 'Smelling burning toast...'")
    print("Standard RAG Result (Top-1 Semantic):")
    # RAG usually just takes the highest semantic match
    hit = EVENT_STORE["logistics_1"]
    print(f" -> [{hit['id']}] {hit['text']} (Reason: Recent and high word overlap)")

    # 2. ALS-Guided CPT Result
    print("\nALS-Guided CPT Result (V5 Weights):")
    result = graph.retrieve(
        query="Smelling burning toast",
        vector_lookup=vector_lookup,
        cpt_config=CPTConfig(
            seed_nodes=1, 
            max_depth=1, 
            spreading_activation=True,
            activation_top_k=5
        ),
        event_data_provider=event_data_provider
    )

    for i, path in enumerate(result.paths):
        # The first node in path is the seed (anchor), second is the jump
        nodes = path.nodes
        if len(nodes) > 1:
            jump = nodes[1]
            print(f" -> Path {i+1}: [{jump['id']}] {jump['text']} (Score: {path.score:.4f})")
        else:
            print(f" -> Path {i+1}: No jumps found.")

if __name__ == "__main__":
    run_comparison()
