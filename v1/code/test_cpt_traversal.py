import json
import torch
import numpy as np
from pathlib import Path

# Paths
MODEL_PATH = "v1/artifacts/pretrained/als_unified_linear.pt"
GRAPH_PATH = "v1/data/narrative_benchmark_graph.json"

class ALSModel(torch.nn.Module):
    def __init__(self, input_dim: int = 4):
        super().__init__()
        self.slp = torch.nn.Linear(input_dim, 1)
    def forward(self, x):
        return torch.sigmoid(self.slp(x))

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def calculate_temporal(d1, d2):
    delta = abs(d1 - d2)
    return 1.0 / (1.0 + np.log1p(delta))

def run_cpt_traversal():
    # 1. Load Model & Weights
    model = ALSModel(input_dim=4)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    # 2. Load Graph
    with open(GRAPH_PATH, "r") as f:
        data = json.load(f)
    
    nodes = {n["id"]: n for n in data["nodes"]}
    edges = data["edges"]
    
    # Adjacency List for Graph Traversal
    adj = {n_id: [] for n_id in nodes}
    for e in edges:
        adj[e["source"]].append(e["target"])

    # 3. Algorithm: Context Path Traversal (CPT)
    # Automatically find the starting spine node (earliest d_days)
    spine_nodes = sorted([n for n in data["nodes"] if n["id"].startswith("S")], key=lambda x: x["d_days"])
    current_node_id = spine_nodes[0]["id"]
    visited = []
    
    print("=== CPT: CONTEXT PATH TRAVERSAL (ALGORITHM VALIDATION) ===")
    print(f"Start Node: {current_node_id}")
    print(f"Metrics: ALS (Metric) | CPT (Algorithm) | $\\tau_I$: 0.4\n")

    path_summary = []

    for step in range(1, 10): # Shorter story traversal
        if current_node_id not in nodes: break
        current = nodes[current_node_id]
        visited.append(current_node_id)
        
        print(f"--- STEP {step}: {current['event'][:75]}... ---")
        print(f"Intensity: {current['emotional_intensity']:.2f} | State: {current['emotional_state']}")

        # A. LOCAL TRAVERSAL (Direct Neighbors)
        # Fetch direct neighbors and rank them by ALS
        neighbors = adj.get(current_node_id, [])
        if neighbors:
            neighbor_scores = []
            for n_id in neighbors:
                other = nodes[n_id]
                # Calculate ALS
                s = (cosine_similarity(current["semantic_vec"], other["semantic_vec"]) + 1) / 2
                e = (cosine_similarity(current["emotional_vec"], other["emotional_vec"]) + 1) / 2
                t = calculate_temporal(current["d_days"], other["d_days"])
                i_val = other["emotional_intensity"]
                
                feat = torch.tensor([[s, e, t, i_val]], dtype=torch.float32)
                with torch.no_grad():
                    score = model(feat).item()
                neighbor_scores.append((n_id, score))
            
            # Pick top-1 neighbor to traverse next
            neighbor_scores.sort(key=lambda x: x[1], reverse=True)
            next_node_id = neighbor_scores[0][0]
            print(f"   [CPT-LOCAL] Neighbor found: {next_node_id} (ALS: {neighbor_scores[0][1]:.3f})")
        else:
            next_node_id = None
            print("   [CPT-LOCAL] No neighbors found (End of spine).")

        # B. SPREADING ACTIVATION (Reminds Me Of...)
        # Triggered by intensity threshold
        if current["emotional_intensity"] >= 0.4:
            print(f"   [CPT-SPREAD] High Intensity detected. Executing global ALS search...")
            
            spreading_results = []
            for other_id, other in nodes.items():
                # Avoid current and local neighbors (we want the "distantly related")
                if other_id == current_node_id or other_id in neighbors:
                    continue
                # Local-Exclusion (Ignore last 24 hours to find deep memory)
                if abs(current["d_days"] - other["d_days"]) < 1.0:
                    continue
                
                s = (cosine_similarity(current["semantic_vec"], other["semantic_vec"]) + 1) / 2
                e = (cosine_similarity(current["emotional_vec"], other["emotional_vec"]) + 1) / 2
                t = calculate_temporal(current["d_days"], other["d_days"])
                i_val = other["emotional_intensity"]
                
                feat = torch.tensor([[s, e, t, i_val]], dtype=torch.float32)
                with torch.no_grad():
                    score = model(feat).item()
                spreading_results.append((other, score))
            
            spreading_results.sort(key=lambda x: x[1], reverse=True)
            top_jump = spreading_results[0][0]
            print(f"   [CPT-SPREAD] 'Reminds me of': \"{top_jump['event'][:60]}...\"")
            print(f"   [REASON] High ALS ({spreading_results[0][1]:.3f}) despite graph distance.")
        else:
            print(f"   [GATING] Intensity low. Spreading activation idle.")
        
        print("")
        if not next_node_id: break
        current_node_id = next_node_id

if __name__ == "__main__":
    run_cpt_traversal()
