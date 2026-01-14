import numpy as np
from datetime import datetime
from pydantic import BaseModel
from typing import Any, Dict


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate the cosine similarity between two vectors.

    Returns 0.0 if either vector has zero norm.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.5
    sim = np.dot(a, b) / (norm_a * norm_b)
    # Scale from [-1, 1] to [0, 1] range
    return (float(sim) + 1.0) / 2.0


def temporal_proximity(timestamp_a: str, timestamp_b: str) -> float:
    """Calculate temporal proximity between two timestamps in ISO 8601 format.

    Uses days-based normalization: 1 / (1 + delta_days).
    """
    time_a = datetime.fromisoformat(timestamp_a.replace("Z", "+00:00"))
    time_b = datetime.fromisoformat(timestamp_b.replace("Z", "+00:00"))
    time_difference = abs(time_a - time_b).total_seconds()
    delta_days = time_difference / 86400.0

    return 1.0 / (1.0 + delta_days)


class ALSConfig(BaseModel):
    """Configuration for the Affective Link Score computation."""
    semantic_weight: float = 0
    emotional_weight: float = 0
    temporal_weight: float = 0
    intensity_weight: float = 0  # Added for V4 compatibility
    bias: float = 0
    model_type: str = "linear"  # "linear" or "mlp"
    mlp_path: str = ""          # Path to .pt weights if using mlp

def calculate_als_score(
    event_a: Dict[str, Any], 
    event_b: Dict[str, Any], 
    config: ALSConfig = None,
    mlp_model: Any = None
) -> float:
    """Computes the link score using either linear or MLP logic."""
    if config is None:
        from .als import DEFAULT_ALS_CONFIG
        config = DEFAULT_ALS_CONFIG

    # Extract features for scoring
    # 1. Semantic Similarity (Event A <-> Event B)
    s = cosine_similarity(
        event_a.get("embedding", np.zeros(768)),
        event_b.get("embedding", np.zeros(768))
    )
    
    # 2. Emotional Alignment (State A <-> State B)
    # Note: If memory doesn't have emotions, default to mid-range
    e = cosine_similarity(
        event_a.get("emotional_embedding", np.zeros(768)),
        event_b.get("emotional_embedding", np.zeros(768))
    )
    
    # 3. Temporal Proximity
    t = temporal_proximity(
        event_a.get("timestamp", ""),
        event_b.get("timestamp", "")
    )
    
    # 4. Target Intensity (from Event B)
    i = event_b.get("emotional_intensity", 0.0)

    if config.model_type == "mlp" and mlp_model is not None:
        import torch
        # [S, E, T, I]
        feat_tensor = torch.tensor([[s, e, t, i]], dtype=torch.float32)
        with torch.no_grad():
            return float(mlp_model(feat_tensor).item())
    
    # Linear ALS
    score = (s * config.semantic_weight + 
             e * config.emotional_weight + 
             t * config.temporal_weight + 
             i * config.intensity_weight + 
             config.bias)
    
    return 1.0 / (1.0 + np.exp(-score))


class HybridMemoryScorer:
    """The 'Dual-Pathway' Scorer.
    
    Uses Linear ALS for routine/procedural continuity and MLP for 
    high-intensity episodic resonance.
    """
    def __init__(self, linear_config: ALSConfig, mlp_model: Any = None, threshold: float = 0.4):
        self.linear_config = linear_config
        self.mlp_model = mlp_model
        self.threshold = threshold

    def score(self, event_a: Dict[str, Any], event_b: Dict[str, Any]) -> float:
        intensity = event_b.get("emotional_intensity", 0.0)
        
        # EPISODIC PATHWAY (MLP) for high-intensity significance
        if intensity >= self.threshold and self.mlp_model is not None:
            return calculate_als_score(
                event_a, event_b, 
                config=ALSConfig(model_type="mlp"), 
                mlp_model=self.mlp_model
            )
        
        # PROCEDURAL PATHWAY (Linear) for routine logitstics
        return calculate_als_score(event_a, event_b, config=self.linear_config)

    def __call__(self, event_a: Dict[str, Any], event_b: Dict[str, Any]) -> float:
        return self.score(event_a, event_b)
        intensity = features.get("i", 0)
        
        # EPISODIC PATHWAY (MLP) for high-intensity
        if intensity >= self.threshold and self.mlp_model is not None:
            import torch
            s, e, t = features.get("s",0), features.get("e",0), features.get("t",0)
            feat_tensor = torch.tensor([[s, e, t, intensity]], dtype=torch.float32)
            with torch.no_grad():
                return float(self.mlp_model(feat_tensor).item())
        
        # PROCEDURAL PATHWAY (Linear) for everything else
        return calculate_als_score(features, self.linear_config)


def calculate_als_score(a: Dict[str, Any], b: Dict[str, Any], als_config: ALSConfig = DEFAULT_ALS_CONFIG) -> float:
    """Compute the Affective Link Score (ALS) for two events.

    Missing features (semantic/emotional vectors or timestamps) contribute 0.
    """
    # Unpack weights and bias; avoid shadowing the second event argument "b".
    w_s = float(als_config.semantic_weight)
    w_e = float(als_config.emotional_weight)
    w_t = float(als_config.temporal_weight)
    bias = float(als_config.bias)

    sem_score = 0.0
    emo_score = 0.0
    temp_score = 0.0

    a_sem = a.get("semantic_vec")
    b_sem = b.get("semantic_vec")
    if a_sem is not None and b_sem is not None:
        sem_score = cosine_similarity(a_sem, b_sem)

    a_emo = a.get("emotional_vec")
    b_emo = b.get("emotional_vec")
    if a_emo is not None and b_emo is not None:
        emo_score = cosine_similarity(a_emo, b_emo)

    a_time = a.get("timestamp")
    b_time = b.get("timestamp")
    if a_time is not None and b_time is not None:
        temp_score = temporal_proximity(a_time, b_time)

    return float(w_s * sem_score + w_e * emo_score + w_t * temp_score + bias)
