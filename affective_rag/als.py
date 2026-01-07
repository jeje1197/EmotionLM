import numpy as np
from datetime import datetime
from pydantic import BaseModel


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate the cosine similarity between two vectors."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a) 
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0

    return dot_product / (norm_a * norm_b)


def temporal_proximity(timestamp_a: str, timestamp_b: str) -> float:
    """Calculate temporal proimity between two timestamps in ISO 8601 format."""
    time_a = datetime.fromisoformat(timestamp_a.replace("Z", "+00:00"))
    time_b = datetime.fromisoformat(timestamp_b.replace("Z", "+00:00"))
    time_difference = abs(time_a - time_b).total_seconds()

    return 1 / (1 + time_difference)


class ALSConfig(BaseModel):
    """Configuration for the Affective Link Score computation."""
    semantic_weight: float = 0
    emotional_weight: float = 0
    temporal_weight: float = 0
    bias: float = 0


"""Trained default ALS weights from v1 experimentation."""
DEFAULT_ALS_CONFIG = ALSConfig(
    semantic_weight=0.0791,
    emotional_weight=-0.5179,
    temporal_weight=3.1470,
    bias=0.0
)


def calculate_als_score(a, b, als_config: ALSConfig = DEFAULT_ALS_CONFIG) -> float:
    """Compute the Affective Link Score (ALS) for two events."""
    w_s, w_e, w_t, b = als_config.semantic_weight, als_config.emotional_weight, als_config.temporal_weight, als_config.bias
    a_sem = a["semantic_vec"]
    b_sem = b["semantic_vec"]
    a_emo = a["emotional_vec"]
    b_emo = b["emotional_vec"]
    a_time = a["timestamp"]
    b_time = b["timestamp"]

    return w_s * cosine_similarity(a_sem, b_sem) + w_e * cosine_similarity(a_emo, b_emo) + w_t * temporal_proximity(a_time, b_time) + b
