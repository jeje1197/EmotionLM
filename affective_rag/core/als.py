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
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


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
    bias: float = 0


"""Trained default ALS weights from v1 experimentation."""
DEFAULT_ALS_CONFIG = ALSConfig(
    semantic_weight=0.0791,
    emotional_weight=-0.5179,
    temporal_weight=3.1470,
    bias=0.0,
)


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
