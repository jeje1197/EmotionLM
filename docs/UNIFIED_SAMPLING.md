# Unified Top-K/Top-P Sampling

## Overview

The Context Path Traversal system now supports **unified neighbor selection** that applies top-k, top-p (nucleus sampling), and stochastic sampling to **all neighbor candidates**, including both graph edge neighbors and spreading activation hits.

**Spreading Activation:** When enabled, the system performs semantic similarity lookups on the current event's text/content to discover memories with related content that may not be directly connected via graph edges. This enables retrieval of contextually relevant memories beyond the explicit graph structure.

## Configuration

### General Neighbor Selection

These settings apply to all neighbor candidates (graph + activation):

```python
CPTConfig(
    top_k=3,                      # Select top 3 neighbors by score
    top_p=0.9,                    # Nucleus sampling: keep top candidates until cumulative prob >= 0.9
    random_sampling=True,         # Enable stochastic sampling
    sampling_temperature=0.5,     # Temperature for softmax sampling (lower = more deterministic)
)
```

### Activation-Specific Overrides

If you want different behavior specifically for spreading activation hits, use the `activation_*` variants:

```python
CPTConfig(
    top_k=5,                      # General: allow 5 neighbors from graph edges
    activation_top_k=2,           # Override: only 2 activation hits
    activation_top_p=0.8,         # Different nucleus sampling for activation
    activation_random=True,       # Enable sampling for activation only
    activation_sampling_temperature=1.0,
)
```

**Note:** If `activation_*` fields are `None` (default), the general settings are used.

## How It Works

### 1. Candidate Scoring

At each traversal step, all candidates (graph neighbors + activation hits) are scored:
- **Graph neighbors:** Direct successors in the graph, scored via ALS or custom `score_fn`
- **Activation hits:** Memories discovered via semantic similarity on the current event's text/content
  - The system calls `vector_lookup(current_event.text)` to find semantically related memories
  - These hits are scored via semantic similarity (and optionally via `activation_als_config`)
  - Scores can be decayed using `activation_score_multiplier` to balance against graph neighbors

### 2. Top-P (Nucleus Sampling)

If `top_p` is set, candidates are filtered to the smallest set whose cumulative score reaches the threshold:

```python
# Example: scores [0.5, 0.3, 0.15, 0.05]
# With top_p=0.9, keep [0.5, 0.3, 0.15] (cumulative: 0.95)
```

### 3. Top-K Filtering

If `top_k` is set, only the top-k highest-scoring candidates are retained (after top-p if both are set).

### 4. Stochastic Sampling

If `random_sampling=True`, candidates are sampled probabilistically using softmax with `sampling_temperature`:
- Lower temperature → more deterministic (prefer high-scoring candidates)
- Higher temperature → more random (flatten distribution)

```python
# Temperature = 0.5 (sharper)
probs = softmax(scores / 0.5)

# Temperature = 2.0 (flatter, more exploration)
probs = softmax(scores / 2.0)
```

### 5. Branching Expansion

After filtering/sampling, up to `branching_factor` candidates are selected for expansion (default 1 = greedy).

## Examples

### Deterministic Top-K

Select top 2 neighbors at each step. With spreading activation enabled, this includes both graph neighbors and semantically similar memories:

```python
config = CPTConfig(
    top_k=2,
    spreading_activation=True,  # Enable semantic similarity lookup on event text
)
result = mg.retrieve(query, vector_lookup=my_lookup, cpt_config=config)
```

### Nucleus Sampling

Keep top 90% probability mass:

```python
config = CPTConfig(
    top_p=0.9,
    spreading_activation=True,
)
result = mg.retrieve(query, vector_lookup=my_lookup, cpt_config=config)
```

### Stochastic Exploration

Sample neighbors with temperature-controlled randomness:

```python
config = CPTConfig(
    random_sampling=True,
    sampling_temperature=0.7,  # Balance between exploitation and exploration
    spreading_activation=True,
)
result = mg.retrieve(query, vector_lookup=my_lookup, cpt_config=config)
```

### Mixed Strategy

Combine multiple sampling strategies:

```python
config = CPTConfig(
    top_p=0.95,                   # First apply nucleus sampling
    random_sampling=True,         # Then sample stochastically
    sampling_temperature=0.5,
    branching_factor=2,           # Expand 2 neighbors per step
    spreading_activation=True,
)
result = mg.retrieve(query, vector_lookup=my_lookup, cpt_config=config)
```

### Activation-Specific Behavior

Allow more graph neighbors, but limit activation hits. This is useful when you want broader graph exploration but controlled semantic spreading:

```python
config = CPTConfig(
    top_k=5,                      # Graph neighbors: up to 5
    activation_top_k=1,           # Activation: only 1 semantically similar memory per step
    spreading_activation=True,    # Perform semantic similarity on event text
)
result = mg.retrieve(query, vector_lookup=my_lookup, cpt_config=config)
```

## Why Unified Sampling?

Episodic memory retrieval is inherently **subjective and context-dependent**. Different retrieval strategies serve different goals:

- **Deterministic (no sampling):** Reproducible, optimal paths based on scoring function
- **Top-P:** Balanced exploration, considers multiple good options
- **Stochastic:** Mimics human memory's non-deterministic nature, enables diversity in path generation
- **Activation-specific:** Fine-grained control over spreading activation vs. graph traversal behavior

The unified system enables flexible experimentation with these strategies without code changes.

## Demo

See [unified_sampling_demo.py](examples/unified_sampling_demo.py) for comprehensive examples.
