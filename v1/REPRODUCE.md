Reproducing v1.0.1 experiments
==============================

This file documents minimal, end‑to‑end steps to regenerate the memory graph and edge‑classification dataset used in the paper, train the ALS model, and run evaluation pipelines.

Requirements
------------
- Python and the project dependencies installed (e.g. `pip install -r requirements.txt`).
- A valid `GEMINI_API_KEY` set in your environment for calling Gemini.

1. One‑shot dataset generation
------------------------------

This runs all three stages: (1) generate the 5×100 memory graph with Gemini, (2) add semantic and emotional embeddings, and (3) build the edge‑classification CSV used for ALS training.

```bash
export GEMINI_API_KEY=YOUR_KEY_HERE  # or setx on Windows
python v1/code/generate_dataset.py --stage all --output-dir data
```

Outputs (under `data/` by default):

- `memory_graph_raw.json` – raw memory graph (no embeddings).
- `memory_graph_embedded.json` – memory graph with semantic and emotional embeddings.
- `edge_classification_dataset.csv` – final edge‑classification dataset used for training.

2. Running dataset stages individually
--------------------------------------

You can also run each dataset‑generation step separately:

1. Generate the memory graph only:

```bash
python v1/code/generate_dataset.py --stage graph --output-dir data
```

2. Add embeddings to an existing raw graph:

```bash
python v1/code/generate_dataset.py --stage embed --output-dir data

# or, to use a specific input graph file
python v1/code/generate_dataset.py --stage embed --output-dir data --graph-in data/memory_graph_raw.json
```

3. Create the edge‑classification CSV from an embedded graph:

```bash
python v1/code/generate_dataset.py --stage csv --output-dir data

# or, to specify input graph and output CSV paths explicitly
python v1/code/generate_dataset.py --stage csv --output-dir data --graph-in data/memory_graph_embedded.json --csv-out data/edge_classification_dataset.csv
```

Using the v1 copies directly
----------------------------

For convenience, `v1/data/` already contains copies of the artifacts used in the paper:

- `v1/data/memory_graph_embedded.json`
- `v1/data/edge_classification_dataset.csv`

You can plug these directly into training/evaluation scripts without rerunning the generation pipeline.

3. Training the ALS model
-------------------------

With the edge‑classification CSV available (either via `generate_dataset.py` or the v1 copies), you can train the ALS model:

```bash
# Single‑stage training (all weights jointly)
python v1/code/train.py --config v1/code/training_config.json --mode single_stage

# Multistage training (semantic + time, then emotion)
python v1/code/train.py --config v1/code/training_config.json --mode multistage
```

The trained model is saved by default to `v1/artifacts/pretrained/als.pt`.

4. Evaluating binary classification
-----------------------------------

To evaluate the trained model on a validation CSV:

```bash
python v1/code/evaluate_binary_classification.py \
	--model v1/artifacts/pretrained/als.pt \
	--data v1/data/edge_classification_dataset.csv
```

This prints accuracy/precision/recall/F1, and can optionally write metrics to JSON with `--out`.

5. Running LLM‑as‑a‑judge for RAG vs Context Path
-------------------------------------------------

To run the end‑to‑end RAG vs Context Path experiment and evaluate it with
LLM‑as‑a‑judge:

```bash
export GEMINI_API_KEY=YOUR_KEY_HERE  # or setx on Windows
python v1/code/evaluate_rag_vs_context_path_responses.py
```

This will:
- load the embedded memory graph from `v1/data/memory_graph_embedded.json` and
	the queries from `v1/data/queries.json`;
- generate answers for each query using both retrieval methods (`rag` and
  `context_path`) and write them to
	`v1/artifacts/predictions/rag_vs_context_path_responses.csv`;
- run the LLM‑as‑a‑judge pipeline configured in
	`v1/code/llm_as_a_judge_config.json` and write scores to
	`v1/artifacts/llm_judge/results.json`.

You can edit `v1/code/llm_as_a_judge_config.json` to adjust the rubric,
prompting, and model parameters used in the paper.

Notes
-----
- The main, supported entrypoint for data generation is `v1/code/generate_dataset.py` with the `--stage` flag.
- All v1 training and evaluation commands are designed to run on CPU by default; see the project‑level README for environment details.
