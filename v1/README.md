v1 scaffold for paper artifacts and code
======================================

This folder contains the versioned code and data bundle for the paper's v1.0.1
release: dataset generation, ALS training, binary evaluation, and an
LLM‑as‑a‑judge pipeline scaffold.

Structure
---------
- `code/` — scripts and configs used in the paper:
	- `generate_dataset.py` — 3‑stage pipeline to create the memory graph, add embeddings, and produce the edge‑classification CSV.
	- `train.py` — trains the ALS edge classifier using `training_config.json` (single‑stage or multistage mode).
	- `training_config.json` — hyperparameters and paths for ALS training.
	- `evaluate_binary_classification.py` — evaluates a trained ALS model on an edge‑classification CSV.
	- `evaluate_rag_vs_context_path_responses.py` — runs the RAG vs Context Path experiment and LLM‑as‑a‑judge evaluation using `llm_as_a_judge_config.json`.
	- `llm_as_a_judge_config.json` — configuration and rubric for LLM‑as‑a‑judge.
- `data/` — v1 data bundle:
	- `memory_graph_embedded.json` — copy of the embedded memory graph used for training.
	- `edge_classification_dataset.csv` — copy of the edge‑classification dataset used for ALS.
- `artifacts/` — pretrained checkpoints and judge outputs (e.g. `pretrained/als.pt`).
- `tasks/` — hyperparameters and small README.
- `REPRODUCE.md` — step‑by‑step commands to regenerate the dataset and run key scripts.

Notes
-----
- For full reproduction, see `v1/REPRODUCE.md` for the exact commands to:
	1. regenerate the memory graph and edge dataset; and
	2. train/evaluate the ALS model and run LLM‑as‑a‑judge.
- The canonical artifacts under the project root `data/` (for example
	`data/memory_graph_embedded.json` and `data/edge_classification_dataset.csv`)
	correspond to the files copied under `v1/data/` and can also be regenerated
	via `v1/code/generate_dataset.py`.
