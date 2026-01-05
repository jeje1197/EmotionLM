#!/usr/bin/env python3
"""Run LLM-as-a-judge evaluation as described in the paper (v1 scaffold).

This script loads a predictions CSV and an LLM-as-a-judge configuration
JSON and calls an LLM (e.g., Gemini) to score each prediction according
to a rubric. Results are written to a JSON file for analysis.

The exact data columns and rubric are specified in llm_as_a_judge_config.json.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import sys
from pathlib import Path as _Path

ROOT = _Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from google import genai  # type: ignore  # noqa: E402
from google.genai import types  # type: ignore  # noqa: E402

import pandas as pd


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
    return genai.Client(api_key=api_key)


def make_prompt(example: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    rubric = cfg["rubric"]
    template = cfg["prompt_template"]
    context = example.get("context", "")
    reference = example.get("reference", "")
    prediction = example.get("prediction", "")

    return template.format(
        scale_min=rubric["scale_min"],
        scale_max=rubric["scale_max"],
        context=context,
        reference=reference,
        prediction=prediction,
    )


def score_example(client: genai.Client, example: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    model_name = cfg.get("model", "gemini-2.5-flash")
    temperature = float(cfg.get("temperature", 0.2))

    system_instruction = cfg.get("system_instruction", "You are an impartial, consistent evaluator of model outputs.")
    prompt = make_prompt(example, cfg)

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json",
    )

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=config,
        generation_config=types.GenerateContentConfig(temperature=temperature),
    )

    try:
        scores = json.loads(response.text)
    except Exception:
        scores = {"raw_response": response.text}

    return scores


def run_judging(cfg_path: Path) -> None:
    cfg = load_config(cfg_path)

    predictions_csv = Path(cfg["predictions_csv"])
    id_col = cfg["id_column"]
    context_col = cfg["context_column"]
    pred_col = cfg["prediction_column"]
    ref_col = cfg["reference_column"]
    output_path = Path(cfg["output_path"])

    df = pd.read_csv(predictions_csv)

    max_examples = int(cfg.get("max_examples", len(df)))
    df = df.head(max_examples)

    client = build_client()

    results: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        example = {
            "id": row[id_col],
            "context": row.get(context_col, ""),
            "prediction": row[pred_col],
            "reference": row.get(ref_col, ""),
        }
        scores = score_example(client, example, cfg)
        results.append({"id": example["id"], "scores": scores})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"LLM-as-a-judge results written to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM-as-a-judge evaluation.")
    parser.add_argument(
        "--config",
        default="v1/code/llm_as_a_judge_config.json",
        help="Path to LLM-as-a-judge configuration JSON.",
    )
    args = parser.parse_args()

    run_judging(Path(args.config))


if __name__ == "__main__":
    main()
