"""Run a deterministic retrieval baseline for the RAG assistant.

Run from the repository root with:

    ./.venv/bin/python -m evals.run_evals

Live answer generation is opt-in because it requires an OpenAI API key:

    ./.venv/bin/python -m evals.run_evals --live-generation
"""

import argparse
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import rag_engine


ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "evals" / "dataset.json"
RESULTS_DIR = ROOT / "evals" / "results"


class DeterministicEmbeddingModel:
    """Small local embedding substitute for repeatable retrieval checks."""

    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.index = {term: index for index, term in enumerate(vocabulary)}

    def __call__(self, texts):
        vectors = []
        for text in texts:
            counts = Counter(_tokens(text))
            vector = np.array(
                [counts.get(term, 0) for term in self.vocabulary],
                dtype=float,
            )
            norm = np.linalg.norm(vector)
            vectors.append(vector / norm if norm else vector)
        return EmbeddingResult(np.array(vectors))


class EmbeddingResult:
    def __init__(self, values):
        self.values = values

    def numpy(self):
        return self.values

    def __array__(self, dtype=None):
        return np.asarray(self.values, dtype=dtype)


def _tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def load_cases():
    with DATASET_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def build_index():
    raw_text = rag_engine.load_text(ROOT / "knowledge.txt")
    cleaned_text = rag_engine.clean_text(raw_text)
    chunks = rag_engine.chunk_text(cleaned_text, chunk_size=80, overlap=20)
    vocabulary = sorted(set(_tokens(cleaned_text)))
    model = DeterministicEmbeddingModel(vocabulary)
    vectors = model(chunks).numpy()
    dataframe = pd.DataFrame({"text_chunks": chunks, "vectors": list(vectors)})
    return dataframe, model


def evaluate_case(case, dataframe, model, live_generation=False):
    context, similarity = rag_engine.search_index(
        dataframe,
        case["question"],
        model,
        top_k=3,
    )
    normalized_context = context.lower()
    expected_terms = case["expected_terms"]
    matched_terms = [
        term for term in expected_terms if term.lower() in normalized_context
    ]
    retrieval_hit = bool(matched_terms) if expected_terms else None

    result = {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "expected_behavior": case["expected_behavior"],
        "retrieved_context": context,
        "similarity_score": float(similarity),
        "expected_terms": expected_terms,
        "matched_terms": matched_terms,
        "retrieval_hit": retrieval_hit,
        "answer": None,
        "grounded": "not_evaluated",
        "citation_correct": "not_evaluated",
        "abstention_correct": "not_evaluated",
        "refusal_correct": "not_evaluated",
    }

    if live_generation:
        answer = rag_engine.generate_answer(case["question"], context)
        result["answer"] = answer

    return result


def summarize(results, live_generation):
    retrieval_cases = [result for result in results if result["retrieval_hit"] is not None]
    hits = sum(result["retrieval_hit"] for result in retrieval_cases)
    summary = {
        "total_cases": len(results),
        "retrieval_cases": len(retrieval_cases),
        "retrieval_hits": hits,
        "retrieval_hit_rate": hits / len(retrieval_cases) if retrieval_cases else None,
        "generation_evaluated": live_generation,
        "grounded_cases_evaluated": 0,
        "citation_cases_evaluated": 0,
        "abstention_cases_evaluated": 0,
        "refusal_cases_evaluated": 0,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live-generation",
        action="store_true",
        help="Call the configured OpenAI model for each case.",
    )
    args = parser.parse_args()

    if args.live_generation and not os.getenv("OPENAI_API_KEY"):
        parser.error("--live-generation requires OPENAI_API_KEY")

    cases = load_cases()
    dataframe, model = build_index()
    results = [
        evaluate_case(case, dataframe, model, args.live_generation)
        for case in cases
    ]
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summarize(results, args.live_generation),
        "results": results,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    output_path = RESULTS_DIR / "latest.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print(json.dumps(output["summary"], indent=2))
    print(f"Detailed results: {output_path}")


if __name__ == "__main__":
    main()