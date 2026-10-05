# RAG Evaluation Baseline

This directory contains the first baseline evaluation for the cybersecurity RAG assistant. It does not change production behavior or call the Streamlit startup path.

## Run the baseline

From the repository root:

```bash
./.venv/bin/python -m evals.run_evals
```

The command uses a deterministic local embedding model to exercise the existing `rag_engine.search_index` function. It writes detailed output to `evals/results/latest.json`.

## Optional generation evaluation

Generation requires an `OPENAI_API_KEY` and makes one API request per case:

```bash
./.venv/bin/python -m evals.run_evals --live-generation
```

The current baseline records generated answers, but grounding, citation, abstention, and refusal judgments remain `not_evaluated`. Those judgments should be defined explicitly before they are automated.

## Dataset categories

The dataset includes answerable, unanswerable, ambiguous, and prompt-injection cases. `expected_terms` are retrieval checks only; they are not claims that a future generated answer must contain those exact words.