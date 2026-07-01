# LLM Evaluation & Hallucination Detection Framework

A multi-metric evaluation system that automatically detects hallucinations, measures faithfulness, relevance, and fluency in LLM-generated responses — built for QA pipelines in RAG and LLM-powered systems.

## Problem

LLMs frequently generate fluent, confident, but factually incorrect responses ("hallucinations") with no built-in way to detect this. This is a critical risk in production systems — healthcare Q&A, legal summarization, customer support chatbots, RAG pipelines — where a wrong but confident answer can cause real harm.

This framework evaluates any `(context, question, llm_response)` triple and returns a verdict: **Faithful**, **Hallucinated**, **Irrelevant**, or **Unverifiable** — backed by 4 independent scoring methods.

## Why 4 Metrics, Not 1

Each metric has a blind spot the others cover:

| Metric | Catches | Misses |
|---|---|---|
| **NLI** (`cross-encoder/nli-deberta-v3-small`) | Direct factual contradictions | Whether the question was actually answered |
| **BERTScore** | Overall semantic drift from context | Subtle single-fact contradictions (high lexical overlap masks them) |
| **Cosine Similarity** (`all-MiniLM-L6-v2`) | Whether response is relevant to the question | Factual correctness |
| **Fluency** (rule-based) | Grammatical/structural quality | Meaning entirely |

Example that demonstrates this in practice — tested live in this project:

```
Context:  "Virat Kohli won 2 IPL trophies as a player and 0 as captain."
Response: "Virat Kohli won 25 IPL trophies."

BERTScore: 0.89 → "Highly Faithful"  ❌ (fooled by lexical overlap)
NLI:       contradiction (0.999)      ✅ (catches the numeric hallucination)

Final verdict: Hallucinated (correct)
```

This is the core finding the project is built around: no single metric is reliable alone.

## Architecture

```
POST /evaluate { context, question, llm_response }
        │
        ▼
 ┌──────────────────────────────┐
 │   4 evaluators run in parallel│
 │  Cosine | Fluency | BERTScore | NLI │
 └──────────────────────────────┘
        │
        ▼
   Aggregator (rule-based verdict logic)
        │
        ▼
   SQLite (persisted) ──► GET /history
        │
        ▼
   Streamlit Dashboard (manual testing + history view)
```

## Tech Stack

- **FastAPI** — REST API (`/evaluate`, `/history`)
- **HuggingFace Transformers** — NLI model (`cross-encoder/nli-deberta-v3-small`)
- **bert-score** — semantic faithfulness scoring
- **sentence-transformers** — cosine similarity (`all-MiniLM-L6-v2`)
- **SQLite** — evaluation history storage
- **Streamlit** — interactive dashboard

All models are CPU-friendly — no GPU required.

## Setup

```bash
git clone <your-repo-url>
cd llm-eval-framework
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Running

**Start the API:**
```bash
uvicorn main:app --reload
```
API docs: `http://localhost:8000/docs`

**Start the dashboard** (separate terminal):
```bash
streamlit run dashboard/app.py
```
Dashboard: `http://localhost:8501`

## API

### `POST /evaluate`

```json
{
  "context": "Photosynthesis converts sunlight into glucose.",
  "question": "How do plants make food?",
  "llm_response": "Plants use moonlight to produce glucose."
}
```

Response:
```json
{
  "final_verdict": "Hallucinated",
  "cosine": {"score": 0.52, "verdict": "Partially Relevant"},
  "fluency": {"issues": [], "verdict": "Fluent"},
  "bert_score": {"score": 0.91, "verdict": "Highly Faithful"},
  "nli": {"label": "contradiction", "score": 0.999, "verdict": "Hallucinated"}
}
```

### `GET /history`
Returns all past evaluations with verdicts and timestamps.

## Project Structure

```
llm-eval-framework/
├── main.py                          # FastAPI app
├── src/
│   ├── aggregator.py                # Combines all 4 scores → final verdict
│   ├── database.py                  # SQLite persistence
│   └── evaluators/
│       ├── nli_evaluator.py
│       ├── bert_score_evaluator.py
│       ├── cosine_evaluator.py
│       └── fluency_evaluator.py
├── dashboard/
│   └── app.py                       # Streamlit UI
├── data/                            # SQLite DB
└── requirements.txt
```

## Known Limitations

- Verdict aggregation uses fixed thresholds (not learned/calibrated) — a clear next step would be calibrating thresholds against a labeled hallucination dataset.
- Models are English-only; no language detection guard yet.
- Evaluators run sequentially, not batched/async — fine for single requests, would need optimization for high-throughput production use.
- No authentication on the API — would need to add this before any real external exposure.

## Author

Sharath — built as a hands-on project to learn FastAPI, NLI, and LLM evaluation techniques from first principles.