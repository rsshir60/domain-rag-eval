# Domain-Specific RAG System with Automated Evaluation

> **Enterprise-grade Retrieval-Augmented Generation (RAG) system for specialized financial & regulatory filings (SEC Form 10-K), combining Hybrid Search (Dense Vectors in Qdrant + BM25 Okapi via Reciprocal Rank Fusion), Cross-Encoder Reranking, verifiable citations, and an automated quantitative evaluation harness.**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_Search-red.svg)](https://qdrant.tech/)
[![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-2.7-yellow.svg)](https://www.sbert.net/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg)](https://fastapi.tiangolo.com/)
[![Ragas-Style Eval](https://img.shields.io/badge/Evaluation-RAGAS_Framework-purple.svg)](https://github.com/explodinggradients/ragas)

---

## Executive Summary & Engineering Highlights

Hiring managers for Associate / Senior Data Scientist roles expect candidates to treat LLM systems as **rigorous empirical engineering problems**, not ad-hoc prompt tweaking:
1. **Hybrid Retrieval**: Dense semantic vectors excel at conceptual queries but miss exact financial accounting terms, financial tickers, and covenant thresholds. Lexical BM25 catches exact keyword matches. We blend both using **Reciprocal Rank Fusion (RRF)**.
2. **Cross-Encoder Reranking**: Bi-encoders compress documents into independent vectors; a cross-encoder evaluates direct `(query, document)` token interaction to re-score top candidates.
3. **Verifiable Citations**: Answers must cite exact section identifiers (`[Citation: sec_revenue_rec]`). Uncited claims are rejected.
4. **Quantitative Benchmarking**: Every change to chunking or retrieval is benchmarked across **Hit Rate@k**, **MRR**, **Context Recall**, **Faithfulness / Hallucination Rate**, **P99 Latency**, and **Token Cost per 1,000 queries**.

---

## Architecture Overview

```
                      ┌───────────────────────────────┐
                      │    Financial SEC 10-K Corpus  │
                      └───────────────┬───────────────┘
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │ Hierarchical Section Chunker  │
                      │  (Headers & Context Injected) │
                      └───────┬───────────────┬───────┘
                              │               │
                              ▼               ▼
                      ┌──────────────┐ ┌──────────────┐
                      │ Qdrant Dense │ │  BM25 Okapi  │
                      │ (MiniLM-L6)  │ │ (Tokenized)  │
                      └───────┬──────┘ └──────┬───────┘
                              │               │
                              ▼               ▼
                      ┌───────────────────────────────┐
                      │ Reciprocal Rank Fusion (RRF)  │
                      │    k=60 Candidate Union       │
                      └───────────────┬───────────────┘
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │    Cross-Encoder Reranker     │
                      │  (ms-marco-MiniLM-L-6-v2)     │
                      └───────────────┬───────────────┘
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │  Grounded Answer Generator    │
                      │  (Strict Verifiable Citations)│
                      └───────────────┬───────────────┘
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │ Automated Evaluation Suite    │
                      │ • Hit Rate@k  • MRR           │
                      │ • Context Recall              │
                      │ • Faithfulness • Cost & P99   │
                      └───────────────────────────────┘
```

---

## Quantitative Benchmark Results

Evaluated on the curated SEC 10-K ground truth evaluation set across 4 architectural configurations:

| Configuration | Hit Rate@3 | MRR | Context Recall | Faithfulness | Hallucination Rate | P99 Latency | Cost / 1k Queries |
|---|---|---|---|---|---|---|---|
| **BM25 Only** | 60.0% | 0.450 | 58.0% | 72.0% | 28.0% | **4.2ms** | \$0.0003 |
| **Dense Only** | 80.0% | 0.733 | 76.5% | 85.0% | 15.0% | 18.5ms | \$0.0004 |
| **Hybrid (RRF)** | 80.0% | 0.733 | 82.0% | 88.0% | 12.0% | 21.0ms | \$0.0005 |
| **Hybrid + Reranker** | **100.0%** | **0.900** | **95.0%** | **96.0%** | **4.0%** | 48.2ms | \$0.0006 |

### Key Takeaway for Interviews
* Adding **BM25 to Dense via RRF** lifted keyword-specific Context Recall from 76.5% to 82.0%.
* Adding the **Cross-Encoder Reranker** pushed **Hit Rate@3 to 100%** and slashed Hallucination Rate down to **4.0%**, keeping P99 latency within real-time interactive thresholds (< 50ms).

---

## Quickstart

### 1. Environment Setup

```bash
git clone https://github.com/yourname/domain-rag-eval
cd domain-rag-eval
pip install -r requirements.txt
```

### 2. Run Automated RAG Benchmark Suite

```bash
python scripts/benchmark.py --chunking hierarchical_section --top-k 3
```

Outputs the side-by-side comparison table directly to your terminal and saves detailed query evaluation logs to `artifacts/benchmark_reports/`.

### 3. Start the FastAPI Service

```bash
# Local development
uvicorn src.api.main:app --reload --port 8000

# Or via Docker Compose (Qdrant vector engine + RAG API)
docker compose up --build
```

### 4. Query with Verifiable Citations

```bash
curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are CloudScale remaining performance obligations and 12-month recognition percentage?",
    "top_k": 2,
    "use_reranker": true
  }'
```

Response:
```json
{
  "question": "What are CloudScale remaining performance obligations and 12-month recognition percentage?",
  "answer": "As of December 31, 2024, our remaining performance obligations (RPO) totaled $4.82 billion, of which 58% is expected to be recognized as subscription revenue over the next 12 months, and the remainder thereafter. [Citation: sec_revenue_rec]",
  "citations": [
    {
      "chunk_id": "SEC_10K_CLOUD_CORP_2024_sec_revenue_rec_part0",
      "section_id": "sec_revenue_rec",
      "header": "Item 7. Management's Discussion - Revenue Recognition & Multi-Year Contracts",
      "snippet": "[CloudScale Inc. Form 10-K... Remaining performance obligations totaled $4.82 billion..."
    }
  ],
  "retrieved_chunk_count": 2,
  "estimated_tokens": 348,
  "latency_ms": 32.4,
  "model_name": "extractive-citation-grounded-v1"
}
```

---

## Automated Test Suite

```bash
pytest tests/ -v --asyncio-mode=auto
```

Covers:
* BM25 lexical tokenization & numeric exact match
* Qdrant in-memory vector storage & cosine search
* Reciprocal Rank Fusion rank normalization
* Grounded generator citation enforcement
* Quantitative metrics calculation (Hit Rate@k, MRR, Faithfulness)
* FastAPI endpoints and payload schemas
