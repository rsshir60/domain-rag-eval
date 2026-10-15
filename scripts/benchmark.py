"""
scripts/benchmark.py

Executes a full comparative evaluation benchmark across RAG retrieval strategies:
  Config 1: BM25 Lexical Only
  Config 2: Dense Vector Only (all-MiniLM-L6-v2)
  Config 3: Hybrid (Dense + BM25 via RRF)
  Config 4: Hybrid + Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2)

Produces a publication-ready comparison table evaluating Hit Rate@k, MRR,
Context Recall, Faithfulness, and P99 Latency.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import typer
from loguru import logger
from tabulate import tabulate

from src.evaluation.evaluator import RAGEvaluator
from src.generation.generator import GroundedGenerator
from src.ingestion.chunker import get_chunker
from src.retrieval.hybrid_retriever import HybridRetriever

CORPUS_PATH = ROOT / "data" / "corpus" / "sample_10k_filing.json"
GOLD_PATH = ROOT / "data" / "gold_eval" / "qa_pairs.json"
REPORT_DIR = ROOT / "artifacts" / "benchmark_reports"

app = typer.Typer(add_completion=False)


@app.command()
def run_comparison(
    chunking: str = typer.Option("hierarchical_section", help="hierarchical_section or fixed_window"),
    top_k: int = typer.Option(3, help="Top-K context chunks"),
):
    logger.info(f"Loading corpus from {CORPUS_PATH} ...")
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        docs = json.load(f)

    chunker = get_chunker(chunking)
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunker.chunk_document(doc))
    logger.info(f"Generated {len(all_chunks)} chunks using strategy '{chunking}'")

    # Initialize components
    hybrid_retriever = HybridRetriever()
    hybrid_retriever.index(all_chunks)
    generator = GroundedGenerator()
    evaluator = RAGEvaluator(hybrid_retriever, generator, GOLD_PATH)

    # Run Benchmark for Configurations
    reports = []

    # Config A: BM25 Only
    logger.info("Evaluating Configuration: BM25 Lexical Only")
    class BM25Wrapper:
        def __init__(self, bm25_ret): self.bm25_ret = bm25_ret
        def retrieve(self, q, top_k=3, **kw):
            return [(c, s, {}) for c, s in self.bm25_ret.search(q, top_k=top_k)]
    eval_bm25 = RAGEvaluator(BM25Wrapper(hybrid_retriever.bm25_retriever), generator, GOLD_PATH)
    reports.append(eval_bm25.benchmark_system("BM25_Only", top_k=top_k, save_report=False))

    # Config B: Dense Only
    logger.info("Evaluating Configuration: Dense Vector Only")
    class DenseWrapper:
        def __init__(self, dense_ret): self.dense_ret = dense_ret
        def retrieve(self, q, top_k=3, **kw):
            return [(c, s, {}) for c, s in self.dense_ret.search(q, top_k=top_k)]
    eval_dense = RAGEvaluator(DenseWrapper(hybrid_retriever.dense_retriever), generator, GOLD_PATH)
    reports.append(eval_dense.benchmark_system("Dense_Vector_Only", top_k=top_k, save_report=False))

    # Config C: Hybrid (RRF without reranker)
    logger.info("Evaluating Configuration: Hybrid (Dense + BM25 RRF)")
    reports.append(evaluator.benchmark_system("Hybrid_RRF", top_k=top_k, use_reranker=False, save_report=False))

    # Config D: Hybrid + Cross-Encoder Reranker
    logger.info("Evaluating Configuration: Hybrid + Cross-Encoder Reranker")
    reports.append(evaluator.benchmark_system("Hybrid_RRF_Reranker", top_k=top_k, use_reranker=True, save_report=True))

    # Compile Comparison Table
    table_rows = [r.summary_dict() for r in reports]
    headers = list(table_rows[0].keys())
    data = [[row[h] for h in headers] for row in table_rows]

    print("\n" + "=" * 90)
    print("DOMAIN-SPECIFIC RAG ARCHITECTURE: AUTOMATED BENCHMARK REPORT")
    print(f"Corpus: SEC 10-K Filings | Chunker: {chunking} | Top-K: {top_k}")
    print("=" * 90)
    print(tabulate(data, headers=headers, tablefmt="github"))
    print("=" * 90 + "\n")

    # Save Markdown Summary
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    md_path = REPORT_DIR / "comparison_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Domain RAG Architecture Benchmark\n\n")
        f.write(f"**Chunking Strategy:** `{chunking}` | **Top-K Context:** `{top_k}`\n\n")
        f.write(tabulate(data, headers=headers, tablefmt="github"))
        f.write("\n")
    logger.info(f"Saved comparison report to {md_path}")


if __name__ == "__main__":
    app()
