"""
tests/test_rag_pipeline.py

Integration tests for chunking, answer generation with citations, and evaluation metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.evaluation.evaluator import RAGEvaluator
from src.generation.generator import GroundedGenerator
from src.ingestion.chunker import FixedWindowChunker, HierarchicalSectionChunker
from src.retrieval.hybrid_retriever import HybridRetriever

ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "data" / "corpus" / "sample_10k_filing.json"
GOLD_PATH = ROOT / "data" / "gold_eval" / "qa_pairs.json"


@pytest.fixture(scope="module")
def loaded_docs():
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_hierarchical_chunker_structure(loaded_docs):
    chunker = HierarchicalSectionChunker(max_chars=300)
    chunks = chunker.chunk_document(loaded_docs[0])
    assert len(chunks) > 0
    # Every chunk should have section header context preserved
    for c in chunks:
        assert c.header != ""
        assert "[" in c.text and "]" in c.text


def test_grounded_generator_citations(loaded_docs):
    chunker = HierarchicalSectionChunker()
    chunks = chunker.chunk_document(loaded_docs[0])
    generator = GroundedGenerator()

    # Query with target context present
    rev_chunks = [c for c in chunks if c.section_id == "sec_revenue_rec"]
    ans = generator.generate("What is the remaining performance obligations total?", rev_chunks)

    assert ans.question != ""
    assert len(ans.citations) > 0
    assert "Citation:" in ans.answer
    assert ans.total_tokens_estimated > 0


def test_evaluator_metrics_calculation(loaded_docs):
    chunker = HierarchicalSectionChunker()
    chunks = chunker.chunk_document(loaded_docs[0])

    retriever = HybridRetriever(reranker_model_name=None)
    retriever.index(chunks)
    generator = GroundedGenerator()

    evaluator = RAGEvaluator(retriever, generator, GOLD_PATH)
    report = evaluator.benchmark_system("Test_Run", top_k=3, use_reranker=False, save_report=False)

    assert report.total_queries == 5
    assert 0.0 <= report.hit_rate_at_k <= 1.0
    assert 0.0 <= report.mrr <= 1.0
    assert 0.0 <= report.avg_faithfulness <= 1.0
    assert report.p99_latency_ms > 0
