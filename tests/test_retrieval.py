"""
tests/test_retrieval.py

Unit tests for BM25, Dense Vector, and Hybrid Reciprocal Rank Fusion retrieval.
"""

from __future__ import annotations

import pytest
from src.ingestion.chunker import Chunk, HierarchicalSectionChunker
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever


@pytest.fixture(scope="module")
def sample_chunks():
    c1 = Chunk(
        chunk_id="doc1_sec1",
        doc_id="doc1",
        section_id="sec_rev",
        header="Item 7. Revenue Recognition",
        text="Remaining performance obligations totaled $4.82 billion as of December 31, 2024.",
        char_length=80,
        strategy="test",
    )
    c2 = Chunk(
        chunk_id="doc1_sec2",
        doc_id="doc1",
        section_id="sec_debt",
        header="Item 8. Note 9 - Debt Facilities",
        text="Consolidated net leverage ratio was 1.82 to 1.00 compared to maximum covenant of 3.50.",
        char_length=85,
        strategy="test",
    )
    c3 = Chunk(
        chunk_id="doc1_sec3",
        doc_id="doc1",
        section_id="sec_capex",
        header="Item 7. Capital Expenditures",
        text="Capital expenditures for fiscal 2024 were $1.74 billion for GPU cluster expansions.",
        char_length=83,
        strategy="test",
    )
    return [c1, c2, c3]


def test_bm25_exact_match(sample_chunks):
    retriever = BM25Retriever()
    retriever.index(sample_chunks)
    hits = retriever.search("What were the capital expenditures for fiscal 2024?", top_k=1)
    assert len(hits) == 1
    assert hits[0][0].section_id == "sec_capex"


def test_dense_semantic_match(sample_chunks):
    retriever = DenseRetriever()
    retriever.index(sample_chunks)
    hits = retriever.search("How much long term borrowing headroom or debt leverage do we have?", top_k=1)
    assert len(hits) == 1
    assert hits[0][0].section_id == "sec_debt"


def test_hybrid_rrf_scoring(sample_chunks):
    hybrid = HybridRetriever(reranker_model_name=None)
    hybrid.index(sample_chunks)
    hits = hybrid.retrieve("remaining performance obligations and multi-year subscription backlog", top_k=2)
    assert len(hits) >= 1
    top_chunk, score, meta = hits[0]
    assert top_chunk.section_id == "sec_rev"
    assert "rrf_score" in meta
