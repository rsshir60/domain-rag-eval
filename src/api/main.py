"""
src/api/main.py

FastAPI application for Domain-Specific RAG and automated evaluation.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.schemas import (
    BenchmarkSummaryResponse,
    CitationResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
)
from src.evaluation.evaluator import RAGEvaluator
from src.generation.generator import GroundedGenerator
from src.ingestion.chunker import get_chunker
from src.retrieval.hybrid_retriever import HybridRetriever

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_CORPUS = ROOT_DIR / "data" / "corpus" / "sample_10k_filing.json"
GOLD_QA = ROOT_DIR / "data" / "gold_eval" / "qa_pairs.json"

APP_VERSION = "1.0.0"

_retriever: HybridRetriever | None = None
_generator: GroundedGenerator | None = None
_evaluator: RAGEvaluator | None = None
_chunk_count: int = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _retriever, _generator, _evaluator, _chunk_count
    logger.info("Initializing Domain-Specific RAG pipeline ...")

    # 1. Ingestion
    chunker = get_chunker("hierarchical_section")
    with open(DATA_CORPUS, "r", encoding="utf-8") as f:
        docs = json.load(f)
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunker.chunk_document(doc))
    _chunk_count = len(all_chunks)

    # 2. Retrieval & Generator
    _retriever = HybridRetriever()
    _retriever.index(all_chunks)
    _generator = GroundedGenerator()

    # 3. Evaluator
    _evaluator = RAGEvaluator(_retriever, _generator, GOLD_QA)
    logger.info(f"Pipeline ready with {_chunk_count} indexed chunks.")
    yield
    logger.info("RAG Service shutting down.")


app = FastAPI(
    title="Domain-Specific RAG Engine with Automated Evaluation",
    description="Hybrid Dense+BM25 with Cross-Encoder Reranking and Quantitative Ragas-style Benchmarking",
    version=APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health():
    return HealthResponse(
        status="ok",
        indexed_chunks=_chunk_count,
        embedding_model="all-MiniLM-L6-v2",
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        version=APP_VERSION,
    )


@app.post("/rag/query", response_model=QueryResponse, tags=["inference"])
async def query_rag(req: QueryRequest):
    """
    Answers a complex financial/technical query with verifiable citations
    derived from hybrid retrieval + cross-encoder reranking.
    """
    if _retriever is None or _generator is None:
        raise HTTPException(status_code=500, detail="RAG system not initialized")

    retrieved = _retriever.retrieve(
        query=req.query,
        top_k=req.top_k,
        use_reranker=req.use_reranker,
    )
    chunks = [c for c, _, _ in retrieved]
    result = _generator.generate(req.query, chunks)

    citations = [
        CitationResponse(
            chunk_id=c.chunk_id,
            section_id=c.section_id,
            header=c.header,
            snippet=c.snippet,
        )
        for c in result.citations
    ]

    return QueryResponse(
        question=result.question,
        answer=result.answer,
        citations=citations,
        retrieved_chunk_count=len(chunks),
        estimated_tokens=result.total_tokens_estimated,
        latency_ms=result.latency_ms,
        model_name=result.model_name,
    )


@app.post("/rag/benchmark", response_model=BenchmarkSummaryResponse, tags=["evaluation"])
async def run_benchmark(config_name: str = "Hybrid_Dense_BM25_Reranker", top_k: int = 3):
    """
    Triggers automated quantitative benchmarking over the gold SEC Q&A evaluation dataset:
    Measures Hit Rate@k, MRR, Context Recall, Faithfulness, P99 Latency, and Token Cost.
    """
    if _evaluator is None:
        raise HTTPException(status_code=500, detail="Evaluator not initialized")

    report = _evaluator.benchmark_system(
        config_name=config_name,
        top_k=top_k,
        use_reranker=True,
        save_report=True,
    )
    s = report.summary_dict()

    return BenchmarkSummaryResponse(
        configuration=s["Configuration"],
        total_queries=report.total_queries,
        hit_rate_at_k=s["Hit Rate@k"],
        mrr=s["MRR"],
        context_recall=s["Context Recall"],
        faithfulness=s["Faithfulness"],
        hallucination_rate=s["Hallucination Rate"],
        p99_latency=s["P99 Latency"],
        cost_per_1k_queries=s["Cost/1k Queries"],
    )
