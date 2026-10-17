"""
src/api/schemas.py

Pydantic models for the Domain RAG & Evaluation API.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Financial or technical query")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of context chunks to retrieve")
    use_reranker: bool = Field(default=True, description="Enable Cross-Encoder reranking")

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "What are CloudScale's remaining performance obligations and 12-month recognition percentage?",
                "top_k": 3,
                "use_reranker": True,
            }
        }
    }


class CitationResponse(BaseModel):
    chunk_id: str
    section_id: str
    header: str
    snippet: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: List[CitationResponse]
    retrieved_chunk_count: int
    estimated_tokens: int
    latency_ms: float
    model_name: str


class BenchmarkSummaryResponse(BaseModel):
    configuration: str
    total_queries: int
    hit_rate_at_k: str
    mrr: str
    context_recall: str
    faithfulness: str
    hallucination_rate: str
    p99_latency: str
    cost_per_1k_queries: str


class HealthResponse(BaseModel):
    status: str
    indexed_chunks: int
    embedding_model: str
    reranker_model: str
    version: str
