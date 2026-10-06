"""
src/retrieval/dense_retriever.py

Dense vector retrieval powered by SentenceTransformers and Qdrant in-memory client.
Default model: 'all-MiniLM-L6-v2' (lightweight, high speed, 384 dimensions).
"""

from __future__ import annotations

import os
from typing import List, Tuple
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.ingestion.chunker import Chunk

COLLECTION_NAME = "domain_documents"


class DenseRetriever:
    """Dense vector retriever using cosine similarity over neural embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        logger.info(f"Loading embedding model: {model_name} ...")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # In-memory Qdrant client (can point to persistent path or Qdrant server)
        self.client = QdrantClient(":memory:")
        self.chunks_map: dict[str, Chunk] = {}
        self._init_collection()

    def _init_collection(self) -> None:
        self.client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE),
        )

    def index(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return
        self._init_collection()
        self.chunks_map = {c.chunk_id: c for c in chunks}
        
        texts = [c.text for c in chunks]
        logger.info(f"Generating dense embeddings for {len(texts)} chunks ...")
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

        points = [
            PointStruct(
                id=i,
                vector=embeddings[i].tolist(),
                payload={"chunk_id": c.chunk_id},
            )
            for i, c in enumerate(chunks)
        ]
        self.client.upsert(collection_name=COLLECTION_NAME, points=points)
        logger.info(f"Indexed {len(points)} vector points into Qdrant collection '{COLLECTION_NAME}'.")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        if not self.chunks_map:
            return []
        query_vector = self.model.encode(query, convert_to_numpy=True, normalize_embeddings=True).tolist()
        hits = self.client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k,
        )
        results: List[Tuple[Chunk, float]] = []
        for hit in hits:
            chunk_id = hit.payload["chunk_id"]
            if chunk_id in self.chunks_map:
                results.append((self.chunks_map[chunk_id], float(hit.score)))
        return results
