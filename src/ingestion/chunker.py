"""
src/ingestion/chunker.py

Chunking strategies for document ingestion and comparative RAG benchmarking:
  1. FixedWindowChunker: Standard fixed token/character length with sliding overlap.
  2. HierarchicalSectionChunker: Preserves Section headers + document context in each chunk.
  3. SentenceWindowChunker: Sentence-level atomicity with adjacent context expansion.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, List


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    section_id: str
    header: str
    text: str
    char_length: int
    strategy: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HierarchicalSectionChunker:
    """
    Financial 10-K & technical manual optimal strategy:
    Prepends document header metadata to every chunk to avoid context loss in retrieval.
    """

    def __init__(self, max_chars: int = 400, overlap_chars: int = 50):
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
        self.name = "hierarchical_section"

    def chunk_document(self, doc: dict) -> List[Chunk]:
        chunks: List[Chunk] = []
        doc_id = doc.get("doc_id", "unknown")
        title = doc.get("title", "")

        for section in doc.get("sections", []):
            sec_id = section.get("section_id", "sec")
            header = section.get("header", "")
            raw_text = section.get("content", "")

            # Split by paragraph or character boundary
            start = 0
            part = 0
            while start < len(raw_text):
                end = min(start + self.max_chars, len(raw_text))
                # Try not to split inside a word
                if end < len(raw_text):
                    last_space = raw_text.rfind(" ", start, end)
                    if last_space > start:
                        end = last_space

                slice_text = raw_text[start:end].strip()
                # Prepend section metadata to give dense embeddings rich contextual signal
                augmented_text = f"[{title} | {header}]\n{slice_text}"
                
                chunk_id = f"{doc_id}_{sec_id}_part{part}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        section_id=sec_id,
                        header=header,
                        text=augmented_text,
                        char_length=len(augmented_text),
                        strategy=self.name,
                    )
                )
                if end >= len(raw_text):
                    break
                start = end - self.overlap_chars
                part += 1

        return chunks


class FixedWindowChunker:
    """Standard baseline fixed character window chunker."""

    def __init__(self, chunk_size: int = 250, chunk_overlap: int = 40):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.name = "fixed_window"

    def chunk_document(self, doc: dict) -> List[Chunk]:
        chunks: List[Chunk] = []
        doc_id = doc.get("doc_id", "unknown")

        for section in doc.get("sections", []):
            sec_id = section.get("section_id", "sec")
            raw_text = section.get("content", "")

            start = 0
            part = 0
            while start < len(raw_text):
                end = min(start + self.chunk_size, len(raw_text))
                slice_text = raw_text[start:end].strip()
                chunk_id = f"{doc_id}_{sec_id}_fixed_{part}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        section_id=sec_id,
                        header=section.get("header", ""),
                        text=slice_text,
                        char_length=len(slice_text),
                        strategy=self.name,
                    )
                )
                if end >= len(raw_text):
                    break
                start = end - self.chunk_overlap
                part += 1

        return chunks


def get_chunker(strategy: str = "hierarchical_section"):
    if strategy == "hierarchical_section":
        return HierarchicalSectionChunker()
    elif strategy == "fixed_window":
        return FixedWindowChunker()
    else:
        raise ValueError(f"Unknown chunking strategy '{strategy}'")
