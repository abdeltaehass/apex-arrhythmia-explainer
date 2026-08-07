"""Phase 21 — retrieval-augmented clinical context for explanation generation."""

from src.rag.corpus import Passage, build_corpus, load_corpus
from src.rag.index import DenseIndex, Hit, HybridIndex, TfidfIndex, build_indexes, load_index
from src.rag.retrieve import (
    CONTEXT_INSTRUCTION,
    RetrievedContext,
    build_query,
    context_condition_names,
    format_context,
    retrieve_for_findings,
)

__all__ = [
    "CONTEXT_INSTRUCTION",
    "DenseIndex",
    "HybridIndex",
    "Hit",
    "Passage",
    "RetrievedContext",
    "TfidfIndex",
    "build_corpus",
    "build_indexes",
    "build_query",
    "context_condition_names",
    "format_context",
    "load_corpus",
    "load_index",
    "retrieve_for_findings",
]
