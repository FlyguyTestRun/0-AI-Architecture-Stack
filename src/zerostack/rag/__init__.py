"""RAG pipeline (layer 3)."""

from zerostack.rag.chunking import Chunk, chunk_text
from zerostack.rag.embeddings import (
    EmbeddingProvider,
    HashingEmbeddings,
    build_embeddings,
)
from zerostack.rag.pipeline import IngestReport, RAGPipeline, build_vector_store
from zerostack.rag.store import (
    ChromaVectorStore,
    MemoryVectorStore,
    QdrantVectorStore,
    SearchResult,
    VectorStore,
)

__all__ = [
    "Chunk",
    "ChromaVectorStore",
    "EmbeddingProvider",
    "HashingEmbeddings",
    "IngestReport",
    "MemoryVectorStore",
    "QdrantVectorStore",
    "RAGPipeline",
    "SearchResult",
    "VectorStore",
    "build_embeddings",
    "build_vector_store",
    "chunk_text",
]
