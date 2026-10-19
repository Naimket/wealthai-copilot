"""Dense indexing and search: fastembed (local, CPU) + Qdrant in local mode."""
from dataclasses import asdict
from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

from wealthai.ingestion.chunk import Chunk

ROOT = Path(__file__).resolve().parents[3]  # repository root
QDRANT_PATH = str(ROOT / "data" / "qdrant")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION = "kid_chunks"
MACRO_COLLECTION = "macro_chunks"

_embedder: TextEmbedding | None = None


def embedder() -> TextEmbedding:
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(model_name=EMBED_MODEL)
    return _embedder


def build_index(chunks: list[Chunk], collection: str = COLLECTION) -> int:
    vectors = [v.tolist() for v in embedder().embed([c.text for c in chunks])]
    client = QdrantClient(path=QDRANT_PATH)
    try:
        if client.collection_exists(collection):
            client.delete_collection(collection)
        client.create_collection(
            collection,
            vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE),
        )
        client.upsert(
            collection,
            points=[
                models.PointStruct(id=i, vector=vec, payload=asdict(chunk))
                for i, (chunk, vec) in enumerate(zip(chunks, vectors))
            ],
        )
        return len(chunks)
    finally:
        client.close()


def search(query: str, k: int = 5):
    vector = next(iter(embedder().query_embed(query))).tolist()
    client = QdrantClient(path=QDRANT_PATH)
    try:
        return client.query_points(COLLECTION, query=vector, limit=k).points
    finally:
        client.close()
