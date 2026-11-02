from functools import lru_cache

from sentence_transformers import SentenceTransformer

from src.ingest import PDF_PATH, ingest_annual_report
from src.rag_pipeline import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


def generate_embeddings(pdf_path=PDF_PATH, batch_size=64):
    chunks = ingest_annual_report(pdf_path)
    if not chunks:
        return []
    vectors = load_embedding_model().encode(
        [chunk["text"] for chunk in chunks],
        batch_size=batch_size,
        show_progress_bar=True,
    )
    for chunk, vector in zip(chunks, vectors):
        chunk["vector"] = vector.tolist()
    return chunks
