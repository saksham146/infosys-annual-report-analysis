import argparse
from pathlib import Path

import chromadb

from src.Embed import generate_embeddings
from src.rag_pipeline import COLLECTION_NAME, DB_PATH


def setup_vector_database(rebuild=False, batch_size=128):
    """Build or update the persistent report index in bounded batches."""
    chunks = generate_embeddings()
    if not chunks:
        raise RuntimeError("No report text was extracted; no index was written.")

    DB_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_PATH))
    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"embedding_model": "all-MiniLM-L6-v2"},
    )

    for offset in range(0, len(chunks), batch_size):
        batch = chunks[offset : offset + batch_size]
        collection.upsert(
            ids=[f"page_{chunk['page']}_chunk_{offset + index}" for index, chunk in enumerate(batch)],
            documents=[chunk["text"] for chunk in batch],
            metadatas=[{"page": int(chunk["page"])} for chunk in batch],
            embeddings=[chunk["vector"] for chunk in batch],
        )
        print(f"Indexed {min(offset + len(batch), len(chunks)):,}/{len(chunks):,} passages")
    print(f"Index ready: {collection.count():,} passages in {DB_PATH}")
    return collection


def main():
    parser = argparse.ArgumentParser(description="Create the local Infosys report search index.")
    parser.add_argument("--rebuild", action="store_true", help="Replace the existing collection after embeddings are ready.")
    args = parser.parse_args()
    setup_vector_database(rebuild=args.rebuild)


if __name__ == "__main__":
    main()
