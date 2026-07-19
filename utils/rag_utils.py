import os
import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss

KB_DIR        = os.path.join("data", "knowledge_base")
INDEX_PATH    = os.path.join("models", "rag", "faiss_index.bin")
CHUNKS_PATH   = os.path.join("models", "rag", "chunks.pkl")
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"

_index        = None
_chunks       = None
_embedder     = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        print("Loading sentence embedder...")
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def load_knowledge_base() -> list[str]:
    """
    Read all .txt files from knowledge base folder.
    Split into chunks of ~3 sentences each.
    """
    chunks = []
    kb_path = Path(KB_DIR)

    if not kb_path.exists():
        raise FileNotFoundError(f"Knowledge base folder not found: {KB_DIR}")

    for txt_file in kb_path.glob("*.txt"):
        text = txt_file.read_text(encoding="utf-8").strip()
        # Split by period + newline or double newline
        sentences = [s.strip() for s in text.replace("\n\n", ".\n").split("\n") if s.strip()]
        # Group into chunks of 2 sentences
        for i in range(0, len(sentences), 2):
            chunk = " ".join(sentences[i:i+2])
            if len(chunk) > 30:   # skip very short chunks
                chunks.append(chunk)

    print(f"Loaded {len(chunks)} chunks from {len(list(kb_path.glob('*.txt')))} files")
    return chunks


def build_index():
    """
    Build FAISS vector index from knowledge base.
    Run this once — saves index to models/rag/
    """
    os.makedirs(os.path.join("models", "rag"), exist_ok=True)

    chunks   = load_knowledge_base()
    embedder = _get_embedder()

    print("Embedding knowledge base...")
    embeddings = embedder.encode(chunks, show_progress_bar=True, batch_size=32)
    embeddings = np.array(embeddings, dtype=np.float32)

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    # Build flat index (exact search — fine for small KB)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # Save
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Index built: {index.ntotal} vectors, dimension {dimension}")
    print(f"Saved to {INDEX_PATH}")
    return index, chunks


def load_index():
    """Load saved FAISS index and chunks."""
    global _index, _chunks
    if _index is None:
        if not os.path.exists(INDEX_PATH):
            print("Index not found — building now...")
            _index, _chunks = build_index()
        else:
            _index  = faiss.read_index(INDEX_PATH)
            with open(CHUNKS_PATH, "rb") as f:
                _chunks = pickle.load(f)
            print(f"Index loaded: {_index.ntotal} vectors")
    return _index, _chunks


def retrieve(query: str, top_k: int = 3) -> str:
    """
    Given a query string, return the top_k most relevant
    chunks from the knowledge base as a single string.
    """
    index, chunks = load_index()
    embedder      = _get_embedder()

    query_vec = embedder.encode([query], batch_size=1)
    query_vec = np.array(query_vec, dtype=np.float32)
    faiss.normalize_L2(query_vec)

    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1 and score > 0.3:   # confidence threshold
            results.append(chunks[idx])

    return "\n".join(results) if results else ""


if __name__ == "__main__":
    # Build index on first run
    build_index()

    # Test retrieval
    test_queries = [
        "high fever chills sweating malaria",
        "frequent urination excessive thirst diabetes",
        "persistent cough blood sputum tuberculosis",
    ]

    print("\nRetrieval test:")
    print("=" * 50)
    for q in test_queries:
        print(f"\nQuery: {q}")
        result = retrieve(q, top_k=2)
        print(f"Result: {result[:200]}...")