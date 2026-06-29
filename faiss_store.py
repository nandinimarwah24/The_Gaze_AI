import json
from pathlib import Path

import faiss
import numpy as np


DATA_DIR = Path(__file__).resolve().parent / "data"
CHUNKS_PATH = DATA_DIR / "chunks.json"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
INDEX_PATH = DATA_DIR / "faiss.index"


def build_faiss_index(embeddings, index_path=INDEX_PATH):
    """Build and persist a cosine-similarity FAISS index."""
    vectors = np.asarray(embeddings, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError("Embeddings must be a non-empty 2D array.")

    # Inner product between L2-normalized vectors is cosine similarity.
    vectors = np.ascontiguousarray(vectors)
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    return index


def load_faiss_store(
    chunks_path=CHUNKS_PATH,
    index_path=INDEX_PATH,
    embeddings_path=EMBEDDINGS_PATH,
):
    """Load chunks and their FAISS index, building the index if necessary."""
    chunks_path = Path(chunks_path)
    index_path = Path(index_path)
    embeddings_path = Path(embeddings_path)

    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Missing {chunks_path}. Run 'python rag.py' to ingest the document."
        )

    with chunks_path.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not index_path.exists():
        if not embeddings_path.exists():
            raise FileNotFoundError(
                f"Missing {index_path} and {embeddings_path}. Run 'python rag.py'."
            )
        embeddings = np.load(embeddings_path)
        index = build_faiss_index(embeddings, index_path)
    else:
        index = faiss.read_index(str(index_path))

    if index.ntotal != len(chunks):
        raise ValueError(
            f"FAISS contains {index.ntotal} vectors, but there are "
            f"{len(chunks)} chunks. Run 'python rag.py' to rebuild the data."
        )

    return chunks, index


def search_faiss(query, model, chunks, index, top_k=3):
    """Return the chunks most similar to the supplied query."""
    if not query.strip():
        return []
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    query_vector = model.encode([query], convert_to_numpy=True)
    query_vector = np.ascontiguousarray(query_vector, dtype=np.float32)
    faiss.normalize_L2(query_vector)

    result_count = min(top_k, index.ntotal)
    scores, indices = index.search(query_vector, result_count)

    results = []
    for score, chunk_index in zip(scores[0], indices[0]):
        if chunk_index < 0:
            continue

        stored_chunk = chunks[int(chunk_index)]
        # Support both the new metadata format and older string-only indexes.
        if isinstance(stored_chunk, dict):
            results.append(
                {
                    "chunk": stored_chunk["text"],
                    "score": float(score),
                    "index": stored_chunk.get("chunk_id", int(chunk_index)),
                    "source": stored_chunk.get("source", "Unknown"),
                    "page": stored_chunk.get("page", "Unknown"),
                }
            )
        else:
            results.append(
                {
                    "chunk": stored_chunk,
                    "score": float(score),
                    "index": int(chunk_index),
                    "source": "Unknown",
                    "page": "Unknown",
                }
            )
    return results
