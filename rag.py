import json
from pathlib import Path

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from faiss_store import CHUNKS_PATH, EMBEDDINGS_PATH, INDEX_PATH, build_faiss_index


DATA_DIR = Path(__file__).resolve().parent / "data"
MODEL_NAME = "all-MiniLM-L6-v2"


def create_chunks_from_pdfs():
    pdf_paths = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {DATA_DIR}.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    chunks = []

    print(f"Found {len(pdf_paths)} PDF files.")
    for pdf_path in pdf_paths:
        print(f"Reading: {pdf_path.name}")
        reader = PdfReader(str(pdf_path))
        file_chunk_count = 0

        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text()
            if not page_text or not page_text.strip():
                continue

            for chunk_text in splitter.split_text(page_text):
                chunks.append(
                    {
                        "text": chunk_text,
                        "source": pdf_path.name,
                        "page": page_number,
                        "chunk_id": len(chunks),
                    }
                )
                file_chunk_count += 1

        print(
            f"  Pages: {len(reader.pages)} | Chunks created: {file_chunk_count}"
        )

    if not chunks:
        raise ValueError("No readable text could be extracted from the PDF files.")
    return chunks


def main():
    chunks = create_chunks_from_pdfs()
    chunk_texts = [chunk["text"] for chunk in chunks]
    print(f"\nTotal chunks created: {len(chunks)}")

    print("\nGenerating embeddings...")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        chunk_texts,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    with CHUNKS_PATH.open("w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)
    np.save(EMBEDDINGS_PATH, embeddings)
    index = build_faiss_index(embeddings)

    print(f"\nSaved metadata to {CHUNKS_PATH}")
    print(f"Saved embeddings to {EMBEDDINGS_PATH}")
    print(f"Saved FAISS index to {INDEX_PATH}")
    print(f"FAISS index contains {index.ntotal} vectors.")


if __name__ == "__main__":
    main()
