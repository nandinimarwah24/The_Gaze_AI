import argparse
import re

from sentence_transformers import SentenceTransformer

from faiss_store import INDEX_PATH, load_faiss_store, search_faiss
from local_llm import (
    DEFAULT_LLM_MODEL,
    DEFAULT_OLLAMA_URL,
    OllamaError,
    answer_with_context,
)


EMBEDDING_MODEL = "all-MiniLM-L6-v2"
FOLLOW_UP_WORDS = {
    "it",
    "its",
    "they",
    "them",
    "their",
    "this",
    "that",
    "these",
    "those",
}


def print_sources(results):
    print("\n========== RETRIEVED SOURCES ==========\n")
    for position, result in enumerate(results, 1):
        print(f"Match #{position} (Score: {result['score']:.4f})")
        print(
            f"PDF: {result['source']} | Page: {result['page']} | "
            f"Chunk ID: {result['index']}"
        )
        print("-" * 50)
        print(result["chunk"].strip())
        print("=" * 50 + "\n")


def is_follow_up(question):
    words = set(re.findall(r"[a-zA-Z]+", question.lower()))
    return bool(words & FOLLOW_UP_WORDS) or len(words) <= 5


def build_retrieval_query(question, chat_history):
    """Add recent user context when a question appears to be a follow-up."""
    if not chat_history or not is_follow_up(question):
        return question

    previous_questions = [
        message["content"]
        for message in chat_history
        if message["role"] == "user"
    ][-2:]
    return " ".join(previous_questions + [question])


def remember_turn(chat_history, question, answer, max_turns):
    chat_history.extend(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    )
    del chat_history[: max(0, len(chat_history) - (max_turns * 2))]


def print_history(chat_history):
    if not chat_history:
        print("\nChat history is empty.")
        return

    print("\n========== SHORT-TERM MEMORY ==========\n")
    for message in chat_history:
        speaker = "You" if message["role"] == "user" else "Assistant"
        print(f"{speaker}: {message['content']}\n")


def run_rag(question, embedding_model, chunks, index, args, chat_history):
    retrieval_query = build_retrieval_query(question, chat_history)
    print(f"\nSearching documents for: '{question}'")
    results = search_faiss(
        retrieval_query,
        embedding_model,
        chunks,
        index,
        args.top_k,
    )
    if not results:
        print("No matching document chunks were found.")
        return

    print(f"Generating answer with local Ollama model '{args.llm_model}'...")
    try:
        answer = answer_with_context(
            question,
            results,
            chat_history=chat_history,
            model=args.llm_model,
            base_url=args.ollama_url,
            timeout=args.timeout,
        )
    except OllamaError as error:
        print(f"\nLLM error: {error}")
        return

    print("\n========== RAG ANSWER ==========\n")
    print(answer)
    remember_turn(chat_history, question, answer, args.memory_turns)

    if args.show_sources:
        print_sources(results)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Chat with PDFs using FAISS, Ollama, and short-term memory."
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="Question text. Omit it to start interactive chat.",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=3,
        help="Number of document chunks to retrieve (default: 3).",
    )
    parser.add_argument(
        "--memory-turns",
        type=int,
        default=5,
        help="Recent question-answer turns retained in memory (default: 5).",
    )
    parser.add_argument(
        "--llm-model",
        default=DEFAULT_LLM_MODEL,
        help=f"Ollama model name (default: {DEFAULT_LLM_MODEL}).",
    )
    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help=f"Ollama server URL (default: {DEFAULT_OLLAMA_URL}).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Seconds to wait for the local LLM (default: 120).",
    )
    parser.add_argument(
        "--show-sources",
        action="store_true",
        help="Print complete retrieved chunks after each answer.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.top_k < 1:
        raise SystemExit("--top-k must be at least 1.")
    if args.memory_turns < 1:
        raise SystemExit("--memory-turns must be at least 1.")
    if args.timeout < 1:
        raise SystemExit("--timeout must be at least 1.")

    print("Loading embedding model...")
    try:
        embedding_model = SentenceTransformer(
            EMBEDDING_MODEL,
            local_files_only=True,
        )
    except OSError as error:
        raise SystemExit(
            f"Embedding model '{EMBEDDING_MODEL}' is not available locally. "
            "Run 'python rag.py' once while connected to the internet."
        ) from error

    try:
        chunks, index = load_faiss_store()
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Error: {error}") from error

    print(f"Loaded {index.ntotal} vectors from {INDEX_PATH}.")
    chat_history = []

    if args.query:
        run_rag(
            " ".join(args.query),
            embedding_model,
            chunks,
            index,
            args,
            chat_history,
        )
        return

    print("\n--- Local PDF Chat: FAISS + Ollama ---")
    print("Commands: /history, /clear, exit")

    while True:
        try:
            question = input("\nYou > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat.")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        if question.lower() == "/clear":
            chat_history.clear()
            print("Short-term memory cleared.")
            continue
        if question.lower() == "/history":
            print_history(chat_history)
            continue

        run_rag(
            question,
            embedding_model,
            chunks,
            index,
            args,
            chat_history,
        )


if __name__ == "__main__":
    main()
