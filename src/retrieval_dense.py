
"""Dense PDF retrieval using BGE-small-en-v1.5.

Supports all existing chunking strategies, cosine similarity,
and persistent local embedding caching.

Examples:
    python -m src.retrieval_dense "What is Engagement Range?"
    python -m src.retrieval_dense "What is Engagement Range?" --strategy hybrid
"""

import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from src.retrieval_common import (
    PROJECT_ROOT,
    CHUNK_DIR,
    CHUNK_FILES,
    load_chunks,
    format_result,
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MODEL_NAME = "BAAI/bge-small-en-v1.5"

QUERY_INSTRUCTION = (
    "Represent this sentence for searching relevant passages: "
)

CACHE_DIR = (
    PROJECT_ROOT / "data" / "processed" / "embeddings"
)


# --------------------------------------------------
# MODEL
# --------------------------------------------------

@lru_cache(maxsize=1)
def get_model():
    """Load the embedding model once per Python process."""
    return SentenceTransformer(MODEL_NAME)


# --------------------------------------------------
# EMBEDDING CACHE
# --------------------------------------------------

def collection_fingerprint(chunks, model):
    """Identify the exact text collection and embedding setup."""

    content = {
        "model": MODEL_NAME,
        "max_seq_length": model.max_seq_length,
        "chunks": [
            [chunk["id"], chunk["text"]]
            for chunk in chunks
        ],
    }

    serialized = json.dumps(
        content,
        ensure_ascii=False,
        sort_keys=True,
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()


def cache_path_for(strategy, chunk_file):
    """Choose a separate cache file for each source path."""

    source_path = (
        Path(chunk_file).resolve()
        if chunk_file is not None
        else (CHUNK_DIR / CHUNK_FILES[strategy]).resolve()
    )

    source_hash = hashlib.sha256(
        str(source_path).encode("utf-8")
    ).hexdigest()[:12]

    return CACHE_DIR / f"dense_{strategy}_{source_hash}.npz"


def get_document_embeddings(
    chunks,
    strategy,
    chunk_file=None,
    batch_size=32,
    rebuild_cache=False,
):
    """Load cached embeddings or compute them locally."""

    model = get_model()

    fingerprint = collection_fingerprint(chunks, model)
    cache_path = cache_path_for(strategy, chunk_file)

    if cache_path.exists() and not rebuild_cache:
        with np.load(cache_path, allow_pickle=False) as data:
            saved_fingerprint = str(
                data["fingerprint"].item()
            )

            if saved_fingerprint == fingerprint:
                embeddings = data["embeddings"]

                if (
                    embeddings.ndim == 2
                    and len(embeddings) == len(chunks)
                    and np.isfinite(embeddings).all()
                ):
                    return embeddings, "cache"

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    if not np.isfinite(embeddings).all():
        raise ValueError("Non-finite document embeddings.")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        cache_path,
        embeddings=embeddings,
        fingerprint=fingerprint,
    )

    return embeddings, "computed"


# --------------------------------------------------
# QUERY EMBEDDING
# --------------------------------------------------

@lru_cache(maxsize=128)
def encode_query(question, use_instruction=True):
    """Encode a question using the same embedding model."""

    model = get_model()

    query_text = (
        QUERY_INSTRUCTION + question
        if use_instruction
        else question
    )

    embedding = model.encode(
        query_text,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    embedding = np.asarray(
        embedding,
        dtype=np.float32,
    )

    if not np.isfinite(embedding).all():
        raise ValueError("Non-finite query embedding.")

    return embedding


# --------------------------------------------------
# DENSE RETRIEVAL
# --------------------------------------------------

def search_dense(
    question: str,
    k: int = 5,
    strategy: str = "hybrid",
    chunk_file: str | Path | None = None,
    batch_size: int = 32,
    use_instruction: bool = True,
    rebuild_cache: bool = False,
) -> list[dict]:
    """Retrieve top-k PDF chunks using cosine similarity."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be non-empty.")

    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer.")

    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be positive.")

    chunks = load_chunks(
        strategy=strategy,
        chunk_file=chunk_file,
    )

    document_embeddings, _ = get_document_embeddings(
        chunks=chunks,
        strategy=strategy,
        chunk_file=chunk_file,
        batch_size=batch_size,
        rebuild_cache=rebuild_cache,
    )

    query_embedding = encode_query(
        question,
        use_instruction=use_instruction,
    )

    # Dot product of normalized vectors = cosine similarity.
    scores = document_embeddings @ query_embedding

    ranked_indices = np.argsort(
        -scores,
        kind="stable",
    )[:k]

    return [
        format_result(
            chunk=chunks[int(i)],
            score=float(scores[i]),
            method="dense",
            strategy=strategy,
        )
        for i in ranked_indices
    ]


# --------------------------------------------------
# COMMAND-LINE INTERFACE
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Dense retrieval for Core Rules PDF"
    )

    parser.add_argument(
        "question",
        type=str,
        help="Question to search",
    )

    parser.add_argument(
        "--strategy",
        choices=["all", *CHUNK_FILES],
        default="all",
        help="Chunking strategy (default: all)",
    )

    parser.add_argument(
        "--chunk-file",
        type=Path,
        default=None,
        help="Optional custom JSONL file",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--no-query-instruction",
        action="store_true",
        help="Disable the BGE query instruction",
    )

    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Recompute document embeddings",
    )

    parser.add_argument(
        "--show-text",
        action="store_true",
        help="Print complete retrieved chunks",
    )

    args = parser.parse_args()

    if args.chunk_file is not None and args.strategy == "all":
        parser.error(
            "--chunk-file requires a specific --strategy"
        )

    strategies = (
        list(CHUNK_FILES)
        if args.strategy == "all"
        else [args.strategy]
    )

    print("\n=== DENSE RETRIEVAL ===")
    print(f"Model: {MODEL_NAME}")
    print(f"Question: {args.question}")
    print(f"Strategies: {', '.join(strategies)}")
    print(f"Top-k per strategy: {args.k}")

    if "semantic_pure" in strategies:
        print(
            "WARNING: semantic_pure contains chunks "
            "exceeding the model's token limit."
        )

    for strategy in strategies:

        print(f"\n{'=' * 65}")
        print(f"STRATEGY: {strategy.upper()}")
        print("=" * 65)

        results = search_dense(
            question=args.question,
            k=args.k,
            strategy=strategy,
            chunk_file=args.chunk_file,
            batch_size=args.batch_size,
            use_instruction=not args.no_query_instruction,
            rebuild_cache=args.rebuild_cache,
        )

        print(f"Results: {len(results)}")

        for rank, result in enumerate(results, 1):

            print(f"\n{'-' * 60}")

            print(
                f"Rank {rank} | "
                f"ID: {result['id']} | "
                f"Cosine: {result['score']:.4f}"
            )

            print(
                f"Pages: {result['page_start']}"
                f"-{result['page_end']}"
            )

            text = result["text"]

            if not args.show_text:
                text = text[:500] + (
                    "..." if len(text) > 500 else ""
                )

            print(text)

    print(f"\n{'=' * 65}")
    print("Dense retrieval completed.")


if __name__ == "__main__":
    main()
