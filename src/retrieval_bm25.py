
"""BM25 retrieval over Warhammer Core Rules chunks.

Supports all five chunking strategies and configurable
BM25 hyperparameters.

By default, runs retrieval separately on all strategies.

Examples:
    python -m src.retrieval_bm25 "What is Engagement Range?"

    python -m src.retrieval_bm25 "What is Engagement Range?" --strategy hybrid

    python -m src.retrieval_bm25 "What is Engagement Range?" --strategy all --k 5
"""

import argparse
import math
import re
from functools import lru_cache
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.retrieval_common import (
    CHUNK_FILES,
    load_chunks,
    format_result,
)


# --------------------------------------------------
# TOKENIZATION
# --------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Apply identical lexical tokenization to documents and queries.

    Lowercase text and split punctuation, including hyphens.
    """
    return re.findall(r"[a-z0-9]+", text.casefold())


# --------------------------------------------------
# INDEX CONSTRUCTION
# --------------------------------------------------

@lru_cache(maxsize=16)
def build_index(
    strategy: str,
    chunk_file: str | None,
    k1: float,
    b: float,
):
    """Build and cache a BM25 index for one chunk collection.

    Cached indexes are reused within the same Python process.
    """

    chunks = load_chunks(
        strategy=strategy,
        chunk_file=chunk_file,
    )

    tokenized_corpus = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    if not any(tokenized_corpus):
        raise ValueError("No indexable text found.")

    index = BM25Okapi(
        tokenized_corpus,
        k1=k1,
        b=b,
    )

    return chunks, index


# --------------------------------------------------
# BM25 RETRIEVAL
# --------------------------------------------------

def search_bm25(
    question: str,
    k: int = 5,
    strategy: str = "hybrid",
    chunk_file: str | Path | None = None,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[dict]:
    """Retrieve top-k chunks from one chunking strategy.

    Args:
        question: Natural-language question.
        k: Maximum number of returned chunks.
        strategy: Chunking strategy to search.
        chunk_file: Optional custom JSONL path.
        k1: BM25 term-frequency saturation parameter.
        b: BM25 length-normalization parameter.

    Returns:
        Ranked retrieval results in the shared format.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be non-empty.")

    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer.")

    if not math.isfinite(k1) or k1 <= 0:
        raise ValueError("k1 must be a finite positive number.")

    if not math.isfinite(b) or not 0 <= b <= 1:
        raise ValueError("b must be between 0 and 1.")

    file_path = (
        str(Path(chunk_file).resolve())
        if chunk_file is not None
        else None
    )

    chunks, index = build_index(
        strategy=strategy,
        chunk_file=file_path,
        k1=k1,
        b=b,
    )

    query_tokens = tokenize(question)

    if not query_tokens:
        return []

    scores = index.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(chunks)),
        key=lambda i: (-float(scores[i]), i),
    )

    results = []

    for i in ranked_indices:
        score = float(scores[i])

        if score <= 0:
            continue

        results.append(
            format_result(
                chunk=chunks[i],
                score=score,
                method="bm25",
                strategy=strategy,
            )
        )

        if len(results) >= k:
            break

    return results


# --------------------------------------------------
# COMMAND-LINE INTERFACE
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="BM25 retrieval for Warhammer Core Rules"
    )

    parser.add_argument(
        "question",
        type=str,
        help="Question to retrieve evidence for",
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
        help="Top-k results per strategy",
    )

    parser.add_argument(
        "--k1",
        type=float,
        default=1.5,
        help="BM25 term-frequency saturation",
    )

    parser.add_argument(
        "--b",
        type=float,
        default=0.75,
        help="BM25 length normalization",
    )

    parser.add_argument(
        "--show-text",
        action="store_true",
        help="Display complete retrieved chunks",
    )

    args = parser.parse_args()

    # A custom file must be associated with one strategy.
    if args.chunk_file is not None and args.strategy == "all":
        parser.error(
            "--chunk-file requires a specific --strategy"
        )

    # Select strategies.
    if args.strategy == "all":
        strategies = list(CHUNK_FILES)
    else:
        strategies = [args.strategy]

    print("\n=== BM25 RETRIEVAL ===")
    print(f"Question: {args.question}")
    print(f"Strategies: {', '.join(strategies)}")
    print(f"Top-k per strategy: {args.k}")
    print(f"Parameters: k1={args.k1}, b={args.b}")

    # Each strategy is indexed and searched independently.
    for strategy in strategies:

        print(f"\n{'=' * 65}")
        print(f"STRATEGY: {strategy.upper()}")
        print("=" * 65)

        results = search_bm25(
            question=args.question,
            k=args.k,
            strategy=strategy,
            chunk_file=args.chunk_file,
            k1=args.k1,
            b=args.b,
        )

        print(f"Results: {len(results)}")

        if not results:
            print("No positive BM25 matches found.")
            continue

        for rank, result in enumerate(results, 1):

            print(f"\n{'-' * 60}")

            print(
                f"Rank {rank} | "
                f"ID: {result['id']} | "
                f"Score: {result['score']:.4f}"
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
    print("BM25 retrieval completed.")


if __name__ == "__main__":
    main()
