
"""Reciprocal Rank Fusion of BM25 and Dense retrieval.

Combines rankings from the same chunk collection.
No LLM calls are required.

Examples:
    python -m src.retrieval_rrf "What is Engagement Range?"

    python -m src.retrieval_rrf "What is Engagement Range?" --strategy hybrid
"""

import argparse
from pathlib import Path

from src.retrieval_common import CHUNK_FILES
from src.retrieval_bm25 import search_bm25
from src.retrieval_dense import search_dense


def search_rrf(
    question: str,
    k: int = 5,
    strategy: str = "hybrid",
    chunk_file: str | Path | None = None,
    candidate_k: int = 20,
    rrf_k: int = 60,
    use_instruction: bool = True,
) -> list[dict]:
    """Fuse BM25 and Dense retrieval rankings.

    Args:
        question: Natural-language query.
        k: Number of final results.
        strategy: Selected chunking strategy.
        chunk_file: Optional custom JSONL file.
        candidate_k: Candidates retrieved per method.
        rrf_k: Rank-fusion smoothing constant.
        use_instruction: Enable BGE query instruction.

    Returns:
        Ranked results using the common retrieval format,
        with additional BM25 and Dense rank information.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be non-empty.")

    if type(k) is not int or k < 1:
        raise ValueError("k must be positive.")

    if type(candidate_k) is not int or candidate_k < k:
        raise ValueError("candidate_k must be >= k.")

    if type(rrf_k) is not int or rrf_k < 1:
        raise ValueError("rrf_k must be positive.")

    bm25_results = search_bm25(
        question=question,
        k=candidate_k,
        strategy=strategy,
        chunk_file=chunk_file,
    )

    dense_results = search_dense(
        question=question,
        k=candidate_k,
        strategy=strategy,
        chunk_file=chunk_file,
        use_instruction=use_instruction,
    )

    fused = {}

    for method, results in (
        ("bm25", bm25_results),
        ("dense", dense_results),
    ):
        for rank, result in enumerate(results, start=1):
            chunk_id = result["id"]

            if chunk_id not in fused:
                fused[chunk_id] = {
                    **result,
                    "score": 0.0,
                    "method": "rrf",
                    "bm25_rank": None,
                    "dense_rank": None,
                }

            fused[chunk_id]["score"] += (
                1.0 / (rrf_k + rank)
            )

            fused[chunk_id][f"{method}_rank"] = rank

    ranked = sorted(
        fused.values(),
        key=lambda item: (
            -item["score"],
            item["id"],
        ),
    )

    return ranked[:k]


def main():
    parser = argparse.ArgumentParser(
        description="RRF retrieval for Core Rules PDF"
    )

    parser.add_argument(
        "question",
        type=str,
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
    )

    parser.add_argument(
        "--k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--candidate-k",
        type=int,
        default=20,
        help="Number of candidates per retriever",
    )

    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF smoothing constant",
    )

    parser.add_argument(
        "--no-query-instruction",
        action="store_true",
    )

    parser.add_argument(
        "--show-text",
        action="store_true",
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

    print("\n=== RRF RETRIEVAL ===")
    print(f"Question: {args.question}")
    print(f"Strategies: {', '.join(strategies)}")
    print(f"Top-k: {args.k}")
    print(f"Candidate pool: {args.candidate_k}")
    print(f"RRF constant: {args.rrf_k}")

    for strategy in strategies:
        print(f"\n{'=' * 65}")
        print(f"STRATEGY: {strategy.upper()}")
        print("=" * 65)

        results = search_rrf(
            question=args.question,
            k=args.k,
            strategy=strategy,
            chunk_file=args.chunk_file,
            candidate_k=args.candidate_k,
            rrf_k=args.rrf_k,
            use_instruction=not args.no_query_instruction,
        )

        print(f"Results: {len(results)}")

        for rank, result in enumerate(results, start=1):
            print(f"\n{'-' * 60}")

            print(
                f"Rank {rank} | "
                f"ID: {result['id']} | "
                f"RRF: {result['score']:.6f}"
            )

            print(
                f"BM25 rank: {result['bm25_rank']} | "
                f"Dense rank: {result['dense_rank']}"
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

    print("\nRRF retrieval completed.")


if __name__ == "__main__":
    main()
