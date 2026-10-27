
"""Offline evaluation of PDF retrieval configurations.

Compares BM25, Dense and RRF across chunking strategies
using source-verified evidence anchors.

Metrics are lexical evidence-coverage proxies, NOT
human judgments of semantic relevance or answer quality.

Example:
    python -m evaluation.evaluate_pdf
    python -m evaluation.evaluate_pdf --strategy hybrid --method bm25
"""

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

from src.retrieval_common import (
    CHUNK_FILES,
    PROJECT_ROOT,
    load_chunks,
)
from src.retrieval_bm25 import search_bm25
from src.retrieval_dense import search_dense, encode_query
from src.retrieval_rrf import search_rrf

from src.retrieval_context import retrieve_with_context
# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DEFAULT_DATASET = (
    PROJECT_ROOT / "evaluation" / "pdf_questions.json"
)

DEFAULT_RESULTS = (
    PROJECT_ROOT / "data" / "evaluation_results"
)

PRIMARY_STRATEGIES = (
    "fixed",
    "structured",
    "semantic",
    "hybrid",
)

METHODS = ("bm25", "dense", "rrf", "rrf_context")

# --------------------------------------------------
# DATASET AND EVIDENCE
# --------------------------------------------------

def normalize(text: str) -> str:
    """Normalize punctuation and whitespace for anchor matching."""
    return " ".join(
        text.casefold()
        .replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("–", "-")
        .replace("—", "-")
        .split()
    )


def load_questions(path: Path, split: str) -> list[dict]:
    """Load questions from the selected evaluation split."""

    with path.open("r", encoding="utf-8-sig") as stream:
        dataset = json.load(stream)

    questions = dataset.get("questions")

    if not isinstance(questions, list) or not questions:
        raise ValueError("Dataset has no valid questions list.")

    selected = []
    seen_ids = set()

    for question in questions:
        question_id = question["id"]

        if question_id in seen_ids:
            raise ValueError(
                f"Duplicate question ID: {question_id}"
            )

        seen_ids.add(question_id)

        if question["split"] not in ("development", "held_out"):
            raise ValueError(
                f"Invalid split for {question_id}"
            )

        evidence = question["required_evidence"]

        if not evidence or any(
            not isinstance(item.get("anchor"), str)
            or not item["anchor"].strip()
            or type(item.get("page")) is not int
            for item in evidence
        ):
            raise ValueError(
                f"Invalid evidence for {question_id}"
            )

        if split == "all" or question["split"] == split:
            selected.append(question)

    if not selected:
        raise ValueError(
            f"No questions found for split: {split}"
        )

    return selected


def matches_anchor(result: dict, evidence: dict) -> bool:
    """Check whether a chunk contains a reference anchor.

    This is a lexical proxy, not a semantic relevance judgment.
    """

    page = evidence["page"]

    return (
        result["page_start"] <= page <= result["page_end"]
        and normalize(evidence["anchor"])
        in normalize(result["text"])
    )


def check_gold_coverage(
    questions: list[dict],
    strategy: str,
    chunk_file: Path | None,
):
    """Verify that reference anchors exist in the collection."""

    chunks = load_chunks(
        strategy=strategy,
        chunk_file=chunk_file,
    )

    missing = []

    for question in questions:
        for number, evidence in enumerate(
            question["required_evidence"], start=1
        ):
            found = any(
                matches_anchor(chunk, evidence)
                for chunk in chunks
            )

            if not found:
                missing.append(
                    f"{question['id']} anchor {number}"
                )

    if missing:
        raise ValueError(
            f"Missing reference evidence in {strategy}: "
            f"{', '.join(missing)}"
        )


# --------------------------------------------------
# RETRIEVAL
# --------------------------------------------------

def retrieve(
    method: str,
    question: str,
    strategy: str,
    chunk_file: Path | None,
    k: int,
    candidate_k: int,
) -> list[dict]:
    """Call the selected retriever using a common interface."""

    kwargs = {
        "question": question,
        "k": k,
        "strategy": strategy,
        "chunk_file": chunk_file,
    }

    if method == "bm25":
        return search_bm25(**kwargs)

    if method == "dense":
        return search_dense(**kwargs)

    if method == "rrf_context":
        return retrieve_with_context(
            question=question,
            strategy=strategy,
            k=k,
            candidate_k=candidate_k,
            expand_top=2,
        )

    return search_rrf(
        **kwargs,
        candidate_k=candidate_k,
    )


# --------------------------------------------------
# EVALUATION METRICS
# --------------------------------------------------

def evaluate_question(
    question: dict,
    results: list[dict],
    k: int,
) -> dict:
    """Calculate source-anchor coverage metrics at k."""

    evidence = question["required_evidence"]

    matched = []
    first_rank = None

    for fact in evidence:
        ranks = [
            rank
            for rank, result in enumerate(results[:k], 1)
            if matches_anchor(result, fact)
        ]

        found = bool(ranks)
        matched.append(found)

        if found:
            if first_rank is None or ranks[0] < first_rank:
                first_rank = ranks[0]

    covered = sum(matched)
    total = len(evidence)

    missing_facts = [
        fact["fact"]
        for fact, found in zip(evidence, matched)
        if not found
    ]

    return {
        "anchor_hit_at_k": int(first_rank is not None),
        "anchor_mrr_at_k": (
            1.0 / first_rank if first_rank else 0.0
        ),
        "anchor_recall_at_k": covered / total,
        "full_anchor_coverage_at_k": int(covered == total),
        "covered_anchors": covered,
        "required_anchors": total,
        "first_anchor_rank": first_rank or "",
        "missing_facts": json.dumps(
            missing_facts,
            ensure_ascii=False,
        ),
        "retrieved_ids": json.dumps(
            [r["id"] for r in results[:k]]
        ),
        "retrieved_pages": json.dumps([
            [r["page_start"], r["page_end"]]
            for r in results[:k]
        ]),
        "context_words": sum(
            r["word_count"] for r in results[:k]
        ),
    }


# --------------------------------------------------
# OUTPUT
# --------------------------------------------------

def save_csv(path: Path, rows: list[dict]):
    """Save evaluation results to CSV."""

    if not rows:
        raise ValueError("Cannot save empty results.")

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
        )

        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------
# MAIN EVALUATION
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Offline PDF retrieval evaluation"
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS,
    )

    parser.add_argument(
        "--split",
        choices=["development", "held_out", "all"],
        default="development",
    )

    parser.add_argument(
        "--strategy",
        choices=["all", *CHUNK_FILES],
        default="all",
    )

    parser.add_argument(
        "--method",
        choices=["all", *METHODS],
        default="all",
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
    )

    args = parser.parse_args()

    if args.k < 1 or args.candidate_k < args.k:
        parser.error("Require 1 <= k <= candidate-k.")

    if args.chunk_file is not None and args.strategy == "all":
        parser.error(
            "--chunk-file requires a specific --strategy"
        )

    strategies = (
        PRIMARY_STRATEGIES
        if args.strategy == "all"
        else (args.strategy,)
    )

    methods = (
        METHODS
        if args.method == "all"
        else (args.method,)
    )

    questions = load_questions(
        args.dataset,
        args.split,
    )

    print("\n=== PDF RETRIEVAL EVALUATION ===")
    print(f"Questions: {len(questions)}")
    print(f"Split: {args.split}")
    print(f"Top-k: {args.k}")
    print(f"Strategies: {', '.join(strategies)}")
    print(f"Methods: {', '.join(methods)}")

    if "semantic_pure" in strategies:
        print(
            "WARNING: semantic_pure contains chunks "
            "exceeding BGE's token limit."
        )

    detailed_results = []
    summary_results = []

    for strategy in strategies:

        print(f"\nChecking evidence: {strategy}")

        check_gold_coverage(
            questions,
            strategy,
            args.chunk_file,
        )

        for method in methods:

            print(f"Evaluating {strategy} / {method}")

            # Warm-up: exclude model and index initialization
            # from the indicative per-query latency.
            retrieve(
                method=method,
                question="Explain movement and combat rules in Warhammer.",
                strategy=strategy,
                chunk_file=args.chunk_file,
                k=args.k,
                candidate_k=args.candidate_k,
            )

            if method in ("dense", "rrf"):
                encode_query.cache_clear()

            local_results = []

            for question in questions:

                start = time.perf_counter()

                retrieved = retrieve(
                    method=method,
                    question=question["question"],
                    strategy=strategy,
                    chunk_file=args.chunk_file,
                    k=args.k,
                    candidate_k=args.candidate_k,
                )

                latency_ms = (
                    time.perf_counter() - start
                ) * 1000

                metrics = evaluate_question(
                    question,
                    retrieved,
                    len(retrieved),
                )

                row = {
                    "strategy": strategy,
                    "method": method,
                    "split": question["split"],
                    "question_id": question["id"],
                    "category": question["category"],
                    "question": question["question"],
                    **metrics,
                    "latency_ms": round(latency_ms, 2),
                }

                local_results.append(row)
                detailed_results.append(row)

            summary = {
                "strategy": strategy,
                "method": method,
                "split": args.split,
                "questions": len(local_results),
            }

            for metric in (
                "anchor_hit_at_k",
                "anchor_mrr_at_k",
                "anchor_recall_at_k",
                "full_anchor_coverage_at_k",
            ):
                summary[metric] = round(
                    statistics.mean(
                        row[metric]
                        for row in local_results
                    ),
                    3,
                )

            summary["median_context_words"] = round(
                statistics.median(
                    row["context_words"]
                    for row in local_results
                )
            )

            summary["median_latency_ms"] = round(
                statistics.median(
                    row["latency_ms"]
                    for row in local_results
                ),
                2,
            )

            summary_results.append(summary)

    # Save separate detailed and aggregate results.

    details_path = (
        args.output_dir
        / f"pdf_retrieval_details_{args.split}.csv"
    )

    summary_path = (
        args.output_dir
        / f"pdf_retrieval_summary_{args.split}.csv"
    )

    save_csv(details_path, detailed_results)
    save_csv(summary_path, summary_results)

    # Display the comparison, ordered by evidence coverage.

    ranked = sorted(
        summary_results,
        key=lambda row: (
            -row["full_anchor_coverage_at_k"],
            -row["anchor_recall_at_k"],
        ),
    )

    print("\n=== COMPARISON ===")

    print(
        f"{'STRATEGY':<16} "
        f"{'METHOD':<8} "
        f"{'HIT':>6} "
        f"{'MRR':>6} "
        f"{'RECALL':>7} "
        f"{'FULL':>6} "
        f"{'MS':>8}"
    )

    for row in ranked:

        print(
            f"{row['strategy']:<16} "
            f"{row['method']:<8} "
            f"{row['anchor_hit_at_k']:>6.3f} "
            f"{row['anchor_mrr_at_k']:>6.3f} "
            f"{row['anchor_recall_at_k']:>7.3f} "
            f"{row['full_anchor_coverage_at_k']:>6.3f} "
            f"{row['median_latency_ms']:>8.2f}"
        )

    print("\nResults saved:")
    print(summary_path)
    print(details_path)


if __name__ == "__main__":
    main()
