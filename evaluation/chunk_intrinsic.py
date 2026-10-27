
"""Intrinsic diagnostics for the five PDF chunking strategies.

Checks data integrity, embedding length, and adjacent overlap.
Does not evaluate semantic coherence or retrieval performance.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median

from sentence_transformers import SentenceTransformer


STRATEGIES = {
    "fixed": "fixed_chunks.jsonl",
    "structured": "structured_chunks.jsonl",
    "semantic": "semantic_chunks.jsonl",
    "semantic_pure": "semantic_pure_chunks.jsonl",
    "hybrid": "hybrid_chunks.jsonl",
}

REQUIRED_FIELDS = {
    "id", "text", "source",
    "page_start", "page_end", "word_count"
}

MODEL_NAME = "BAAI/bge-small-en-v1.5"


def load_chunks(path):
    """Load and validate a JSONL chunk collection."""

    chunks = []
    seen_ids = set()

    with path.open(encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue

            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path.name}:{line_number}: invalid JSON"
                ) from exc

            if (
                not isinstance(chunk, dict)
                or not REQUIRED_FIELDS.issubset(chunk)
            ):
                raise ValueError(
                    f"{path.name}:{line_number}: invalid schema"
                )

            chunk_id = chunk["id"]

            if not isinstance(chunk_id, str) or not chunk_id:
                raise ValueError("Invalid chunk ID")

            if chunk_id in seen_ids:
                raise ValueError(f"Duplicate ID: {chunk_id}")

            seen_ids.add(chunk_id)

            if (
                not isinstance(chunk["text"], str)
                or not chunk["text"].strip()
            ):
                raise ValueError(f"Empty text: {chunk_id}")

            if (
                not isinstance(chunk["source"], str)
                or not chunk["source"].strip()
            ):
                raise ValueError(f"Invalid source: {chunk_id}")

            actual_words = len(chunk["text"].split())

            if (
                type(chunk["word_count"]) is not int
                or chunk["word_count"] != actual_words
            ):
                raise ValueError(
                    f"Word-count mismatch: {chunk_id}"
                )

            start = chunk["page_start"]
            end = chunk["page_end"]

            if not (
                type(start) is int
                and type(end) is int
                and 1 <= start <= end
            ):
                raise ValueError(
                    f"Invalid page range: {chunk_id}"
                )

            chunks.append(chunk)

    if not chunks:
        raise ValueError(f"No chunks found: {path}")

    return chunks


def adjacent_overlap(left, right):
    """Find exact word overlap between consecutive chunks."""

    max_size = min(len(left), len(right))

    for size in range(max_size, 0, -1):
        if left[-size:] == right[:size]:
            return size

    return 0


def reconstruct(chunks):
    """Reconstruct word sequence, removing adjacent overlap."""

    previous = chunks[0]["text"].split()
    reconstructed = previous.copy()
    repeated_words = 0

    for chunk in chunks[1:]:
        current = chunk["text"].split()

        overlap = adjacent_overlap(previous, current)

        reconstructed.extend(current[overlap:])
        repeated_words += overlap
        previous = current

    return reconstructed, repeated_words


def percentile_95(values):
    """Compute the nearest-rank 95th percentile."""

    ordered = sorted(values)
    index = math.ceil(0.95 * len(ordered)) - 1

    return ordered[index]


def audit(chunks, tokenizer, token_limit):
    """Compute a minimal set of intrinsic diagnostics."""

    word_lengths = [
        chunk["word_count"] for chunk in chunks
    ]

    token_lengths = [
        len(
            tokenizer.encode(
                chunk["text"],
                add_special_tokens=True,
                truncation=False
            )
        )
        for chunk in chunks
    ]

    reconstructed, repeated_words = reconstruct(chunks)

    total_words = sum(word_lengths)

    metrics = {
        "chunks": len(chunks),
        "median_words": median(word_lengths),
        "p95_tokens": percentile_95(token_lengths),
        "max_tokens": max(token_lengths),
        "over_limit": sum(
            length > token_limit
            for length in token_lengths
        ),
        "overlap_pct": round(
            100 * repeated_words / total_words, 1
        ),
    }

    return metrics, reconstructed


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=Path("data/processed/chunks")
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluation_results/chunk_intrinsic_summary.csv")
    )

    args = parser.parse_args()

    model = SentenceTransformer(MODEL_NAME)
    tokenizer = model.tokenizer
    token_limit = model.max_seq_length

    if not isinstance(token_limit, int) or token_limit < 1:
        raise ValueError("Invalid embedding token limit")

    results = []
    sequences = {}

    for strategy, filename in STRATEGIES.items():
        path = args.chunks_dir / filename

        chunks = load_chunks(path)

        metrics, sequence = audit(
            chunks, tokenizer, token_limit
        )

        results.append({
            "strategy": strategy,
            **metrics
        })

        sequences[strategy] = sequence

    reference = sequences["hybrid"]

    for result in results:
        strategy = result["strategy"]

        result["same_content_as_hybrid"] = (
            sequences[strategy] == reference
        )

    args.output.parent.mkdir(
        parents=True, exist_ok=True
    )

    with args.output.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys())
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nEmbedding model: {MODEL_NAME}")
    print(f"Token limit: {token_limit}")
    print("All JSONL integrity checks passed.\n")

    for result in results:
        print(f"--- {result['strategy']} ---")

        for key, value in result.items():
            if key != "strategy":
                print(f"  {key}: {value}")

        print()

    print(f"Results saved to: {args.output}")
    print(
        "Note: content equivalence is assessed across "
        "chunk collections, not against the original PDF."
    )


if __name__ == "__main__":
    main()
