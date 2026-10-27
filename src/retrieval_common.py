
"""Shared utilities for PDF retrieval.

Loads and validates chunk collections and standardizes
retrieval results across BM25, Dense and RRF.
"""

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHUNK_DIR = PROJECT_ROOT / "data" / "processed" / "chunks"

CHUNK_FILES = {
    "fixed": "fixed_chunks.jsonl",
    "structured": "structured_chunks.jsonl",
    "semantic": "semantic_chunks.jsonl",
    "semantic_pure": "semantic_pure_chunks.jsonl",
    "hybrid": "hybrid_chunks.jsonl",
}

REQUIRED_FIELDS = {
    "id",
    "text",
    "source",
    "page_start",
    "page_end",
    "word_count",
}


def load_chunks(
    strategy: str,
    chunk_file: str | Path | None = None,
) -> list[dict]:
    """Load and validate chunks from a JSONL file.

    Args:
        strategy: One of the five chunking strategies.
        chunk_file: Optional alternative JSONL path.

    Returns:
        List of validated chunk dictionaries.
    """

    if strategy not in CHUNK_FILES:
        raise ValueError(
            f"Unknown strategy: {strategy}. "
            f"Available: {list(CHUNK_FILES)}"
        )

    path = (
        Path(chunk_file)
        if chunk_file is not None
        else CHUNK_DIR / CHUNK_FILES[strategy]
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Chunk file not found: {path}"
        )

    chunks = []
    seen_ids = set()

    with path.open("r", encoding="utf-8-sig") as f:

        for line_number, line in enumerate(f, 1):

            if not line.strip():
                continue

            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path.name}, line {line_number}: "
                    f"invalid JSON"
                ) from exc

            if not isinstance(chunk, dict):
                raise ValueError(
                    f"Invalid record at line {line_number}"
                )

            missing = REQUIRED_FIELDS - chunk.keys()

            if missing:
                raise ValueError(
                    f"Line {line_number}: "
                    f"missing fields {sorted(missing)}"
                )

            chunk_id = chunk["id"]

            if not isinstance(chunk_id, str) or not chunk_id:
                raise ValueError(
                    f"Invalid ID at line {line_number}"
                )

            if chunk_id in seen_ids:
                raise ValueError(
                    f"Duplicate chunk ID: {chunk_id}"
                )

            seen_ids.add(chunk_id)

            if (
                not isinstance(chunk["text"], str)
                or not chunk["text"].strip()
            ):
                raise ValueError(
                    f"Empty text in chunk {chunk_id}"
                )

            if (
                not isinstance(chunk["source"], str)
                or not chunk["source"].strip()
            ):
                raise ValueError(
                    f"Invalid source in chunk {chunk_id}"
                )

            start = chunk["page_start"]
            end = chunk["page_end"]

            if not (
                type(start) is int
                and type(end) is int
                and 1 <= start <= end
            ):
                raise ValueError(
                    f"Invalid page range in {chunk_id}"
                )

            word_count = chunk["word_count"]

            if (
                type(word_count) is not int
                or word_count != len(chunk["text"].split())
            ):
                raise ValueError(
                    f"Invalid word count in {chunk_id}"
                )

            chunks.append(chunk)

    if not chunks:
        raise ValueError(
            f"No chunks found in {path}"
        )

    return chunks


def format_result(
    chunk: dict,
    score: float,
    method: str,
    strategy: str,
) -> dict:
    """Return a standardized retrieval result.

    The score is method-specific and must not be compared
    directly across different retrieval methods.
    """

    metadata = {
        key: value
        for key, value in chunk.items()
        if key not in REQUIRED_FIELDS
    }

    return {
        "id": chunk["id"],
        "text": chunk["text"],
        "source": chunk["source"],
        "page_start": chunk["page_start"],
        "page_end": chunk["page_end"],
        "word_count": chunk["word_count"],
        "score": float(score),
        "method": method,
        "strategy": strategy,
        "metadata": metadata,
    }


if __name__ == "__main__":

    print("\n=== CHUNK LOADING VALIDATION ===\n")

    for strategy in CHUNK_FILES:

        chunks = load_chunks(strategy)

        print(
            f"{strategy:15s} "
            f"{len(chunks):4d} chunks | "
            f"pages "
            f"{min(c['page_start'] for c in chunks)}-"
            f"{max(c['page_end'] for c in chunks)}"
        )

    print("\nAll chunk collections loaded successfully.")
