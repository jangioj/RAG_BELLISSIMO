import argparse
import json
import re
from pathlib import Path
from statistics import mean, median

import numpy as np
from sentence_transformers import SentenceTransformer

from chunking_semantic import build_units, MODEL_NAME


def hybrid_chunks(
    markdown,
    model,
    target_words=200,
    min_words=80,
    max_words=350,
    context_window=2,
    heading_bonus=0.06,
    length_penalty=0.12,
    batch_size=32,
):
    """
    Hybrid chunking combining:
    - Contextual semantic distance
    - Markdown heading information
    - Adaptive length constraints

    Select the highest-scoring feasible paragraph boundary.
    """

    if not 0 < min_words <= target_words <= max_words:
        raise ValueError("Invalid word limits")

    if context_window < 1:
        raise ValueError("Context window must be positive")

    if batch_size < 1:
        raise ValueError("Batch size must be positive")

    units = build_units(markdown)

    if not units:
        return []

    if any(u["word_count"] > max_words for u in units):
        raise ValueError(
            "Oversized text unit: inspect source before splitting."
        )

    # --- EMBEDDINGS ---

    embeddings = model.encode(
        [u["text"] for u in units],
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=batch_size,
        show_progress_bar=True,
    )

    # --- CONTEXTUAL SEMANTIC DISTANCE ---

    distances = []

    for boundary in range(1, len(units)):

        left = embeddings[
            max(0, boundary - context_window):boundary
        ].mean(axis=0)

        right = embeddings[
            boundary:min(
                len(units), boundary + context_window
            )
        ].mean(axis=0)

        cosine = np.dot(left, right) / (
            np.linalg.norm(left)
            * np.linalg.norm(right)
        )

        distances.append(1 - float(cosine))

    # --- CUMULATIVE WORD COUNTS ---

    cumulative = np.r_[
        0,
        np.cumsum([u["word_count"] for u in units]),
    ]

    chunks = []
    start = 0

    # --- HYBRID BOUNDARY SELECTION ---

    while start < len(units):

        candidates = []

        for end in range(start + 1, len(units) + 1):

            words = int(cumulative[end] - cumulative[start])

            if words > max_words:
                break

            if words < min_words and end < len(units):
                continue

            remaining = int(
                cumulative[-1] - cumulative[end]
            )

            semantic = (
                distances[end - 1]
                if end < len(units)
                else 0.0
            )

            has_heading = (
                end < len(units)
                and bool(
                    re.match(
                        r"^#{1,6}\s",
                        units[end]["text"]
                    )
                )
            )

            # Combined boundary score.
            score = (
                semantic
                + (heading_bonus if has_heading else 0.0)
                - length_penalty
                * abs(words - target_words)
                / target_words
            )

            candidates.append((
                score,
                end,
                words,
                semantic,
                has_heading,
                remaining,
            ))

        if not candidates:
            raise RuntimeError("No feasible chunk boundary")

        # Avoid leaving an excessively short final chunk.
        feasible = [
            c for c in candidates
            if c[5] == 0 or c[5] >= min_words
        ]

        selected = max(
            feasible or candidates,
            key=lambda c: (c[0], c[2]),
        )

        score, end, words, semantic, heading, _ = selected

        selected_units = units[start:end]

        chunks.append({
            "id": f"hybrid_{len(chunks) + 1:04d}",
            "text": "\n\n".join(
                u["text"] for u in selected_units
            ),
            "source": "Core Rules.pdf",
            "page_start": selected_units[0]["page_start"],
            "page_end": selected_units[-1]["page_end"],
            "word_count": words,
            "boundary_distance": (
                float(semantic)
                if end < len(units)
                else None
            ),
            "next_has_heading": (
                heading if end < len(units) else None
            ),
            "boundary_score": (
                float(score)
                if end < len(units)
                else None
            ),
        })

        start = end

    return chunks


def main():

    project_root = Path(__file__).resolve().parent.parent

    # --- CLI ARGUMENTS ---

    parser = argparse.ArgumentParser(
        description="Hybrid semantic and structure-aware chunking"
    )

    parser.add_argument(
        "--target-words",
        type=int,
        default=200,
        help="Preferred chunk length (default: 200)"
    )

    parser.add_argument(
        "--min-words",
        type=int,
        default=80,
        help="Minimum preferred length (default: 80)"
    )

    parser.add_argument(
        "--max-words",
        type=int,
        default=350,
        help="Maximum chunk length (default: 350)"
    )

    parser.add_argument(
        "--context-window",
        type=int,
        default=2,
        help="Context units per side (default: 2)"
    )

    parser.add_argument(
        "--heading-bonus",
        type=float,
        default=0.06,
        help="Heading boundary score bonus (default: 0.06)"
    )

    parser.add_argument(
        "--length-penalty",
        type=float,
        default=0.12,
        help="Length deviation penalty weight (default: 0.12)"
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_NAME,
        help="Sentence Transformers embedding model"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size (default: 32)"
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "data" / "processed"
        / "core_rules_clean.md",
        help="Input Markdown file"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "data" / "processed"
        / "chunks" / "hybrid_chunks.jsonl",
        help="Output JSONL file"
    )

    parser.add_argument(
        "--no_save",
        "--no-save",
        action="store_true",
        help="Disable saving (default: save enabled)"
    )

    parser.add_argument(
        "--preview-count",
        type=int,
        default=3,
        help="Number of chunks to preview (default: 3)"
    )

    args = parser.parse_args()

    # --- PARAMETER VALIDATION ---

    if not 0 < args.min_words <= args.target_words <= args.max_words:
        parser.error(
            "Require 0 < min-words <= target-words <= max-words"
        )

    if args.context_window < 1:
        parser.error("--context-window must be >= 1")

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    if args.preview_count < 0:
        parser.error("--preview-count must be >= 0")

    if not (
        np.isfinite(args.heading_bonus)
        and np.isfinite(args.length_penalty)
    ):
        parser.error("Score weights must be finite")

    # --- LOAD AND CHUNK ---

    markdown = args.input.read_text(encoding="utf-8")

    print("Loading embedding model...")
    model = SentenceTransformer(args.model_name)

    chunks = hybrid_chunks(
        markdown,
        model,
        target_words=args.target_words,
        min_words=args.min_words,
        max_words=args.max_words,
        context_window=args.context_window,
        heading_bonus=args.heading_bonus,
        length_penalty=args.length_penalty,
        batch_size=args.batch_size,
    )

    units = build_units(markdown)

    # --- VALIDATION ---

    if not chunks:
        raise ValueError("No chunks generated.")

    original = "\n\n".join(
        u["text"] for u in units
    )

    reconstructed = "\n\n".join(
        c["text"] for c in chunks
    )

    assert original == reconstructed, (
        "Chunking did not preserve the original text."
    )

    assert all(
        0 < c["word_count"] <= args.max_words
        for c in chunks
    )

    assert all(
        1 <= c["page_start"] <= c["page_end"]
        for c in chunks
    )

    assert len({c["id"] for c in chunks}) == len(chunks)

    assert all(
        "<!-- PDF_PAGE_BREAK -->" not in c["text"]
        for c in chunks
    )

    # --- DIAGNOSTICS ---

    sizes = [c["word_count"] for c in chunks]

    print("\n=== HYBRID CHUNKING ===")
    print(f"Model: {args.model_name}")
    print(f"Target words: {args.target_words}")
    print(f"Minimum words: {args.min_words}")
    print(f"Maximum words: {args.max_words}")
    print(f"Context window: {args.context_window}")
    print(f"Heading bonus: {args.heading_bonus}")
    print(f"Length penalty: {args.length_penalty}")
    print(f"Batch size: {args.batch_size}")
    print(f"Text units: {len(units)}")

    print("\n=== CHUNK STATISTICS ===")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Minimum words: {min(sizes)}")
    print(f"Maximum words: {max(sizes)}")
    print(f"Mean words: {mean(sizes):.1f}")
    print(f"Median words: {median(sizes):.1f}")
    print(
        f"Chunks under 80 words: "
        f"{sum(n < 80 for n in sizes)}"
    )
    print(
        f"Chunks over 500 words: "
        f"{sum(n > 500 for n in sizes)}"
    )

    # --- LONGEST CHUNKS ---

    print("\n=== THREE LONGEST CHUNKS ===")

    for chunk in sorted(
        chunks,
        key=lambda c: c["word_count"],
        reverse=True,
    )[:3]:

        print(
            f"{chunk['id']} | "
            f"{chunk['word_count']} words | "
            f"Pages {chunk['page_start']}-"
            f"{chunk['page_end']}"
        )

    # --- PREVIEWS ---

    print("\n=== CHUNK PREVIEWS ===")

    for chunk in chunks[:args.preview_count]:

        print(f"\nID: {chunk['id']}")

        print(
            f"Pages: {chunk['page_start']}-"
            f"{chunk['page_end']}"
        )

        print(f"Words: {chunk['word_count']}")
        print(f"Semantic distance: {chunk['boundary_distance']}")
        print(f"Heading boundary: {chunk['next_has_heading']}")
        print(f"Boundary score: {chunk['boundary_score']}")

        preview = " ".join(chunk["text"].split())
        print(f"Preview: {preview[:250]}")

    # --- OPTIONAL JSONL EXPORT ---

    if args.no_save:
        print("\nSaving disabled (--no_save).")

    else:
        args.output.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with args.output.open("w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(
                    json.dumps(
                        chunk,
                        ensure_ascii=False
                    ) + "\n"
                )

        print(
            f"\nSaved {len(chunks)} chunks to: {args.output}"
        )


if __name__ == "__main__":
    main()

