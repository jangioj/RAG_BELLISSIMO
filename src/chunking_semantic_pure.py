import argparse
import json
from pathlib import Path
from statistics import mean, median

import numpy as np
from sentence_transformers import SentenceTransformer

from chunking_semantic import build_units, MODEL_NAME


def semantic_change_chunks(
    markdown: str,
    model: SentenceTransformer,
    context_window: int = 2,
    change_percentile: float = 70,
    source: str = "Core Rules.pdf",
    batch_size: int = 32,
    model_name: str = MODEL_NAME,
) -> tuple[list[dict], dict]:
    """
    Semantic-driven chunking using contextual change points.

    Boundaries depend exclusively on embedding-based
    semantic distances, not on word-count constraints.
    """

    if context_window < 1:
        raise ValueError("context_window must be >= 1")

    if not 0 < change_percentile < 100:
        raise ValueError("Invalid change percentile")

    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    units = build_units(markdown)

    if not units:
        return [], {}

    # 1. Embed paragraph-level units.
    embeddings = model.encode(
        [u["text"] for u in units],
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=batch_size,
        show_progress_bar=True,
    )

    # 2. Compute contextual semantic distance
    # across each possible boundary.
    distances = []

    for boundary in range(1, len(units)):

        left = embeddings[
            max(0, boundary - context_window):boundary
        ].mean(axis=0)

        right = embeddings[
            boundary:min(
                len(units),
                boundary + context_window
            )
        ].mean(axis=0)

        cosine = np.dot(left, right) / (
            np.linalg.norm(left)
            * np.linalg.norm(right)
        )

        distances.append(1 - float(cosine))

    distances = np.array(distances)

    threshold = (
        float(np.percentile(distances, change_percentile))
        if len(distances)
        else None
    )

    # 3. Select high-distance local maxima.
    boundaries = []

    for i, score in enumerate(distances):

        previous = (
            distances[i - 1]
            if i > 0
            else float("-inf")
        )

        following = (
            distances[i + 1]
            if i < len(distances) - 1
            else float("-inf")
        )

        if (
            score >= threshold
            and score >= previous
            and score > following
        ):
            boundaries.append(i + 1)

    # 4. Build chunks using semantic boundaries only.
    chunks = []
    start = 0

    for end in boundaries + [len(units)]:

        selected = units[start:end]

        text = "\n\n".join(
            unit["text"] for unit in selected
        )

        chunks.append({
            "id": f"semantic_pure_{len(chunks) + 1:04d}",
            "text": text,
            "source": source,
            "page_start": selected[0]["page_start"],
            "page_end": selected[-1]["page_end"],
            "word_count": sum(
                u["word_count"] for u in selected
            ),
            "start_unit": start,
            "end_unit": end,
            "boundary_reason": (
                "semantic_peak"
                if end < len(units)
                else "end_of_document"
            ),
            "boundary_distance": (
                float(distances[end - 1])
                if end < len(units)
                else None
            ),
        })

        start = end

    diagnostics = {
        "model": model_name,
        "text_units": len(units),
        "context_window": context_window,
        "change_percentile": change_percentile,
        "distance_threshold": threshold,
        "semantic_boundaries": len(boundaries),
    }

    return chunks, diagnostics


def main():

    project_root = Path(__file__).resolve().parent.parent

    # --- CLI ARGUMENTS ---

    parser = argparse.ArgumentParser(
        description="Pure semantic Markdown chunking"
    )

    parser.add_argument(
        "--context-window",
        type=int,
        default=2,
        help="Context units on each side (default: 2)"
    )

    parser.add_argument(
        "--change-percentile",
        type=float,
        default=70,
        help="Semantic distance percentile (default: 70)"
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
        / "chunks" / "semantic_pure_chunks.jsonl",
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

    if args.context_window < 1:
        parser.error("--context-window must be >= 1")

    if not 0 < args.change_percentile < 100:
        parser.error(
            "--change-percentile must be between 0 and 100"
        )

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    if args.preview_count < 0:
        parser.error("--preview-count must be >= 0")

    # --- LOAD AND CHUNK ---

    markdown = args.input.read_text(encoding="utf-8")

    print("Loading embedding model...")
    model = SentenceTransformer(args.model_name)

    chunks, diagnostics = semantic_change_chunks(
        markdown,
        model,
        context_window=args.context_window,
        change_percentile=args.change_percentile,
        batch_size=args.batch_size,
        model_name=args.model_name,
    )

    units = build_units(markdown)

    # --- VALIDATION ---

    if not chunks:
        raise ValueError("No chunks generated.")

    reconstructed = "\n\n".join(
        c["text"] for c in chunks
    )

    original = "\n\n".join(
        u["text"] for u in units
    )

    assert reconstructed == original

    assert all(
        1 <= c["page_start"] <= c["page_end"]
        for c in chunks
    )

    assert all(
        "<!-- PDF_PAGE_BREAK -->" not in c["text"]
        for c in chunks
    )

    assert len({c["id"] for c in chunks}) == len(chunks)

    # --- DIAGNOSTICS ---

    sizes = [c["word_count"] for c in chunks]

    print("\n=== PURE SEMANTIC CHUNKING ===")

    for key, value in diagnostics.items():
        print(f"{key}: {value}")

    print("\n=== CHUNK STATISTICS ===")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Minimum words: {min(sizes)}")
    print(f"Maximum words: {max(sizes)}")
    print(f"Mean words: {mean(sizes):.1f}")
    print(f"Median words: {median(sizes):.1f}")

    print(
        f"Chunks under 80 words: "
        f"{sum(s < 80 for s in sizes)}"
    )

    print(
        f"Chunks over 500 words: "
        f"{sum(s > 500 for s in sizes)}"
    )

    print(
        f"Chunks over 800 words: "
        f"{sum(s > 800 for s in sizes)}"
    )

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
        print(f"Boundary: {chunk['boundary_reason']}")
        print(
            f"Semantic distance: "
            f"{chunk['boundary_distance']}"
        )

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
