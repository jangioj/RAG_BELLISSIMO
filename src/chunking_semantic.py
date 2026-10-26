import argparse
import json
import re
from pathlib import Path
from statistics import mean, median

import numpy as np
from sentence_transformers import SentenceTransformer


PAGE_MARKER = "<!-- PDF_PAGE_BREAK -->"
HEADING_RE = re.compile(r"^#{1,6}\s+.+$")
MODEL_NAME = "BAAI/bge-small-en-v1.5"


def build_units(markdown: str) -> list[dict]:
    """
    Extract paragraph-level units from Markdown.

    Preserve page numbers and keep Markdown tables together.
    Attach standalone headings to their following paragraph.
    Heading levels do not determine chunk boundaries.
    """

    blocks = []
    lines = []
    page = 1
    start_page = 1

    def flush():
        if lines:
            content = "\n".join(lines).strip()

            if content:
                blocks.append({
                    "text": content,
                    "page_start": start_page,
                    "page_end": page,
                })

            lines.clear()

    for line in markdown.splitlines():

        if line.strip() == PAGE_MARKER:
            flush()
            page += 1

        elif not line.strip():
            flush()

        else:
            if not lines:
                start_page = page

            lines.append(line)

    flush()

    units = []
    pending_headings = []

    def add_unit(parts):
        text = "\n\n".join(p["text"] for p in parts)

        units.append({
            "text": text,
            "page_start": parts[0]["page_start"],
            "page_end": parts[-1]["page_end"],
            "word_count": len(text.split()),
        })

    for block in blocks:

        if HEADING_RE.fullmatch(block["text"]):
            pending_headings.append(block)

        else:
            add_unit(pending_headings + [block])
            pending_headings = []

    if pending_headings:
        add_unit(pending_headings)

    return units


def semantic_chunks(
    markdown: str,
    model: SentenceTransformer,
    target_words: int = 180,
    min_words: int = 80,
    max_words: int = 250,
    boundary_quantile: float = 0.25,
    source: str = "Core Rules.pdf",
    batch_size: int = 32,
) -> list[dict]:
    """
    Detect semantic changes between adjacent text units.

    Strong local similarity minima are preferred as boundaries.
    Word limits prevent excessively large chunks.
    """

    if not 0 < min_words <= target_words <= max_words:
        raise ValueError("Invalid chunk-size parameters.")

    if not 0 < boundary_quantile < 1:
        raise ValueError("Invalid boundary quantile.")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    units = build_units(markdown)

    if not units:
        return []

    # Do not silently split oversized paragraphs or tables.
    oversized = [
        u for u in units if u["word_count"] > max_words
    ]

    if oversized:
        raise ValueError(
            f"{len(oversized)} units exceed max_words. "
            "Inspect these units before splitting them."
        )

    # Compute one normalized embedding per text unit.
    embeddings = model.encode(
        [u["text"] for u in units],
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=batch_size,
        show_progress_bar=True,
    )

    # Normalized dot product = cosine similarity.
    similarities = np.sum(
        embeddings[:-1] * embeddings[1:],
        axis=1,
    )

    threshold = (
        float(np.quantile(similarities, boundary_quantile))
        if len(similarities) else None
    )

    typical_similarity = (
        float(np.median(similarities))
        if len(similarities) else None
    )

    # Find unusually low local similarities.
    boundaries = set()

    for i, similarity in enumerate(similarities):

        is_local_minimum = (
            (i == 0 or similarity <= similarities[i - 1])
            and (
                i == len(similarities) - 1
                or similarity <= similarities[i + 1]
            )
        )

        if similarity <= threshold and is_local_minimum:
            boundaries.add(i + 1)

    chunks = []
    current = []
    current_words = 0

    def emit(reason: str):
        nonlocal current_words

        if not current:
            return

        text = "\n\n".join(u["text"] for u in current)

        chunks.append({
            "id": f"semantic_{len(chunks) + 1:04d}",
            "text": text,
            "source": source,
            "page_start": current[0]["page_start"],
            "page_end": current[-1]["page_end"],
            "word_count": current_words,
            "boundary_reason": reason,
        })

        current.clear()
        current_words = 0

    for i, unit in enumerate(units):

        if current:

            if current_words + unit["word_count"] > max_words:
                emit("max_words")

            elif (
                i in boundaries
                and current_words >= min_words
            ):
                emit("semantic_change")

            elif (
                current_words >= target_words
                and similarities[i - 1] <= typical_similarity
            ):
                emit("soft_target")

        current.append(unit)
        current_words += unit["word_count"]

    emit("end_of_document")

    return chunks


def main():

    project_root = Path(__file__).resolve().parent.parent

    # --- CLI ARGUMENTS ---

    parser = argparse.ArgumentParser(
        description="Length-constrained semantic Markdown chunking"
    )

    parser.add_argument(
        "--target-words",
        type=int,
        default=180,
        help="Preferred chunk length (default: 180)"
    )

    parser.add_argument(
        "--min-words",
        type=int,
        default=80,
        help="Minimum words before semantic splitting (default: 80)"
    )

    parser.add_argument(
        "--max-words",
        type=int,
        default=250,
        help="Maximum words per chunk (default: 250)"
    )

    parser.add_argument(
        "--boundary-quantile",
        type=float,
        default=0.25,
        help="Low-similarity boundary quantile (default: 0.25)"
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
        / "chunks" / "semantic_chunks.jsonl",
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

    if not 0 < args.boundary_quantile < 1:
        parser.error("--boundary-quantile must be between 0 and 1")

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    if args.preview_count < 0:
        parser.error("--preview-count must be >= 0")

    # --- LOAD AND CHUNK ---

    markdown = args.input.read_text(encoding="utf-8")

    print("Loading embedding model...")
    model = SentenceTransformer(args.model_name)

    chunks = semantic_chunks(
        markdown,
        model,
        target_words=args.target_words,
        min_words=args.min_words,
        max_words=args.max_words,
        boundary_quantile=args.boundary_quantile,
        batch_size=args.batch_size,
    )

    units = build_units(markdown)

    # --- VALIDATION ---

    if not chunks:
        raise ValueError("No chunks generated.")

    sizes = [c["word_count"] for c in chunks]

    assert sum(sizes) == len(
        markdown.replace(PAGE_MARKER, "").split()
    )

    assert all(
        PAGE_MARKER not in c["text"]
        for c in chunks
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

    # --- DIAGNOSTICS ---

    print("\n=== SEMANTIC CHUNKING ===")
    print(f"Model: {args.model_name}")
    print(f"Target words: {args.target_words}")
    print(f"Minimum words: {args.min_words}")
    print(f"Maximum words: {args.max_words}")
    print(f"Boundary quantile: {args.boundary_quantile}")
    print(f"Batch size: {args.batch_size}")
    print(f"PDF pages: {markdown.count(PAGE_MARKER) + 1}")
    print(f"Text units: {len(units)}")

    print("\n=== CHUNK STATISTICS ===")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Minimum words: {min(sizes)}")
    print(f"Maximum words: {max(sizes)}")
    print(f"Mean words: {mean(sizes):.1f}")
    print(f"Median words: {median(sizes):.1f}")
    print(f"Chunks under 80 words: {sum(n < 80 for n in sizes)}")

    print("\n=== BOUNDARY DECISIONS ===")

    for reason in [
        "semantic_change",
        "soft_target",
        "max_words",
    ]:
        count = sum(
            c["boundary_reason"] == reason
            for c in chunks
        )

        print(f"{reason}: {count}")

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
