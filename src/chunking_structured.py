import argparse
import json
import re
from pathlib import Path
from statistics import mean, median


PAGE_MARKER = "<!-- PDF_PAGE_BREAK -->"
HEADING_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*$")


def parse_blocks(markdown: str) -> list[dict]:
    """Extract text blocks with page and heading metadata."""

    blocks = []
    page = 1
    buffer = []
    start_page = 1

    def flush():
        if buffer:
            content = "\n".join(buffer).strip()

            if content:
                blocks.append({
                    "text": content,
                    "page_start": start_page,
                    "page_end": page,
                    "word_count": len(content.split()),
                })

            buffer.clear()

    for line in markdown.splitlines():

        if line.strip() == PAGE_MARKER:
            flush()
            page += 1

        elif not line.strip():
            flush()

        elif HEADING_RE.fullmatch(line.strip()):
            flush()

            blocks.append({
                "text": line.strip(),
                "page_start": page,
                "page_end": page,
                "word_count": len(line.split()),
            })

        else:
            if not buffer:
                start_page = page

            buffer.append(line)

    flush()

    # Track the active heading hierarchy.
    heading_tree = {}

    for block in blocks:
        match = HEADING_RE.fullmatch(block["text"])

        if match:
            level = len(match.group(1))

            heading_tree = {
                k: v for k, v in heading_tree.items()
                if k < level
            }

            heading_tree[level] = match.group(2)

        block["heading_path"] = [
            heading_tree[k] for k in sorted(heading_tree)
        ]

    return blocks


def structure_aware_chunks(
    markdown: str,
    target_words: int = 250,
    min_section_fill: float = 0.45,
    source: str = "Core Rules.pdf",
) -> list[dict]:
    """Group coherent Markdown blocks into retrieval chunks."""

    if target_words <= 0:
        raise ValueError("target_words must be positive")

    if not 0 < min_section_fill < 1:
        raise ValueError(
            "min_section_fill must be between 0 and 1"
        )

    blocks = parse_blocks(markdown)

    chunks = []
    current = []
    pending = []
    word_total = 0
    section = "Core Rules"

    def emit():
        nonlocal word_total

        if not current:
            return

        joined = "\n\n".join(
            b["text"] for b in current
        )

        headings = [
            match.group(2)
            for b in current
            if (
                match := HEADING_RE.fullmatch(b["text"])
            )
        ]

        chunks.append({
            "id": f"structured_{len(chunks) + 1:04d}",
            "text": joined,
            "source": source,
            "page_start": current[0]["page_start"],
            "page_end": current[-1]["page_end"],
            "section": section,
            "heading_path": current[0]["heading_path"],
            "headings_in_chunk": headings,
            "word_count": word_total,
        })

        current.clear()
        word_total = 0

    def add(block):
        nonlocal word_total

        if (
            current
            and word_total + block["word_count"] > target_words
        ):
            emit()

        if block["word_count"] <= target_words:
            current.append(block)
            word_total += block["word_count"]
            return

        # Fallback for exceptionally long blocks.
        matches = list(
            re.finditer(r"\S+", block["text"])
        )

        for start in range(0, len(matches), target_words):
            group = matches[start:start + target_words]

            part = block.copy()
            part["text"] = block["text"][
                group[0].start():group[-1].end()
            ]
            part["word_count"] = len(group)

            if (
                current
                and word_total + part["word_count"] > target_words
            ):
                emit()

            current.append(part)
            word_total += part["word_count"]

    for block in blocks:
        heading = HEADING_RE.fullmatch(block["text"])

        if heading and len(heading.group(1)) <= 2:

            # Major headings begin a new section.
            for pending_block in pending:
                add(pending_block)

            pending.clear()
            emit()

            section = heading.group(2)
            pending.append(block)

        elif heading:

            # Prefer a subsection boundary when the
            # current chunk is sufficiently populated.
            if word_total >= target_words * min_section_fill:
                emit()

            pending.append(block)

        else:

            # Attach headings to following content.
            for pending_block in pending:
                add(pending_block)

            pending.clear()
            add(block)

    for pending_block in pending:
        add(pending_block)

    emit()

    return chunks


def main():

    project_root = Path(__file__).resolve().parent.parent

    # --- CLI ARGUMENTS ---

    parser = argparse.ArgumentParser(
        description="Structure-aware Markdown chunking"
    )

    parser.add_argument(
        "--target-words",
        type=int,
        default=250,
        help="Maximum target words per chunk (default: 250)"
    )

    parser.add_argument(
        "--min-section-fill",
        type=float,
        default=0.45,
        help="Minimum section fill ratio (default: 0.45)"
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
        / "chunks" / "structured_chunks.jsonl",
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

    if args.preview_count < 0:
        parser.error("--preview-count must be >= 0")

    # --- LOAD AND CHUNK ---

    markdown = args.input.read_text(encoding="utf-8")

    blocks = parse_blocks(markdown)

    chunks = structure_aware_chunks(
        markdown,
        target_words=args.target_words,
        min_section_fill=args.min_section_fill,
    )

    # --- VALIDATION ---

    if not chunks:
        raise ValueError("No chunks generated.")

    assert sum(
        c["word_count"] for c in chunks
    ) == sum(
        b["word_count"] for b in blocks
    )

    assert all(
        PAGE_MARKER not in c["text"]
        for c in chunks
    )

    assert all(
        0 < c["word_count"] <= args.target_words
        for c in chunks
    )

    assert all(
        1 <= c["page_start"] <= c["page_end"]
        for c in chunks
    )

    assert len({c["id"] for c in chunks}) == len(chunks)

    # --- DIAGNOSTICS ---

    sizes = [c["word_count"] for c in chunks]

    print("\n=== STRUCTURE-AWARE CHUNKING ===")
    print(f"Target words: {args.target_words}")
    print(f"Minimum section fill: {args.min_section_fill}")
    print(f"PDF pages: {markdown.count(PAGE_MARKER) + 1}")
    print(f"Number of blocks: {len(blocks)}")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Minimum words: {min(sizes)}")
    print(f"Maximum words: {max(sizes)}")
    print(f"Mean words: {mean(sizes):.1f}")
    print(f"Median words: {median(sizes):.1f}")
    print(f"Chunks under 100 words: {sum(n < 100 for n in sizes)}")

    print("\n=== CHUNK PREVIEWS ===")

    for chunk in chunks[:args.preview_count]:

        print(f"\nID: {chunk['id']}")

        print(
            f"Pages: {chunk['page_start']}-"
            f"{chunk['page_end']}"
        )

        print(f"Words: {chunk['word_count']}")
        print(f"Section: {chunk['section']}")
        print(f"Heading path: {chunk['heading_path']}")
        print("Preview:")
        print(chunk["text"][:250])

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
