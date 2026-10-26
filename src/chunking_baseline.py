import argparse
import json
import re
from pathlib import Path
from statistics import mean, median


PAGE_MARKER = "<!-- PDF_PAGE_BREAK -->"
TOKEN_PATTERN = re.compile(r"<!-- PDF_PAGE_BREAK -->|\S+")


def fixed_size_chunks(
    markdown: str,
    chunk_size_words: int = 250,
    overlap_words: int = 50,
    source: str = "Core Rules.pdf",
) -> list[dict]:
    """
    Split Markdown into overlapping fixed-size word windows.

    Preserve Markdown formatting and track original PDF pages.
    Page markers are excluded from chunk text and word counts.
    """

    if chunk_size_words <= 0:
        raise ValueError("Chunk size must be positive.")

    if not 0 <= overlap_words < chunk_size_words:
        raise ValueError(
            "Overlap must be non-negative and smaller than chunk size."
        )

    words = []
    page = 1

    for match in TOKEN_PATTERN.finditer(markdown):
        if match.group() == PAGE_MARKER:
            page += 1
        else:
            words.append((match.start(), match.end(), page))

    chunks = []
    step = chunk_size_words - overlap_words

    for start in range(0, len(words), step):
        end = min(start + chunk_size_words, len(words))

        first_word = words[start]
        last_word = words[end - 1]

        chunk_text = markdown[
            first_word[0]:last_word[1]
        ].replace(PAGE_MARKER, "").strip()

        chunks.append({
            "id": f"fixed_{len(chunks) + 1:04d}",
            "text": chunk_text,
            "source": source,
            "page_start": first_word[2],
            "page_end": last_word[2],
            "word_count": end - start,
            "start_word": start,
            "end_word": end,
        })

        if end == len(words):
            break

    return chunks


def main():

    project_root = Path(__file__).resolve().parent.parent

    # --- CLI ARGUMENTS ---

    parser = argparse.ArgumentParser(
        description="Fixed-size Markdown chunking"
    )

    parser.add_argument(
        "--chunk-size-words",
        type=int,
        default=250,
        help="Maximum number of words per chunk (default: 250)"
    )

    parser.add_argument(
        "--overlap-words",
        type=int,
        default=50,
        help="Overlapping words between chunks (default: 50)"
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
        / "chunks" / "fixed_chunks.jsonl",
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

    chunks = fixed_size_chunks(
        markdown,
        chunk_size_words=args.chunk_size_words,
        overlap_words=args.overlap_words,
    )

    # --- VALIDATION ---

    if not chunks:
        raise ValueError("No chunks generated.")

    assert all(PAGE_MARKER not in c["text"] for c in chunks)
    assert all(
        0 < c["word_count"] <= args.chunk_size_words
        for c in chunks
    )
    assert all(
        1 <= c["page_start"] <= c["page_end"]
        for c in chunks
    )

    # --- DIAGNOSTICS ---

    sizes = [c["word_count"] for c in chunks]

    print("\n=== FIXED-SIZE CHUNKING ===")
    print(f"Chunk size: {args.chunk_size_words}")
    print(f"Overlap: {args.overlap_words}")
    print(f"PDF pages: {markdown.count(PAGE_MARKER) + 1}")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Minimum words: {min(sizes)}")
    print(f"Maximum words: {max(sizes)}")
    print(f"Mean words: {mean(sizes):.1f}")
    print(f"Median words: {median(sizes):.1f}")

    print("\n=== CHUNK PREVIEWS ===")

    for chunk in chunks[:args.preview_count]:
        print(f"\nID: {chunk['id']}")
        print(
            f"Pages: {chunk['page_start']}-"
            f"{chunk['page_end']}"
        )
        print(f"Words: {chunk['word_count']}")
        print(f"Preview: {chunk['text'][:250]}")

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
                    json.dumps(chunk, ensure_ascii=False) + "\n"
                )

        print(
            f"\nSaved {len(chunks)} chunks to: {args.output}"
        )


if __name__ == "__main__":
    main()
