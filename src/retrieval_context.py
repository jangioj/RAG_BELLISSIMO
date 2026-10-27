
"""Expand RRF results with adjacent PDF chunks.

Adds immediate neighbours of the top-ranked results when
they share a source and at least one page.

This is a context-assembly step, not a change to RRF ranking.
"""

import argparse

from src.retrieval_common import load_chunks, format_result
from src.retrieval_rrf import search_rrf


def retrieve_with_context(
    question: str,
    strategy: str = "structured",
    k: int = 5,
    expand_top: int = 2,
    candidate_k: int = 20,
) -> list[dict]:

    if not 0 <= expand_top <= k:
        raise ValueError("expand_top must be between 0 and k.")

    # Original ranked retrieval
    results = search_rrf(
        question=question,
        strategy=strategy,
        k=k,
        candidate_k=candidate_k,
    )

    if expand_top == 0:
        return results

    chunks = load_chunks(strategy=strategy)

    positions = {
        chunk["id"]: index
        for index, chunk in enumerate(chunks)
    }

    # Preserve all original top-k results.
    context = list(results)
    seen = {result["id"] for result in results}

    for result in results[:expand_top]:
        index = positions[result["id"]]

        for neighbour_index in (index - 1, index + 1):

            if not 0 <= neighbour_index < len(chunks):
                continue

            neighbour = chunks[neighbour_index]

            if neighbour["id"] in seen:
                continue

            if neighbour["source"] != result["source"]:
                continue

            # Require overlapping page ranges.
            if (
                neighbour["page_start"] > result["page_end"]
                or neighbour["page_end"] < result["page_start"]
            ):
                continue

            expanded = format_result(
                chunk=neighbour,
                score=0.0,
                method="context_neighbor",
                strategy=strategy,
            )

            expanded["neighbor_of"] = result["id"]

            context.append(expanded)
            seen.add(neighbour["id"])

    return context


def main():

    parser = argparse.ArgumentParser(
        description="RRF with adjacent chunk expansion"
    )

    parser.add_argument("question")

    parser.add_argument(
        "--strategy",
        default="structured",
    )

    parser.add_argument("--k", type=int, default=5)

    parser.add_argument(
        "--expand-top",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--show-text",
        action="store_true",
    )

    args = parser.parse_args()

    results = retrieve_with_context(
        question=args.question,
        strategy=args.strategy,
        k=args.k,
        expand_top=args.expand_top,
    )

    print("\n=== RRF + CONTEXT EXPANSION ===")
    print(f"Question: {args.question}")
    print(f"Original top-k: {args.k}")
    print(f"Total context chunks: {len(results)}")

    for result in results:

        if result["method"] == "context_neighbor":
            origin = f"NEIGHBOR of {result['neighbor_of']}"
        else:
            origin = "RRF"

        print(
            f"\n{result['id']} | {origin} | "
            f"pages {result['page_start']}-{result['page_end']}"
        )

        text = result["text"]

        if not args.show_text:
            text = text[:250] + (
                "..." if len(text) > 250 else ""
            )

        print(text)

    print(
        "\nTotal context words:",
        sum(result["word_count"] for result in results),
    )


if __name__ == "__main__":
    main()
