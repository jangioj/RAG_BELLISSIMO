
"""End-to-end evaluation runner for the Multi-Channel RAG.

Runs the frozen benchmark against the real backend.

Records:
- Routing and resolved question
- Retrieved PDF and SQL evidence
- Generated answer and citation identifiers
- Token usage and wall-clock latency
- Diagnostic checks

IMPORTANT:
These checks do NOT constitute factual correctness,
faithfulness, or semantic citation verification.
Those require separate manual or RAGAS evaluation.

Examples:
    python -m evaluation.run_backend_eval --limit 3
    python -m evaluation.run_backend_eval
    python -m evaluation.run_backend_eval --split held_out
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from time import perf_counter

from evaluation.benchmark import CASES, VERSION
from src.rag_backend import ask_rag


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "evaluation" / "results"


def source_type_check(case, result):
    """Check citation source types, not factual support."""

    if not case["answerable"]:
        return None

    required = {
        "pdf": {"P"},
        "sql": {"S"},
        "mixed": {"P", "S"},
    }.get(case["expected_route"])

    if required is None:
        return None

    citations = result.get("citations") or []

    actual = {
        citation[0]
        for citation in citations
        if isinstance(citation, str) and citation
    }

    return required.issubset(actual)


def pdf_page_coverage(case, result):
    """Diagnostic proxy: presence of gold PDF pages.

    This checks page coverage, not whether the
    retrieved chunks contain all required facts.
    """

    expected = {
        item["page"]
        for item in case["pdf_evidence"]
    }

    if not expected:
        return None

    retrieved = set()

    for source in result.get("sources") or []:
        if source.get("type") != "pdf":
            continue

        start = source.get("page_start")
        end = source.get("page_end")

        if isinstance(start, int):
            if not isinstance(end, int):
                end = start

            retrieved.update(range(start, end + 1))

    found = sorted(expected & retrieved)

    return {
        "required_pages": sorted(expected),
        "found_pages": found,
        "fraction": len(found) / len(expected),
    }


def citation_id_check(result):
    """Check that cited identifiers exist in evidence."""

    citations = result.get("citations") or []
    sources = result.get("sources") or []

    valid_ids = {
        item.get("id")
        for item in sources
    }

    answer = result.get("answer") or ""

    inline_ids = set(
        re.findall(r"\[([PS]\d+)\]", answer)
    )

    return {
        "identifiers_exist": all(
            citation in valid_ids
            for citation in citations
        ),
        "inline_matches_reported": (
            inline_ids == set(citations)
        ),
    }


def evaluate_case(case):
    """Run one benchmark case, including optional history."""

    started = perf_counter()
    seed_result = None

    try:
        history = []

        if case.get("history_seed_question"):
            seed_result = ask_rag(
                case["history_seed_question"]
            )

            if seed_result["status"] != "ok":
                result = {
                    "status": "seed_failed",
                    "route": None,
                    "answer": None,
                    "citations": [],
                    "sources": [],
                    "usage": None,
                    "error": seed_result.get("error"),
                }

                return build_record(
                    case,
                    result,
                    perf_counter() - started,
                    seed_result,
                )

            history = seed_result["history"]

        result = ask_rag(
            question=case["question"],
            history=history,
        )

    except Exception as exc:
        result = {
            "status": "exception",
            "route": None,
            "answer": None,
            "citations": [],
            "sources": [],
            "usage": None,
            "error": {
                "code": "EVALUATION_EXCEPTION",
                "message": type(exc).__name__,
            },
        }

    return build_record(
        case,
        result,
        perf_counter() - started,
        seed_result,
    )


def build_record(case, result, elapsed, seed_result):
    """Combine benchmark gold data and observed output."""

    expected_route = case["expected_route"]
    actual_route = result.get("route")

    route_correct = (
        actual_route == expected_route
        if expected_route is not None
        else None
    )

    return {
        "id": case["id"],
        "split": case["split"],
        "category": case["category"],
        "question": case["question"],
        "reference_answer": case["reference_answer"],
        "expected_route": expected_route,
        "answerable": case["answerable"],
        "gold_pdf_evidence": case["pdf_evidence"],
        "gold_sql_evidence": case["sql_evidence"],
        "history_seed_question": case.get(
            "history_seed_question"
        ),
        "seed_status": (
            seed_result.get("status")
            if seed_result is not None
            else None
        ),
        "observed": {
            "status": result.get("status"),
            "route": actual_route,
            "resolved_question": result.get(
                "resolved_question"
            ),
            "answer": result.get("answer"),
            "citations": result.get("citations"),
            "sources": result.get("sources"),
            "usage": result.get("usage"),
            "error": result.get("error"),
        },
        "diagnostics": {
            "route_correct": route_correct,
            "required_citation_types_present":
                source_type_check(case, result),
            "pdf_page_coverage_proxy":
                pdf_page_coverage(case, result),
            "citation_identifiers":
                citation_id_check(result),
        },
        "elapsed_seconds": round(elapsed, 2),
        "human_review": {
            "answer_correct": None,
            "evidence_sufficient": None,
            "citations_support_claims": None,
            "notes": "",
        },
    }


def summarize(records):
    """Calculate simple, explicitly limited diagnostics."""

    labeled = [
        record for record in records
        if record["diagnostics"]["route_correct"]
        is not None
    ]

    routes_correct = sum(
        record["diagnostics"]["route_correct"]
        is True
        for record in labeled
    )

    return {
        "tested": len(records),
        "status_ok": sum(
            record["observed"]["status"] == "ok"
            for record in records
        ),
        "route_correct": routes_correct,
        "route_labeled": len(labeled),
        "routing_accuracy": (
            routes_correct / len(labeled)
            if labeled else None
        ),
        "factual_correctness": None,
        "faithfulness": None,
        "note": (
            "Routing accuracy is automatic. "
            "Answer correctness and faithfulness "
            "have not yet been evaluated."
        ),
    }


def save_report(path, records, split):
    """Persist results after every completed case."""

    report = {
        "benchmark_version": VERSION,
        "evaluated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "split": split,
        "metrics": summarize(records),
        "cases": records,
    }

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=["development", "held_out", "all"],
        default="development",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of cases (pilot runs).",
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    selected = [
        case for case in CASES
        if args.split == "all"
        or case["split"] == args.split
    ]

    if args.limit is not None:
        selected = selected[:args.limit]

    suffix = (
        f"_pilot{args.limit}"
        if args.limit is not None
        else ""
    )

    output = (
        RESULTS_DIR
        / f"backend_eval_{args.split}{suffix}.json"
    )

    records = []

    for index, case in enumerate(selected, 1):
        print(
            f"\n[{index}/{len(selected)}] "
            f"{case['id']}: {case['question']}",
            flush=True,
        )

        record = evaluate_case(case)
        records.append(record)

        save_report(output, records, args.split)

        observed = record["observed"]
        checks = record["diagnostics"]

        print(
            f"Status: {observed['status']} | "
            f"Route: {observed['route']} | "
            f"Route correct: {checks['route_correct']}"
        )

        print(
            f"Elapsed: "
            f"{record['elapsed_seconds']}s"
        )

        print(
            f"Answer: "
            f"{observed['answer']}"
        )

    summary = summarize(records)

    print("\n=== AUTOMATIC DIAGNOSTICS ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
