
"""Offline scoring of the development evaluation"""

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "evaluation/results/backend_eval_development.json"
OUTPUT = ROOT / "evaluation/results/development_score.json"


def extract_citations(answer):
    """Recognize [P1], [S1], and grouped citations [P1, S1]."""
    result = set()
    for group in re.findall(
        r"\[([PS]\d+(?:\s*,\s*[PS]\d+)*)\]",
        answer or "",
    ):
        result.update(
            part.strip() for part in group.split(",")
        )
    return result


def score_case(case):
    observed = case["observed"]
    status = observed.get("status")
    route = observed.get("route")
    expected = case["expected_route"]

    sources = observed.get("sources") or []
    valid_ids = {s["id"] for s in sources}
    reported = set(observed.get("citations") or [])
    inline = extract_citations(observed.get("answer"))

    route_produced = route is not None
    route_match = (
        route == expected if route_produced else None
    )

    # M06 already supplies the database characteristic in
    # the question; a PDF-only answer is also sufficient.
    acceptable_routes = {expected} if expected else set()
    if case["id"] == "M06":
        acceptable_routes.add("pdf")

    return {
        "id": case["id"],
        "category": case["category"],
        "status": status,
        "expected_route": expected,
        "actual_route": route,
        "route_match": route_match,
        "route_acceptable": (
            route in acceptable_routes
            if route_produced else None
        ),
        "error_code": (
            (observed.get("error") or {}).get("code")
        ),
        "citations_valid": (
            reported == inline
            and reported.issubset(valid_ids)
            if status in {"ok", "insufficient_evidence"}
            else None
        ),
        "citation_claim_support": None,
        "factual_correctness": None,
        "faithfulness": None,
    }


def main():
    with INPUT.open(encoding="utf-8") as f:
        data = json.load(f)

    scored = [score_case(c) for c in data["cases"]]

    routed = [
        c for c in scored
        if c["actual_route"] is not None
    ]

    successful = [
        c for c in scored if c["status"] == "ok"
    ]

    summary = {
        "benchmark_version": data["benchmark_version"],
        "split": "development",
        "total": len(scored),
        "status_counts": dict(
            Counter(c["status"] for c in scored)
        ),
        "routing_attempts_with_result": len(routed),
        "routing_nominal_correct": sum(
            c["route_match"] is True for c in routed
        ),
        "routing_acceptable": sum(
            c["route_acceptable"] is True for c in routed
        ),
        "successful_answers": len(successful),
        "citation_format_valid_among_successes": sum(
            c["citations_valid"] is True
            for c in successful
        ),
        "gemini_error_cases": [
            c["id"] for c in scored
            if c["error_code"] == "GEMINI_ERROR"
        ],
        "known_issues": {
            "M05": "SQL retrieved the wrong Intercessor variant.",
            "M06": "PDF-only is an acceptable alternative route.",
            "T01": "Weapon name contains character-encoding errors.",
            "P01": "Answer omits the 1CP cost in the reference.",
        },
        "limitations": [
            "Routing scores exclude cases without a route.",
            "M06 alternative routing is separately adjudicated.",
            "Citation validity checks IDs and syntax only.",
            "Factual correctness and faithfulness remain unscored.",
            "No held-out cases are included.",
        ],
    }

    report = {
        "summary": summary,
        "cases": scored,
    }

    OUTPUT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
