
"""Validate benchmark references against PDF and SQLite.

Offline evaluation: no Gemini calls or API costs.

Run:
    python -m evaluation.validate_benchmark_gold
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
import unicodedata

from evaluation.benchmark import (
    CASES,
    VERSION,
    validate_structure,
)


ROOT = Path(__file__).resolve().parents[1]


def locate(filename: str, provided: str | None) -> Path:
    """Find a source file or use an explicit path."""

    candidates = (
        [Path(provided)]
        if provided
        else [
            ROOT / filename,
            ROOT / "data" / filename,
            ROOT / "data" / "raw" / filename,
        ]
    )

    for path in candidates:
        if path.is_file():
            return path.resolve()

    raise FileNotFoundError(
        f"Cannot find {filename}. "
        "Supply its path using --db or --pdf."
    )


def normalize(text: str) -> str:
    """Normalize Unicode, case and whitespace."""

    text = unicodedata.normalize(
        "NFKC", str(text)
    ).casefold()

    return re.sub(r"\s+", " ", text).strip()


def main() -> None:

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--db",
        help="Path to wahadb.sqlite",
    )

    parser.add_argument(
        "--pdf",
        help="Path to Core Rules.pdf",
    )

    args = parser.parse_args()

    # 1. Validate dataset structure.

    validate_structure()

    db_path = locate("wahadb.sqlite", args.db)
    pdf_path = locate("Core Rules.pdf", args.pdf)

    try:
        import fitz  # PyMuPDF

    except ImportError as exc:
        raise SystemExit(
            "Install PyMuPDF: "
            "python -m pip install pymupdf"
        ) from exc

    failures = []
    sql_checks = 0
    pdf_checks = 0

    # 2. Extract and normalize PDF pages.

    with fitz.open(pdf_path) as doc:
        pages = [
            normalize(page.get_text())
            for page in doc
        ]

    # 3. Open SQLite strictly read-only.

    uri = db_path.as_uri() + "?mode=ro"

    with sqlite3.connect(uri, uri=True) as conn:

        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only = ON")

        for case in CASES:

            case_id = case["id"]

            # 4. Verify PDF evidence.

            for evidence in case.get(
                "pdf_evidence", []
            ):

                pdf_checks += 1

                page = evidence["page"]
                anchor = evidence["anchor"]

                if not (1 <= page <= len(pages)):
                    failures.append(
                        f"{case_id}: PDF page "
                        f"{page} does not exist"
                    )

                elif normalize(anchor) not in pages[page - 1]:
                    failures.append(
                        f"{case_id}: missing PDF "
                        f"anchor on page {page}: "
                        f"{anchor!r}"
                    )

            # 5. Verify SQL evidence.

            evidence = case.get("sql_evidence")

            if evidence is None:
                continue

            sql_checks += 1

            sql = evidence["reference_sql"]

            if not re.match(
                r"^\s*SELECT\b",
                sql,
                flags=re.IGNORECASE,
            ):
                failures.append(
                    f"{case_id}: reference query "
                    "is not a SELECT"
                )
                continue

            try:
                actual = [
                    dict(row)
                    for row in conn.execute(
                        sql,
                        evidence["parameters"],
                    )
                ]

            except sqlite3.Error as exc:
                failures.append(
                    f"{case_id}: SQL failed: {exc}"
                )
                continue

            expected = evidence["verified_rows"]

            if actual != expected:
                failures.append(
                    f"{case_id}: SQL mismatch: "
                    f"expected={expected!r}; "
                    f"actual={actual!r}"
                )

    # 6. Generate reproducible validation report.

    report = {
        "benchmark_version": VERSION,
        "checked_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "database": db_path.name,
        "pdf": pdf_path.name,
        "total_cases": len(CASES),
        "sql_queries_checked": sql_checks,
        "pdf_anchors_checked": pdf_checks,
        "negative_cases_without_gold_evidence": sum(
            c["category"] == "unanswerable"
            for c in CASES
        ),
        "passed": not failures,
        "failures": failures,
    }

    output = (
        ROOT
        / "evaluation"
        / "results"
        / "benchmark_gold_validation.json"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Cases: {len(CASES)}")
    print(f"PDF anchors checked: {pdf_checks}")
    print(f"SQL queries checked: {sql_checks}")
    print(
        f"Result: "
        f"{'PASS' if not failures else 'FAIL'}"
    )

    print(
        f"Report: {output.relative_to(ROOT)}"
    )

    for problem in failures:
        print("FAIL:", problem)

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
