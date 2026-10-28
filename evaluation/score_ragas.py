
"""RAGAS evaluation of saved PDF / SQL / mixed RAG responses.

Uses Google Gemini via the official google-genai SDK (not OpenAI).
Does NOT re-run retrieval, SQL, or backend responses.

Pilot:
    python -m evaluation.score_ragas --split development --limit 1 --retry-errors
Full development:
    python -m evaluation.score_ragas --split development
Held-out after backend run:
    python -m evaluation.score_ragas --split held_out
"""

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv
from google import genai
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.metrics.collections import Faithfulness, FactualCorrectness


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "evaluation" / "results"
DEFAULT_MODEL = "gemini-3.5-flash-lite"


class GoogleGeminiJudge(InstructorBaseRagasLLM):
    """Bridge RAGAS structured-output calls to official Google Gemini SDK."""

    def __init__(self, client, model):
        self.client = client
        self.model = model

    @staticmethod
    def _parse(response, response_model):
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, response_model):
            return parsed
        if parsed is not None:
            return response_model.model_validate(parsed)
        text = getattr(response, "text", None)
        if text:
            return response_model.model_validate_json(text)
        raise ValueError("Gemini returned no JSON structured response")

    def generate(self, prompt, response_model):
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": response_model,
                "temperature": 0,
            },
        )
        return self._parse(response, response_model)

    async def agenerate(self, prompt, response_model):
        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": response_model,
                "temperature": 0,
            },
        )
        return self._parse(response, response_model)


def build_contexts(sources):
    """Preserve actual retrieved passages and actual SQL result rows."""
    contexts = []
    for source in sources:
        kind = source.get("type")
        source_id = source.get("id", "?")
        if kind == "pdf":
            passage = source.get("text")
            if passage:
                contexts.append(
                    f"[PDF {source_id}: {source.get('source', 'unknown')}, "
                    f"page {source.get('page_start', '?')}]\n{passage}"
                )
        elif kind == "sql":
            rows = source.get("rows")
            if rows is not None:
                contexts.append(
                    f"[SQLite {source_id}]\n"
                    f"SQL: {source.get('sql', '')}\n"
                    f"Parameters: {json.dumps(source.get('parameters') or {}, ensure_ascii=False)}\n"
                    f"Result rows: {json.dumps(rows, ensure_ascii=False)}"
                )
    return contexts


def score_to_float(result):
    score = float(result.value)
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError(f"Invalid RAGAS score: {score}")
    return round(score, 4)


def load_existing(path):
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as file:
        report = json.load(file)
    return {record["id"]: record for record in report.get("cases", [])}


def save_report(path, *, split, model, eligible_count, records):
    def average(field):
        scores = [r[field] for r in records if r.get(field) is not None]
        return round(mean(scores), 4) if scores else None

    payload = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "input_file": f"backend_eval_{split}.json",
        "judge_provider": "Google Gemini (official google-genai SDK)",
        "judge_model": model,
        "ragas_metrics": ["Faithfulness", "FactualCorrectness (F1)"],
        "total_successful_backend_cases": eligible_count,
        "cases_processed": len(records),
        "faithfulness_scored": sum(r.get("faithfulness") is not None for r in records),
        "factual_correctness_scored": sum(
            r.get("factual_correctness") is not None for r in records
        ),
        "mean_faithfulness": average("faithfulness"),
        "mean_factual_correctness": average("factual_correctness"),
        "methodology": {
            "faithfulness": "RAGAS evidence support using retrieved PDF text and SQL result rows.",
            "factual_correctness": "RAGAS F1 versus manually validated gold answer.",
            "scope": "Only backend cases with status=ok and a nonempty answer.",
            "limitations": [
                "Failed backend cases remain in the backend evaluation and are not scored here.",
                "Missing RAGAS scores are excluded from means; they are not zeros.",
                "LLM judge scores may require human verification.",
            ],
        },
        "cases": records,
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Score saved RAG answers with Gemini + RAGAS"
    )
    parser.add_argument(
        "--split",
        choices=["development", "held_out"],
        default="development",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")

    input_path = RESULTS_DIR / f"backend_eval_{args.split}.json"
    if not input_path.is_file():
        raise SystemExit(f"Backend evaluation file not found: {input_path}")

    load_dotenv(ROOT / ".env")
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise SystemExit("Missing GEMINI_API_KEY in project .env")

    with input_path.open(encoding="utf-8") as file:
        data = json.load(file)

    eligible = [
        case for case in data["cases"]
        if case.get("observed", {}).get("status") == "ok"
        and case.get("observed", {}).get("answer")
    ]
    selected = eligible if args.limit is None else eligible[:args.limit]
    if not selected:
        raise SystemExit("No successful backend answers to evaluate")

    suffix = "" if args.limit is None else f"_pilot{args.limit}"
    output_path = RESULTS_DIR / f"ragas_{args.split}{suffix}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    previous = load_existing(output_path)

    # Reuse a successfully completed pilot-1 when starting the full evaluation.
    if args.limit is None:
        pilot = RESULTS_DIR / f"ragas_{args.split}_pilot1.json"
        for case_id, record in load_existing(pilot).items():
            previous.setdefault(case_id, record)

    client = genai.Client(api_key=key)
    try:
        judge = GoogleGeminiJudge(client=client, model=args.model)
        faithfulness = Faithfulness(llm=judge)
        correctness = FactualCorrectness(llm=judge, mode="f1")
        records = []

        for position, case in enumerate(selected, start=1):
            case_id = case["id"]
            observed = case["observed"]
            record = previous.get(case_id) or {
                "id": case_id,
                "category": case.get("category"),
                "faithfulness": None,
                "factual_correctness": None,
                "errors": {},
            }
            record.setdefault("errors", {})
            record.setdefault("faithfulness", None)
            record.setdefault("factual_correctness", None)
            contexts = build_contexts(observed.get("sources") or [])

            metrics = [
                (
                    "faithfulness",
                    faithfulness,
                    {
                        "user_input": observed.get("resolved_question") or case["question"],
                        "response": observed["answer"],
                        "retrieved_contexts": contexts,
                    },
                ),
                (
                    "factual_correctness",
                    correctness,
                    {
                        "response": observed["answer"],
                        "reference": case["reference_answer"],
                    },
                ),
            ]

            print(f"\n[{position}/{len(selected)}] {case_id}", flush=True)
            for metric_name, scorer, kwargs in metrics:
                if record[metric_name] is None and (
                    args.retry_errors or metric_name not in record["errors"]
                ):
                    try:
                        if metric_name == "faithfulness" and not contexts:
                            raise ValueError("No retrieved evidence available")
                        result = scorer.score(**kwargs)
                        record[metric_name] = score_to_float(result)
                        record["errors"].pop(metric_name, None)
                    except Exception as exc:
                        message = f"{type(exc).__name__}: {exc}"
                        record["errors"][metric_name] = message[:1500]
                        print(
                            f"  ERROR {metric_name}: {message[:500]}",
                            flush=True,
                        )

                print(f"  {metric_name}: {record[metric_name]}", flush=True)

            records.append(record)
            save_report(
                output_path,
                split=args.split,
                model=args.model,
                eligible_count=len(eligible),
                records=records,
            )

        with output_path.open(encoding="utf-8") as file:
            summary = json.load(file)

        print("\n=== RAGAS SUMMARY ===")
        for name in (
            "total_successful_backend_cases",
            "cases_processed",
            "faithfulness_scored",
            "factual_correctness_scored",
            "mean_faithfulness",
            "mean_factual_correctness",
        ):
            print(f"{name}: {summary[name]}")
        print(f"Saved: {output_path}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
