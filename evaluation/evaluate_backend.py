
import json
from pathlib import Path
from src.rag_backend import ask_rag

CASES = [
    ("PDF_1", "Can a unit that Advanced shoot if it carries an Assault weapon? Can it charge?", "pdf"),
    ("PDF_2", "Can a model with a 3+ Save benefit from cover against an AP 0 ranged attack?", "pdf"),
    ("SQL_1", "What is Abaddon's Toughness?", "sql"),
    ("SQL_2", "What weapons does Abaddon have?", "sql"),
    ("MIXED_1", "What roll is needed to wound Abaddon with a Strength 10 weapon?", "mixed"),
    ("MIXED_2", "What roll is needed to wound Abaddon with a Strength 5 weapon?", "mixed"),
]

results = []

for case_id, question, expected_route in CASES:
    print(f"\nRunning {case_id}...", flush=True)

    result = ask_rag(question)

    citations = result.get("citations") or []
    citation_types = {c[0] for c in citations if c}
    required = {
        "pdf": {"P"},
        "sql": {"S"},
        "mixed": {"P", "S"},
    }[expected_route]

    route_ok = result["route"] == expected_route
    citations_ok = required.issubset(citation_types)

    record = {
        "id": case_id,
        "question": question,
        "expected_route": expected_route,
        "actual_route": result["route"],
        "status": result["status"],
        "route_ok": route_ok,
        "citations_ok": citations_ok,
        "answer": result["answer"],
        "citations": citations,
        "error": result["error"],
        "usage": result["usage"],
    }
    results.append(record)

    print(f"Route: {result['route']} | Correct: {route_ok}")
    print(f"Citations: {citations} | Expected types present: {citations_ok}")
    print(f"Answer: {result['answer']}")

output = Path("evaluation/backend_eval_results.json")
output.write_text(
    json.dumps(results, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print(f"\nResults saved to {output}")
print(f"Successful routes: {sum(r['route_ok'] for r in results)}/{len(results)}")
print(f"Required citation types present: {sum(r['citations_ok'] for r in results)}/{len(results)}")
