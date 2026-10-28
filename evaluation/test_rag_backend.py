
"""Unit tests for the Multi-Channel RAG backend."""

import json
import os
import unittest

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.rag_backend import ask_rag


def fake_route(route="pdf"):
    return {
        "status": "ok",
        "route": route,
        "resolved_question": "Test question",
        "pdf_question": "General gameplay rule?",
        "sql_question": "Specific unit statistic?",
        "usage": None,
        "error": None,
    }


def fake_pdf():
    return [{
        "id": "chunk_001",
        "source": "Core Rules.pdf",
        "text": "A test rule from the Core Rules.",
        "page_start": 10,
        "page_end": 10,
        "score": 0.03,
        "method": "rrf",
    }]


class TestRagBackend(unittest.TestCase):

    @patch("src.rag_backend.search_rrf")
    @patch("src.rag_backend.route_query")
    def test_pdf_retrieval_error(self, router, retriever):
        router.return_value = fake_route("pdf")
        retriever.side_effect = RuntimeError("Test failure")

        result = ask_rag("Test question", api_key="test-key")

        self.assertEqual(result["status"], "error")
        self.assertEqual(
            result["error"]["code"],
            "PDF_RETRIEVAL_ERROR",
        )

    @patch("src.rag_backend.query_database")
    @patch("src.rag_backend.route_query")
    def test_empty_sql_result(self, router, database):
        router.return_value = fake_route("sql")
        database.return_value = {
            "status": "empty",
            "sql": "SELECT name FROM datasheets",
            "parameters": {},
            "columns": ["name"],
            "rows": [],
            "row_count": 0,
            "tables": ["datasheets"],
            "truncated": False,
            "usage": None,
        }

        result = ask_rag("Test question", api_key="test-key")

        self.assertEqual(
            result["status"], "insufficient_evidence"
        )
        self.assertIsNone(result["error"])
        self.assertEqual(result["sources"][0]["id"], "S1")

    @patch("src.rag_backend.query_database")
    @patch("src.rag_backend.route_query")
    def test_sql_error_normalization(self, router, database):
        router.return_value = fake_route("sql")
        database.return_value = {
            "status": "error",
            "error": "GEMINI_ERROR",
            "usage": None,
        }

        result = ask_rag("Test question", api_key="test-key")

        self.assertEqual(result["status"], "error")
        self.assertEqual(
            result["error"]["code"], "GEMINI_ERROR"
        )
        self.assertEqual(
            result["error"]["source"], "sql_retriever"
        )

    @patch("src.rag_backend.search_rrf")
    @patch("src.rag_backend.route_query")
    def test_missing_api_key(self, router, retriever):
        router.return_value = fake_route("pdf")
        retriever.return_value = fake_pdf()

        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            result = ask_rag("Test question")

        self.assertEqual(result["status"], "error")
        self.assertEqual(
            result["error"]["code"], "MISSING_API_KEY"
        )

    @patch("src.rag_backend.genai.Client")
    @patch("src.rag_backend.search_rrf")
    @patch("src.rag_backend.route_query")
    def test_invalid_citation(self, router, retriever, client_class):
        router.return_value = fake_route("pdf")
        retriever.return_value = fake_pdf()

        response = SimpleNamespace(
            text=json.dumps({
                "answer": "Unsupported claim [P99].",
                "source_ids": ["P99"],
                "answerable": True,
            }),
            usage_metadata=SimpleNamespace(
                prompt_token_count=10,
                candidates_token_count=5,
                total_token_count=15,
            ),
        )

        client = MagicMock()
        client.models.generate_content.return_value = response
        client_class.return_value = client

        result = ask_rag(
            "Test question",
            api_key="test-key",
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(
            result["error"]["code"],
            "INVALID_ANSWER_OUTPUT",
        )


if __name__ == "__main__":
    unittest.main()
