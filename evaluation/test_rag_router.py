import unittest

from src.rag_router import route_question


class TestRagRouter(unittest.TestCase):

    def check_routes(self, cases):
        for question, expected in cases:
            with self.subTest(question=question):
                actual = route_question(question)
                self.assertEqual(actual, expected)

    def test_pdf_only(self):
        cases = [
            ("How does the Fight phase work?", "pdf"),
            ("When can a unit Advance?", "pdf"),
            ("How are Wound Rolls determined?", "pdf"),
            ("What is Engagement Range?", "pdf"),
            ("How does the Shooting phase work?", "pdf"),
        ]
        self.check_routes(cases)

    def test_sql_only(self):
        cases = [
            ("What is Abaddon's Toughness?", "sql"),
            ("What weapons does Abaddon have?", "sql"),
            ("List the keywords of a Terminator unit.", "sql"),
            ("What abilities does Abaddon have?", "sql"),
            ("What are Abaddon's weapon stats?", "sql"),
        ]
        self.check_routes(cases)

    def test_mixed(self):
        cases = [
            (
                "What result is needed to wound Abaddon "
                "with a Strength 10 weapon?",
                "mixed",
            ),
            (
                "Can Abaddon Advance and still fire his weapons?",
                "mixed",
            ),
            (
                "How do the Core Rules apply to "
                "Abaddon's Toughness?",
                "mixed",
            ),
        ]
        self.check_routes(cases)

    def test_challenging_queries(self):
        cases = [
            # Unit characteristic without an explicit SQL keyword.
            ("What is Abaddon's Movement value?", "sql"),

            # General rule with no explicit phase keyword.
            ("Explain the Overwatch rule.", "pdf"),

            # General rule using terms also present in datasheets.
            (
                "How is Strength compared to Toughness "
                "when making a Wound Roll?",
                "pdf",
            ),

            # Unit-specific interaction with a general rule.
            (
                "Can Terminators shoot after Advancing?",
                "mixed",
            ),
        ]
        self.check_routes(cases)

    def test_ambiguous_question(self):
        self.assertEqual(
            route_question("Tell me about Warhammer 40,000."),
            "mixed",
        )

    def test_invalid_input(self):
        for question in ["", "   ", None, 123]:
            with self.subTest(question=question):
                with self.assertRaises(ValueError):
                    route_question(question)


if __name__ == "__main__":
    unittest.main()
