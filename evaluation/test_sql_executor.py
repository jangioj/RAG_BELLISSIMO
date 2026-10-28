
"""SQLite safety and regression tests."""

import unittest

from src.sql_executor import execute_readonly


UNIT_ID = "000000924"


class TestSafeSQLite(unittest.TestCase):

    def query(self, sql, params=None):
        return execute_readonly(sql, params)

    def test_abaddon_stats(self):
        result = self.query("""
            SELECT
                d.Id AS id,
                d.name,
                d.factionId AS faction,
                m.t AS toughness
            FROM datasheets d
            JOIN datasheets_models m
                ON m.DatasheetId = d.Id
            WHERE d.Id = :id
        """, {"id": UNIT_ID})

        self.assertEqual(result["status"], "ok", result["error"])
        self.assertEqual(result["rows"], [{
            "id": UNIT_ID,
            "name": "Abaddon The Despoiler",
            "faction": "CSM",
            "toughness": "5",
        }])

    def test_abilities(self):
        result = self.query("""
            SELECT COALESCE(
                NULLIF(TRIM(da.name), ''),
                NULLIF(TRIM(a.name), '')
            ) AS ability
            FROM datasheets d
            JOIN datasheets_abilities da
                ON da.DatasheetId = d.Id
            LEFT JOIN abilities a
                ON a.Id = da.abilityId
                AND (
                    a.factionId = d.factionId
                    OR a.factionId = ''
                )
            WHERE d.Id = :id
        """, {"id": UNIT_ID})

        self.assertEqual(result["status"], "ok", result["error"])
        self.assertEqual(len(result["rows"]), 9)
        self.assertIn(
            "Dark Destiny",
            [row["ability"] for row in result["rows"]],
        )
        self.assertTrue(
            all(row["ability"] for row in result["rows"])
        )

    def test_keywords_and_wargear(self):
        result = self.query("""
            SELECT keyword
            FROM datasheets_keywords
            WHERE DatasheetId = :id
        """, {"id": UNIT_ID})

        self.assertEqual(result["status"], "ok", result["error"])
        self.assertEqual(len(result["rows"]), 8)
        self.assertIn(
            "Infantry",
            [row["keyword"] for row in result["rows"]],
        )

        result = self.query("""
            SELECT name, type
            FROM datasheets_wargear
            WHERE DatasheetId = :id
        """, {"id": UNIT_ID})

        self.assertEqual(result["status"], "ok", result["error"])
        self.assertEqual(len(result["rows"]), 3)
        self.assertIn(
            {"name": "Talon of Horus", "type": "Ranged"},
            result["rows"],
        )

    def test_missing_and_injection(self):
        result = self.query(
            "SELECT name FROM datasheets WHERE Id = :id",
            {"id": "non-existing"},
        )
        self.assertEqual(result["status"], "empty")

        result = self.query(
            "SELECT Id FROM datasheets WHERE name = :name",
            {"name": "Abaddon' OR 1=1 --"},
        )
        self.assertEqual(result["status"], "empty")

    def test_row_limit(self):
        result = self.query(
            "SELECT Id FROM datasheets ORDER BY Id"
        )
        self.assertEqual(result["status"], "limited")
        self.assertEqual(result["row_count"], 100)
        self.assertTrue(result["truncated"])

    def test_cte_and_like(self):
        result = self.query("""
            WITH matches AS (
                SELECT Id
                FROM datasheets
                WHERE name LIKE :pattern
            )
            SELECT Id FROM matches
        """, {"pattern": "%Abaddon%"})

        self.assertEqual(result["status"], "ok", result["error"])

    def test_forbidden_queries(self):
        cases = [
            ("DELETE FROM datasheets", "UNAUTHORIZED_SQL"),
            (
                "WITH x AS (SELECT 1) DELETE FROM datasheets",
                "UNAUTHORIZED_SQL",
            ),
            (
                "ATTACH DATABASE ':memory:' AS x",
                "UNAUTHORIZED_SQL",
            ),
            ("PRAGMA user_version=1", "UNAUTHORIZED_SQL"),
            (
                "SELECT load_extension('x')",
                "UNAUTHORIZED_SQL",
            ),
            (
                "SELECT name FROM sqlite_master",
                "UNAUTHORIZED_SQL",
            ),
            (
                "SELECT 1; DELETE FROM datasheets",
                "SQL_ERROR",
            ),
        ]

        for sql, expected_code in cases:
            with self.subTest(sql=sql):
                result = self.query(sql)
                self.assertEqual(result["status"], "error")
                self.assertEqual(
                    result["error"]["code"],
                    expected_code,
                )

    def test_work_limit(self):
        result = self.query("""
            SELECT COUNT(*)
            FROM datasheets a
            CROSS JOIN datasheets b
            CROSS JOIN datasheets c
        """)

        self.assertEqual(
            result["error"]["code"],
            "RESOURCE_LIMIT",
        )

    def test_duplicate_columns(self):
        result = self.query(
            "SELECT name, name FROM datasheets LIMIT 1"
        )

        self.assertEqual(
            result["error"]["code"],
            "DUPLICATE_COLUMNS",
        )


if __name__ == "__main__":
    unittest.main()
