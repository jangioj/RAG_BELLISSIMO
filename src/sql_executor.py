
"""Bounded, read-only SQLite queries for RAG_ANGIOLILLO."""

from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path

DB_PATH = (
    Path(__file__).resolve().parents[1]
    / "data" / "raw" / "wahadb.sqlite"
)

ROW_LIMIT = 100
TIME_LIMIT_SECONDS = 3.0
VM_STEP_LIMIT = 2_000_000
PROGRESS_INTERVAL = 1_000

ALLOWED_TABLES = {
    "abilities", "datasheets", "datasheets_abilities",
    "datasheets_detachment_abilities", "datasheets_enhancements",
    "datasheets_keywords", "datasheets_leader", "datasheets_models",
    "datasheets_models_cost", "datasheets_options",
    "datasheets_stratagems", "datasheets_unit_composition",
    "datasheets_wargear", "detachment_abilities", "detachments",
    "detachments_chapter_dp", "enhancements", "factions",
    "last_update", "source", "stratagems",
}

ALLOWED_FUNCTIONS = {
    "abs", "avg", "coalesce", "count", "glob", "group_concat",
    "ifnull", "length", "like", "lower", "max", "min",
    "nullif", "round", "substr", "sum", "trim", "upper",
}


def execute_readonly(
    sql: str,
    parameters: dict | None = None,
    db_path: str | Path | None = None,
) -> dict:
    """Run one bounded read-only SQL statement."""

    params = {} if parameters is None else parameters

    result = {
        "status": "error",
        "sql": sql,
        "parameters": params,
        "columns": [],
        "rows": [],
        "row_count": 0,
        "tables": [],
        "truncated": False,
        "error": None,
    }

    def fail(code: str, message: str) -> dict:
        result["error"] = {"code": code, "message": message}
        return result

    # Preliminary filter; the SQLite authorizer is the security boundary.
    if not isinstance(sql, str) or len(sql) > 12_000 or not re.match(
        r"^\s*(SELECT|WITH)\b", sql, re.IGNORECASE
    ):
        return fail(
            "UNAUTHORIZED_SQL",
            "Only one SELECT query is allowed.",
        )

    if not isinstance(params, dict) or any(
        not isinstance(k, str)
        or not isinstance(v, (str, int, float, type(None)))
        for k, v in params.items()
    ):
        return fail(
            "INVALID_PARAMETERS",
            "Use a dict of named, scalar parameters.",
        )

    path = Path(db_path) if db_path is not None else DB_PATH

    if not path.is_file():
        return fail(
            "DB_UNAVAILABLE",
            "SQLite database file not found.",
        )

    used_tables = set()
    denied = False
    budget_exceeded = False
    callbacks = 0
    connection = None
    start = time.monotonic()

    def authorizer(action, obj, other, schema, source):
        nonlocal denied

        if action == sqlite3.SQLITE_SELECT:
            return sqlite3.SQLITE_OK

        if action == sqlite3.SQLITE_READ:
            if schema in ("main", None) and obj in ALLOWED_TABLES:
                used_tables.add(obj)
                return sqlite3.SQLITE_OK

        if action == sqlite3.SQLITE_FUNCTION:
            if (
                isinstance(other, str)
                and other.lower() in ALLOWED_FUNCTIONS
            ):
                return sqlite3.SQLITE_OK

        denied = True
        return sqlite3.SQLITE_DENY

    def progress():
        nonlocal callbacks, budget_exceeded

        callbacks += 1

        if (
            callbacks * PROGRESS_INTERVAL >= VM_STEP_LIMIT
            or time.monotonic() - start >= TIME_LIMIT_SECONDS
        ):
            budget_exceeded = True
            return 1

        return 0

    try:
        uri = path.resolve().as_uri() + "?mode=ro"

        connection = sqlite3.connect(
            uri, uri=True, timeout=2
        )

        connection.execute("PRAGMA query_only=ON")
        connection.set_authorizer(authorizer)
        connection.set_progress_handler(
            progress, PROGRESS_INTERVAL
        )

        # sqlite3.execute accepts exactly one SQL statement.
        cursor = connection.execute(sql, params)

        columns = [info[0] for info in cursor.description]

        if len(columns) != len(set(columns)):
            return fail(
                "DUPLICATE_COLUMNS",
                "Use unique SQL column aliases.",
            )

        records = cursor.fetchmany(ROW_LIMIT + 1)

        if any(
            isinstance(v, bytes)
            for record in records
            for v in record
        ):
            return fail(
                "UNSUPPORTED_RESULT",
                "Binary results are unsupported.",
            )

        result["columns"] = columns
        result["truncated"] = len(records) > ROW_LIMIT

        result["rows"] = [
            dict(zip(columns, record))
            for record in records[:ROW_LIMIT]
        ]

        result["row_count"] = len(result["rows"])

        result["status"] = (
            "limited" if result["truncated"]
            else "ok" if result["rows"]
            else "empty"
        )

    except sqlite3.Error as exc:
        if budget_exceeded:
            fail(
                "RESOURCE_LIMIT",
                "SQLite execution exceeded its budget.",
            )
        elif denied:
            fail(
                "UNAUTHORIZED_SQL",
                "SQL operation or resource not allowed.",
            )
        else:
            fail(
                "SQL_ERROR",
                f"SQLite error: {str(exc)[:180]}",
            )

    finally:
        result["tables"] = sorted(used_tables)

        if connection is not None:
            connection.close()

    return result
