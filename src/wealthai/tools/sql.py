"""Read-only SQL tool over the synthetic portfolio database.

The LLM writes the SQL, so we never trust it. Four independent layers:
1. The file is opened read-only (URI mode=ro) and with PRAGMA query_only=ON.
2. A SQLite authorizer allows only SELECT, only the tables in ALLOWED_TABLES and only
   whitelisted functions (everything else, including sqlite_master and ATTACH, is denied).
3. One statement, starting with SELECT or WITH.
4. Row cap and wall-clock limit.

Errors come back as SqlToolError with a readable message, so the agent can fix its query.
"""
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB = ROOT / "data" / "processed" / "portfolios.db"

ALLOWED_TABLES = frozenset({"clients", "instruments", "holdings"})
ALLOWED_FUNCTIONS = frozenset({
    "count", "sum", "avg", "min", "max", "total", "round", "abs", "coalesce", "ifnull", "nullif",
    "lower", "upper", "length", "substr", "group_concat", "date", "strftime",
})
MAX_ROWS = 50
TIMEOUT_SECONDS = 2.0

SCHEMA_DESCRIPTION = """\
SQLite database with FICTIONAL client portfolios (all amounts in CHF).

clients(client_id, name, age, city, risk_profile, horizon_years)
  risk_profile is one of: 'Conservative', 'Balanced', 'Growth'.
instruments(instrument_id, name, isin, asset_class, region, kid_source)
  asset_class is one of: 'Equity', 'Bond', 'Cash', 'Commodity'.
  kid_source is the KID file for the instrument, or NULL when we have no KID for it.
holdings(client_id, instrument_id, units, price_chf, market_value_chf)
  One row per client and instrument. A client's portfolio value is the SUM of market_value_chf.

Join holdings.client_id = clients.client_id and holdings.instrument_id = instruments.instrument_id.
Only SELECT is allowed.
"""

_COMMENTS = re.compile(r"--[^\n]*|/\*.*?\*/", re.S)
_STARTS_OK = re.compile(r"^(select|with)\b", re.I)


class SqlToolError(Exception):
    """The query was rejected or failed; the message is meant for the agent."""


@dataclass
class QueryResult:
    sql: str
    columns: list[str]
    rows: list[tuple]
    truncated: bool = False

    def to_markdown(self) -> str:
        if not self.rows:
            return "(no rows)"
        lines = ["| " + " | ".join(self.columns) + " |", "|" + "---|" * len(self.columns)]
        lines += ["| " + " | ".join("" if v is None else str(v) for v in row) + " |" for row in self.rows]
        if self.truncated:
            lines.append(f"(only the first {len(self.rows)} rows are shown)")
        return "\n".join(lines)


def _authorizer(action, arg1, arg2, db_name, source):
    if action == sqlite3.SQLITE_SELECT:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_READ:
        return sqlite3.SQLITE_OK if (arg1 or "").lower() in ALLOWED_TABLES else sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION:
        return sqlite3.SQLITE_OK if (arg2 or "").lower() in ALLOWED_FUNCTIONS else sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_DENY


def run_query(sql: str, db_path: Path = DEFAULT_DB, max_rows: int = MAX_ROWS) -> QueryResult:
    if not _STARTS_OK.match(_COMMENTS.sub(" ", sql).strip()):
        raise SqlToolError("Only a single SELECT query is allowed.")
    db_path = Path(db_path)
    if not db_path.exists():
        raise SqlToolError("The portfolio database does not exist (run scripts/make_portfolios.py).")

    conn = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        conn.execute("PRAGMA query_only = ON")  # before the authorizer, which denies PRAGMA
        conn.set_authorizer(_authorizer)
        deadline = time.monotonic() + TIMEOUT_SECONDS
        conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
        cursor = conn.execute(sql)
        columns = [d[0] for d in cursor.description or []]
        rows = cursor.fetchmany(max_rows + 1)
    except sqlite3.Error as exc:
        message = str(exc)
        if "not authorized" in message:
            message = (f"{message}. Only SELECT on {sorted(ALLOWED_TABLES)} with basic "
                       f"functions is allowed.")
        elif "interrupted" in message:
            message = f"The query took longer than {TIMEOUT_SECONDS:.0f} s and was stopped."
        raise SqlToolError(message) from exc
    finally:
        conn.close()
    return QueryResult(sql=sql, columns=columns, rows=rows[:max_rows], truncated=len(rows) > max_rows)
