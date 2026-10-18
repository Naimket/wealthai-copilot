"""Run one query through the read-only SQL tool: uv run python scripts/query_sql.py "SELECT ..."."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from wealthai.tools.sql import SCHEMA_DESCRIPTION, SqlToolError, run_query  # noqa: E402

if len(sys.argv) < 2:
    print(SCHEMA_DESCRIPTION)
    sys.exit(0)

try:
    print(run_query(" ".join(sys.argv[1:])).to_markdown())
except SqlToolError as exc:
    print(f"REJECTED: {exc}")
