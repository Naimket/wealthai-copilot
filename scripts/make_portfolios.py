"""Create the synthetic portfolio database: uv run python scripts/make_portfolios.py [n_clients] [seed]."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from wealthai.portfolio.synthetic import DEFAULT_DB, build_database, load_kid_instruments  # noqa: E402
from wealthai.tools.sql import run_query  # noqa: E402

n_clients = int(sys.argv[1]) if len(sys.argv) > 1 else 25
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42

path = build_database(DEFAULT_DB, load_kid_instruments(), n_clients=n_clients, seed=seed)
print(f"Database written to {path}\n")

print("Instruments:")
print(run_query("SELECT instrument_id, name, isin, asset_class, kid_source FROM instruments").to_markdown())
print("\nClients and portfolio value (top 5):")
print(run_query("""SELECT c.client_id, c.name, c.risk_profile, ROUND(SUM(h.market_value_chf)) AS aum_chf
                   FROM clients c JOIN holdings h USING (client_id)
                   GROUP BY c.client_id ORDER BY aum_chf DESC LIMIT 5""").to_markdown())
