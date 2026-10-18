"""Tests for the synthetic portfolios and the read-only SQL tool (no LLM, no PDFs needed)."""
import sqlite3
from contextlib import closing

import pytest

from wealthai.portfolio.synthetic import Instrument, build_database
from wealthai.tools.sql import SqlToolError, run_query

N_CLIENTS = 20
FAKE_KID_FUND = Instrument("Test World ETF", "IE00TEST0001", "Equity", "Global", kid_source="test.pdf")


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    return build_database(tmp_path_factory.mktemp("db") / "p.db", [FAKE_KID_FUND], n_clients=N_CLIENTS, seed=1)


def q(db, sql, **kw):
    return run_query(sql, db_path=db, **kw)


# --- the tool answers real questions ---------------------------------------------------

def test_counts_clients(db):
    assert q(db, "SELECT COUNT(*) FROM clients").rows == [(N_CLIENTS,)]


def test_join_and_aggregate(db):
    r = q(db, """SELECT c.name, SUM(h.market_value_chf) AS aum
                 FROM clients c JOIN holdings h USING (client_id)
                 GROUP BY c.client_id ORDER BY aum DESC LIMIT 3""")
    assert r.columns == ["name", "aum"] and len(r.rows) == 3
    assert r.rows[0][1] >= r.rows[1][1] >= r.rows[2][1] > 0


def test_cte_is_allowed(db):
    r = q(db, "WITH t AS (SELECT client_id FROM clients) SELECT COUNT(*) FROM t")
    assert r.rows == [(N_CLIENTS,)]


def test_row_cap_sets_truncated(db):
    r = q(db, "SELECT * FROM holdings", max_rows=5)
    assert len(r.rows) == 5 and r.truncated


def test_markdown_output(db):
    md = q(db, "SELECT client_id, name FROM clients LIMIT 2").to_markdown()
    assert md.startswith("| client_id | name |") and md.count("\n") == 3


# --- the tool refuses anything dangerous ------------------------------------------------

@pytest.mark.parametrize("sql", [
    "DELETE FROM clients",
    "UPDATE clients SET age = 1",
    "INSERT INTO clients VALUES (99, 'x', 1, 'x', 'Growth', 1)",
    "DROP TABLE clients",
    "PRAGMA table_info(clients)",
    "ATTACH DATABASE 'x.db' AS x",
    "SELECT 1; DROP TABLE clients",
    "-- sneaky comment\nDELETE FROM clients",
])
def test_rejects_non_select(db, sql):
    with pytest.raises(SqlToolError):
        q(db, sql)


@pytest.mark.parametrize("sql", [
    "SELECT * FROM sqlite_master",
    "SELECT * FROM sqlite_schema",
    "SELECT * FROM does_not_exist",
    "SELECT load_extension('x')",
    "SELECT readfile('/etc/passwd')",
    "SELECT randomblob(1000000000)",
])
def test_rejects_forbidden_objects_and_functions(db, sql):
    with pytest.raises(SqlToolError):
        q(db, sql)


def test_stops_runaway_query(db):
    with pytest.raises(SqlToolError, match="took longer"):
        q(db, "SELECT COUNT(*) FROM holdings a, holdings b, holdings c, holdings d, holdings e, holdings f")


def test_missing_database_has_clear_message(tmp_path):
    with pytest.raises(SqlToolError, match="does not exist"):
        run_query("SELECT 1", db_path=tmp_path / "nope.db")


# --- the synthetic data is sane ---------------------------------------------------------

def test_every_client_has_positive_holdings(db):
    r = q(db, """SELECT c.client_id, COUNT(h.instrument_id), SUM(h.market_value_chf)
                 FROM clients c LEFT JOIN holdings h USING (client_id) GROUP BY c.client_id""")
    assert len(r.rows) == N_CLIENTS
    assert all(n >= 4 and total > 0 for _, n, total in r.rows)  # 4 asset classes, at least 1 each


def test_conservative_clients_hold_more_bonds_than_growth(db):
    def bond_share(profile):
        r = q(db, f"""SELECT SUM(CASE WHEN i.asset_class = 'Bond' THEN h.market_value_chf ELSE 0 END)
                             / SUM(h.market_value_chf)
                      FROM holdings h JOIN instruments i USING (instrument_id)
                      JOIN clients c USING (client_id) WHERE c.risk_profile = '{profile}'""")
        return r.rows[0][0]
    shares = {p: bond_share(p) for p in ("Conservative", "Growth")}
    if None not in shares.values():  # a profile can be absent with few clients
        assert shares["Conservative"] > shares["Growth"]


def test_kid_instrument_is_linked(db):
    r = q(db, "SELECT name, kid_source FROM instruments WHERE kid_source IS NOT NULL")
    assert r.rows == [("Test World ETF", "test.pdf")]


def test_same_seed_same_data(tmp_path):
    a = build_database(tmp_path / "a.db", [FAKE_KID_FUND], n_clients=5, seed=7)
    b = build_database(tmp_path / "b.db", [FAKE_KID_FUND], n_clients=5, seed=7)
    def dump(path):
        with closing(sqlite3.connect(path)) as conn:
            return list(conn.iterdump())

    assert dump(a) == dump(b)
