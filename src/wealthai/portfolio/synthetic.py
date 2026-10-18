"""Synthetic (fictional) client portfolios for the SQL tool.

Every client, amount and position is randomly generated from a fixed seed:
no real person, bank or account is involved. Reproducible: same seed, same data.

Instruments come from two places:
- the KIDs in the knowledge base (name and ISIN are read from the PDFs), so a
  question can join "what does client X hold" with "what does that KID say";
- a few fictional instruments with no KID (bonds, cash, gold, Swiss equity), so the
  agent also meets positions it has no document for.
"""
import random
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB = ROOT / "data" / "processed" / "portfolios.db"
RAW_DIR = ROOT / "data" / "raw"

SCHEMA = """
CREATE TABLE clients (
    client_id     INTEGER PRIMARY KEY,
    name          TEXT NOT NULL,
    age           INTEGER NOT NULL,
    city          TEXT NOT NULL,
    risk_profile  TEXT NOT NULL CHECK (risk_profile IN ('Conservative', 'Balanced', 'Growth')),
    horizon_years INTEGER NOT NULL
);
CREATE TABLE instruments (
    instrument_id INTEGER PRIMARY KEY,
    name          TEXT NOT NULL,
    isin          TEXT NOT NULL UNIQUE,
    asset_class   TEXT NOT NULL CHECK (asset_class IN ('Equity', 'Bond', 'Cash', 'Commodity')),
    region        TEXT NOT NULL,
    kid_source    TEXT
);
CREATE TABLE holdings (
    client_id        INTEGER NOT NULL REFERENCES clients (client_id),
    instrument_id    INTEGER NOT NULL REFERENCES instruments (instrument_id),
    units            REAL NOT NULL,
    price_chf        REAL NOT NULL,
    market_value_chf REAL NOT NULL,
    PRIMARY KEY (client_id, instrument_id)
);
"""


@dataclass(frozen=True)
class Instrument:
    name: str
    isin: str
    asset_class: str
    region: str
    kid_source: str | None = None


# Fictional instruments: the "SYN" ISINs are deliberately not valid ISINs.
EXTRA_INSTRUMENTS = [
    Instrument("Synthetic CHF Bond Fund", "SYN000000001", "Bond", "Switzerland"),
    Instrument("Synthetic Global Bond Fund", "SYN000000002", "Bond", "Global"),
    Instrument("Synthetic Swiss Equity Fund", "SYN000000003", "Equity", "Switzerland"),
    Instrument("Synthetic Gold Tracker", "SYN000000004", "Commodity", "Global"),
    Instrument("CHF Cash Account", "SYN000000005", "Cash", "Switzerland"),
]

# Share of the portfolio per asset class, by risk profile.
TARGETS = {
    "Conservative": {"Equity": 0.30, "Bond": 0.50, "Cash": 0.15, "Commodity": 0.05},
    "Balanced": {"Equity": 0.55, "Bond": 0.30, "Cash": 0.10, "Commodity": 0.05},
    "Growth": {"Equity": 0.80, "Bond": 0.10, "Cash": 0.05, "Commodity": 0.05},
}

FIRST_NAMES = [
    "Lucas", "Emma", "Noah", "Mia", "Louis", "Lea", "Elias", "Sofia", "Gabriel", "Clara",
    "Matteo", "Nora", "Jonas", "Lina", "Theo", "Alice", "Samuel", "Julia", "Adrien", "Eva",
    "Marco", "Anna", "David", "Sara", "Felix",
]
LAST_NAMES = [
    "Meier", "Dubois", "Rossi", "Keller", "Favre", "Bernasconi", "Weber", "Martin", "Brunner",
    "Girard", "Moser", "Perret", "Fischer", "Rochat", "Huber", "Chevalley", "Schmid", "Bonvin",
    "Zimmermann", "Cattin", "Baumann", "Lambert", "Steiner", "Pasquier", "Graf",
]
CITIES = ["Geneva", "Lausanne", "Zurich", "Basel", "Bern", "Lugano", "Nyon", "Zug", "Fribourg", "Neuchatel"]


def load_kid_instruments(raw_dir: Path = RAW_DIR) -> list[Instrument]:
    """One equity instrument per KID PDF, with the name and ISIN printed in the document."""
    from wealthai.ingestion.parse import parse_pdf  # lazy: needs PyMuPDF

    instruments = []
    for pdf in sorted(raw_dir.glob("*.pdf")):
        doc = parse_pdf(pdf)
        name = doc.product_name
        if "emerging" in name.lower():
            region = "Emerging Markets"
        elif "s&p 500" in name.lower():
            region = "United States"
        else:
            region = "Global"
        instruments.append(
            Instrument(name, doc.isin or f"UNKNOWN-{pdf.stem}", "Equity", region, kid_source=pdf.name)
        )
    return instruments


def _split(rng: random.Random, total: float, n: int) -> list[float]:
    """Random split of `total` into n positive parts (Dirichlet via gamma draws)."""
    draws = [rng.gammavariate(2.0, 1.0) for _ in range(n)]
    return [total * d / sum(draws) for d in draws]


def build_database(
    db_path: Path = DEFAULT_DB,
    instruments: list[Instrument] | None = None,
    n_clients: int = 25,
    seed: int = 42,
) -> Path:
    """Create (overwriting) the SQLite database and fill it with synthetic data."""
    instruments = (instruments or []) + EXTRA_INSTRUMENTS
    rng = random.Random(seed)

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)

    prices = {i.isin: 1.0 if i.asset_class == "Cash" else round(rng.uniform(20, 200), 2) for i in instruments}
    by_class: dict[str, list[tuple[int, Instrument]]] = {}
    for instrument_id, ins in enumerate(instruments, start=1):
        by_class.setdefault(ins.asset_class, []).append((instrument_id, ins))

    with closing(sqlite3.connect(db_path)) as conn:
        conn.executescript(SCHEMA)
        conn.executemany(
            "INSERT INTO instruments VALUES (?, ?, ?, ?, ?, ?)",
            [(i, ins.name, ins.isin, ins.asset_class, ins.region, ins.kid_source)
             for i, ins in enumerate(instruments, start=1)],
        )

        names = [(f, l) for f in FIRST_NAMES for l in LAST_NAMES]
        rng.shuffle(names)
        for client_id in range(1, n_clients + 1):
            first, last = names[client_id - 1]
            profile = rng.choice(list(TARGETS))
            age = rng.randint(28, 78) if profile != "Growth" else rng.randint(28, 60)
            horizon = rng.randint(3, 10) if profile == "Conservative" else rng.randint(8, 25)
            conn.execute(
                "INSERT INTO clients VALUES (?, ?, ?, ?, ?, ?)",
                (client_id, f"{first} {last}", age, rng.choice(CITIES), profile, horizon),
            )

            total = min(max(round(rng.lognormvariate(13.0, 0.8), -3), 150_000), 8_000_000)
            for asset_class, share in TARGETS[profile].items():
                value = total * share * rng.uniform(0.8, 1.2)
                candidates = by_class[asset_class]
                chosen = rng.sample(candidates, k=rng.randint(1, min(3, len(candidates))))
                for (instrument_id, ins), part in zip(chosen, _split(rng, value, len(chosen))):
                    price = prices[ins.isin]
                    units = round(part / price, 2)
                    conn.execute(
                        "INSERT INTO holdings VALUES (?, ?, ?, ?, ?)",
                        (client_id, instrument_id, units, price, round(units * price, 2)),
                    )
        conn.commit()
    return db_path
