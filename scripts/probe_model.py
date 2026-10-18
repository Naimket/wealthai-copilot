"""Prueba rápida de un modelo: texto plano + salida estructurada (Pydantic).

Uso:
    uv run python scripts/probe_model.py <id-del-modelo>

El id exacto se copia de AI Studio (botón "Get code" del modelo).
No toca .env: solo cambia el modelo durante esta prueba.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from pydantic import BaseModel  # noqa: E402

from wealthai import llm  # noqa: E402


class Check(BaseModel):
    capital: str
    is_in_europe: bool


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("Uso: uv run python scripts/probe_model.py <id-del-modelo>")
    llm.MODEL = sys.argv[1]
    print(f"Modelo: {llm.MODEL}\n")

    t0 = time.time()
    try:
        out = llm.generate("Responde solo con una palabra: ¿capital de Suiza?")
        print(f"[1] Texto plano   OK  ({time.time() - t0:.1f}s): {str(out).strip()[:80]}")
    except Exception as e:
        print(f"[1] Texto plano   FALLA: {str(e)[:300]}")
        return

    t0 = time.time()
    try:
        out = llm.generate(
            "¿Cuál es la capital de Suiza y está en Europa?", schema=Check
        )
        ok = isinstance(out, Check)
        print(
            f"[2] Estructurada  {'OK' if ok else 'FALLA (no devuelve Pydantic)'}"
            f"  ({time.time() - t0:.1f}s): {out}"
        )
    except Exception as e:
        print(f"[2] Estructurada  FALLA: {str(e)[:300]}")


if __name__ == "__main__":
    main()
