"""Thin Gemini wrapper: loads .env, structured output, retries on 429/5xx."""
import os
import time
from pathlib import Path

from google import genai
from google.genai import errors, types

ROOT = Path(__file__).resolve().parents[2]


def _load_env() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env()
MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")
_client: genai.Client | None = None
MIN_INTERVAL = float(os.environ.get("GEMINI_MIN_INTERVAL", "4.2"))  # free tier flash-lite: 15 requests/minute
_last_call = 0.0


def _throttle() -> None:
    """Keep at least MIN_INTERVAL seconds between calls so the free-tier RPM limit is never hit."""
    global _last_call
    wait = _last_call + MIN_INTERVAL - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last_call = time.monotonic()


def client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


def generate(prompt: str, schema=None, max_attempts: int = 6):
    """Call Gemini. With a Pydantic `schema` returns the parsed object, else plain text.

    429 (rate limit) and 5xx (server busy) are transient -> exponential backoff.
    Other 4xx errors are our fault -> raised immediately.
    """
    config = types.GenerateContentConfig(
        temperature=0,
        response_mime_type="application/json" if schema else None,
        response_schema=schema,
    )
    for attempt in range(max_attempts):
        try:
            _throttle()
            response = client().models.generate_content(model=MODEL, contents=prompt, config=config)
            return response.parsed if schema else response.text
        except (errors.ServerError, errors.ClientError) as exc:
            if "PerDay" in str(exc):  # daily quota: waiting minutes will not help
                raise
            transient = isinstance(exc, errors.ServerError) or getattr(exc, "code", None) == 429
            if not transient or attempt == max_attempts - 1:
                raise
            wait = min(60, 5 * 2**attempt)
            print(f"  [retry {attempt + 1}] {getattr(exc, 'code', '?')} -> waiting {wait}s")
            time.sleep(wait)
