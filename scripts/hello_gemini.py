"""First structured call to Gemini: extract key facts from a (fictitious) KID excerpt."""
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors
from pydantic import BaseModel, Field

load_dotenv()


class FundFacts(BaseModel):
    fund_name: str
    risk_indicator: int = Field(description="Summary risk indicator (SRI), scale 1 to 7")
    ongoing_charges_pct: float | None = Field(
        default=None, description="Ongoing charges per year, in percent")
    holding_period_years: int = Field(description="Recommended holding period in years")


# Fictitious text, written only for this exercise.
KID_EXCERPT = """
Key Information Document - Example Global Equity UCITS ETF (Acc)
Objective: the fund tracks a broad index of developed-market equities.
Indicateur de risque : 4 sur 7. Période de détention recommandée : 5 ans.
"""

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def generate_with_retry(max_attempts: int = 4, **kwargs):
    """Retry on temporary server errors (5xx) with exponential backoff."""
    for attempt in range(1, max_attempts + 1):
        try:
            return client.models.generate_content(**kwargs)
        except errors.ServerError as exc:
            if attempt == max_attempts:
                raise
            wait = 2**attempt
            print(f"Server error {exc.code}; retry {attempt}/{max_attempts - 1} in {wait}s")
            time.sleep(wait)


t0 = time.perf_counter()
response = generate_with_retry(
    model=os.getenv("GEMINI_MODEL", "gemini-flash-latest"),
        contents=(
        "Extract the requested fields from this document. "
        "If a field is not stated in the document, return null. Never guess.\n\n"
        f"{KID_EXCERPT}"
    ),
    config={
        "response_mime_type": "application/json",
        "response_schema": FundFacts,
    },
)
print(f"API call took {time.perf_counter() - t0:.1f}s")

facts: FundFacts = response.parsed
print(facts)
print(facts.model_dump_json(indent=2))
print("Charges missing:", facts.ongoing_charges_pct is None)