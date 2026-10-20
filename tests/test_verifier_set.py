"""Model-free checks of the verifier stress-test data and its metrics function."""
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CASES = [json.loads(line) for line in (ROOT / "data" / "golden" / "verifier_set.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def test_dataset_is_balanced_and_well_formed():
    assert len({c["id"] for c in CASES}) == len(CASES)
    assert any(c["supported"] for c in CASES) and any(not c["supported"] for c in CASES)
    assert all(c["passages"] and c["answer"] and c["question"] for c in CASES)
    assert {c["kind"] for c in CASES if c["supported"]} == {"faithful"}


def test_faithful_answers_only_use_numbers_from_passages():
    """A label check: a 'faithful' answer must not contain a figure that is absent from its passages (derived ones are declared)."""
    for c in CASES:
        if not c["supported"]:
            continue
        allowed = _numbers(" ".join(c["passages"])) | set(c.get("derived", []))
        extra = _numbers(c["answer"]) - allowed
        assert not extra, f"{c['id']}: {extra}"


def test_each_corrupted_answer_declares_its_error_and_the_passages_lack_it():
    """The corrupted span must be in the answer and absent from the passages, otherwise the case is not a corruption."""
    for c in CASES:
        if c["supported"]:
            continue
        span = c["corrupted_span"]
        assert span in c["answer"], c["id"]
        assert span not in " ".join(c["passages"]), c["id"]


def test_summarise_counts_detection_and_false_alarms():
    spec = importlib.util.spec_from_file_location("eval_verifier", ROOT / "scripts" / "eval_verifier.py")
    sys.path.insert(0, str(ROOT / "src"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rows = {c["id"]: {"supported": c["supported"], "unsupported": ""} for c in CASES}   # a perfect verifier
    s = mod.summarise(CASES, rows)
    assert s["caught"] == s["corrupted"] and s["false_alarms"] == 0 and s["missed"] == []
    first_bad = next(c["id"] for c in CASES if not c["supported"])
    first_good = next(c["id"] for c in CASES if c["supported"])
    rows[first_bad]["supported"] = True        # one corruption slips through
    rows[first_good]["supported"] = False      # one false alarm
    s = mod.summarise(CASES, rows)
    assert s["caught"] == s["corrupted"] - 1 and s["false_alarms"] == 1
    assert s["missed"] == [first_bad] and s["alarm_ids"] == [first_good]
