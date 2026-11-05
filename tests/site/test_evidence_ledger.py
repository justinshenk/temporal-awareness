"""Contract tests for the evidence ledger that backs every number on the public site."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = REPO_ROOT / "site" / "src" / "data" / "evidence_ledger.json"
VERIFIED_RESULTS = REPO_ROOT / "results" / "verified"
PROBE_VALIDATION = REPO_ROOT / "results" / "probe_validation_results.json"

REQUIRED_CLAIM_FIELDS = {
    "id",
    "question",
    "claim",
    "value",
    "model",
    "dataset",
    "n",
    "method",
    "source",
    "locator",
    "status",
    "featured",
    "caveat",
}
STATUSES = {"paper", "project-report", "repository", "unverified"}


@pytest.fixture(scope="module")
def ledger() -> dict:
    return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def claims(ledger: dict) -> dict[str, dict]:
    return {claim["id"]: claim for claim in ledger["claims"]}


def _csv_accuracy(filename: str, layer: int) -> float:
    with (VERIFIED_RESULTS / filename).open() as handle:
        for row in csv.DictReader(handle):
            if int(row["layer"]) == layer:
                return float(row["cv_accuracy_mean"])
    raise AssertionError(f"layer {layer} missing from {filename}")


def test_every_claim_has_the_full_schema(ledger: dict) -> None:
    for claim in ledger["claims"]:
        missing = REQUIRED_CLAIM_FIELDS - claim.keys()
        assert not missing, f"{claim.get('id')} is missing {sorted(missing)}"
        for field in REQUIRED_CLAIM_FIELDS - {"featured", "n"}:
            assert claim[field], f"{claim['id']} has an empty {field}"
        assert claim["status"] in STATUSES, f"{claim['id']} has status {claim['status']}"


def test_claim_ids_are_unique(ledger: dict) -> None:
    ids = [claim["id"] for claim in ledger["claims"]]
    assert len(ids) == len(set(ids))


def test_every_claim_resolves_to_a_linked_source(ledger: dict) -> None:
    for claim in ledger["claims"]:
        source = ledger["sources"].get(claim["source"])
        assert source, f"{claim['id']} cites unknown source {claim['source']}"
        assert source["url"].startswith("https://"), f"{claim['source']} has no URL"


def test_every_claim_belongs_to_a_known_question(ledger: dict) -> None:
    question_ids = {question["id"] for question in ledger["questions"]}
    for claim in ledger["claims"]:
        assert claim["question"] in question_ids, f"{claim['id']} has unknown question"


def test_paper_citations_carry_the_agreed_venues(ledger: dict) -> None:
    rios = ledger["sources"]["rios"]
    assert "2606.05194" in rios["url"]
    assert "BlackboxNLP" in rios["venue"]
    assert "extended abstract" in rios["venue"]
    mraz = ledger["sources"]["mraz"]
    assert "2608.03892" in mraz["url"]
    assert "Mechanistic Interpretability Workshop" in mraz["venue"]
    assert "ICML" in mraz["venue"]


def test_lead_steering_entry_records_its_conditions(claims: dict[str, dict]) -> None:
    gap = claims["steering-gap-32b"]
    assert gap["value"] == "59.4 points"
    assert gap["model"] == "Qwen3-32B"
    assert gap["n"] == 100
    assert "L44" in gap["method"]
    assert "greedy" in gap["method"].lower()
    assert "5.6 points" in gap["caveat"], "the random-direction control must stay beside the gap"
    assert "prompting baseline" in gap["caveat"]


def test_small_steering_sample_is_recorded(claims: dict[str, dict]) -> None:
    odds = claims["steering-odds-4b"]
    assert odds["n"] == 20
    assert "suggestive" in odds["caveat"]


def test_probe_accuracies_carry_the_baseline_caveat(ledger: dict) -> None:
    probes = [claim for claim in ledger["claims"] if claim["id"].startswith("probe-")]
    assert probes, "expected probe entries in the ledger"
    for claim in probes:
        assert "no surface baseline yet" in claim["caveat"], f"{claim['id']} lacks the caveat"


def test_implicit_peak_describes_its_dataset(claims: dict[str, dict]) -> None:
    peak = claims["probe-peak-4b"]
    assert peak["value"] == "99.2%"
    assert "implicit" in peak["dataset"].lower()
    assert "L26" in peak["method"]
    assert peak["featured"] is True


def test_gpt2_training_score_is_retired_and_labelled(claims: dict[str, dict]) -> None:
    train = claims["probe-train-gpt2"]
    assert train["value"] == "92.5%"
    assert train["featured"] is False
    assert "training set" in train["method"]


def test_gpt2_entries_match_verified_results(claims: dict[str, dict]) -> None:
    assert _csv_accuracy("full_train_results.csv", 8) == pytest.approx(0.925)
    assert claims["probe-train-gpt2"]["n"] == 400
    assert _csv_accuracy("full_test_results.csv", 6) == pytest.approx(0.84)
    assert claims["probe-within-test-gpt2"]["value"] == "84%"
    assert _csv_accuracy("control_ablated_results.csv", 10) == pytest.approx(1.0)
    assert claims["ablation-gpt2"]["status"] == "unverified"
    assert claims["ablation-gpt2"]["featured"] is False


def test_gpt2_transfer_matches_verified_results(claims: dict[str, dict]) -> None:
    results = json.loads(PROBE_VALIDATION.read_text(encoding="utf-8"))
    layer_ten = next(row for row in results["implicit_results"] if row["layer"] == 10)
    assert layer_ten["implicit_acc"] == pytest.approx(0.84)
    assert "L10" in claims["probe-transfer-gpt2"]["method"]


def test_withdrawn_correlation_is_absent(ledger: dict) -> None:
    assert "0.935" not in json.dumps(ledger)
