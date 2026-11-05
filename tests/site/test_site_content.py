"""Content guardrails for the public site: its question, its claims and its language."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SITE = REPO_ROOT / "site"
PAGES = SITE / "src" / "pages"
COMPONENTS = SITE / "src" / "components"
LAYOUT = SITE / "src" / "layouts" / "BaseLayout.astro"
ASTRO_CONFIG = SITE / "astro.config.mjs"
LEDGER_PATH = SITE / "src" / "data" / "evidence_ledger.json"

NEW_PAGES = {
    "index.astro",
    "representation.mdx",
    "evidence.astro",
    "explore.astro",
    "related-threads.mdx",
    "contribute.astro",
}
REMOVED_PAGES = {"understand.mdx", "what-we-score.astro", "start-here.mdx"}
REDIRECTS = {
    "/understand": "/representation",
    "/what-we-score": "/evidence",
    "/start-here": "/contribute",
}
BANNED_PHRASES = [
    "0.935",
    "early warning",
    "early-warning",
    "before it shows up",
    "before anything goes wrong",
    "~4×",
    "sense of time",
    "thinking about time",
    "inner leaning",
    "longer game",
    "improves planning",
    "improved planning",
    "steering sits below",
    "steering layers sit below",
]
AGENCY_VERBS = (
    r"thinks?|thinking|wants?|knows?|decides?|weighs?|looks?|looking|plans?|planning|"
    r"believes?|prefers?|understands?|chooses?|choosing|cares?|intends?|tries|trying"
)
AGENCY_PATTERN = re.compile(
    rf"\b(?:the |a |an |this |each |every )?(?:language models?|models?|LLMs?|AIs?)\s+"
    rf"(?:\w+ly\s+|can\s+|may\s+|might\s+|could\s+|will\s+|would\s+|does\s+|do\s+|is\s+|are\s+)?"
    rf"(?:{AGENCY_VERBS})\b",
    re.IGNORECASE,
)
NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?(?:[–-]\d+(?:\.\d+)?)?\s?(?:%|×)")
MARKUP_ATTRIBUTE = re.compile(r"\b(?:class|className|style)=(?:\"[^\"]*\"|\{`[^`]*`\})")


def _read(path: Path) -> str:
    assert path.is_file(), f"missing expected file: {path}"
    return path.read_text(encoding="utf-8")


def _public_sources() -> dict[str, str]:
    sources = {path.name: _read(path) for path in PAGES.iterdir() if path.is_file()}
    sources.update({path.name: _read(path) for path in COMPONENTS.iterdir() if path.is_file()})
    sources[LAYOUT.name] = _read(LAYOUT)
    return sources


def _visible_copy(source: str) -> str:
    return re.sub(r"\s+", " ", MARKUP_ATTRIBUTE.sub("", source))


def test_new_pages_exist_and_old_pages_are_gone() -> None:
    names = {path.name for path in PAGES.iterdir()}
    assert NEW_PAGES <= names, f"missing pages: {sorted(NEW_PAGES - names)}"
    assert not REMOVED_PAGES & names, f"pages to remove: {sorted(REMOVED_PAGES & names)}"


def test_home_leads_with_the_central_question() -> None:
    source = _visible_copy(_read(PAGES / "index.astro"))
    heading = re.search(r"<h1[^>]*>(.*?)</h1>", source)
    assert heading, "home page needs an h1"
    assert "Can we measure the time horizon behind a language model's output?" in heading.group(1)
    assert "How is time represented in LLMs?" in source
    assert "How does a time horizon become an output?" in source


def test_home_links_both_papers() -> None:
    source = _read(PAGES / "index.astro")
    assert "sources.rios" in source
    assert "sources.mraz" in source


def test_home_keeps_probe_accuracies_out_of_its_headlines() -> None:
    source = _read(PAGES / "index.astro")
    for number in ("99.2", "92.5", "83.5"):
        assert number not in source, f"{number} belongs in the evidence section, not on the home page"


def test_probe_accuracy_in_the_argument_carries_the_caveat() -> None:
    assert "no surface baseline yet" in _read(PAGES / "representation.mdx")


def test_temporal_scope_uses_the_paper_definition() -> None:
    source = _visible_copy(_read(PAGES / "representation.mdx"))
    assert "bounded interval of time" in source
    assert "time horizon" in source.lower()


def test_layer_comparison_keeps_its_limits() -> None:
    source = _visible_copy(_read(COMPONENTS / "LayerAxis.astro"))
    assert "no uniquely optimal layer" in source


def test_planning_result_keeps_its_narrow_wording() -> None:
    assert "planning-related capability metric" in _read(PAGES / "representation.mdx")


@pytest.mark.parametrize("phrase", BANNED_PHRASES)
def test_banned_phrases_stay_off_the_site(phrase: str) -> None:
    for name, source in _public_sources().items():
        assert phrase.lower() not in source.lower(), f"{name} contains {phrase!r}"


def test_copy_does_not_give_models_agency() -> None:
    for name, source in _public_sources().items():
        match = AGENCY_PATTERN.search(_visible_copy(source))
        assert match is None, f"{name} gives a model agency: {match.group(0) if match else ''}"


def test_ledger_copy_does_not_give_models_agency() -> None:
    match = AGENCY_PATTERN.search(_read(LEDGER_PATH))
    assert match is None, f"ledger gives a model agency: {match.group(0) if match else ''}"


def test_every_number_on_the_site_is_in_the_ledger() -> None:
    ledger_text = json.dumps(json.loads(_read(LEDGER_PATH)), ensure_ascii=False)
    for name, source in _public_sources().items():
        for token in NUMBER_PATTERN.findall(_visible_copy(source)):
            assert token in ledger_text, f"{name} shows {token!r}, which is not in the ledger"


def test_old_urls_redirect_to_their_replacements() -> None:
    config = _read(ASTRO_CONFIG)
    for old, new in REDIRECTS.items():
        assert f'"{old}": `${{base}}{new}`' in config, f"{old} -> {new}"
    assert "site:" not in config, "a fixed site URL sends redirects to the wrong domain"


def test_navigation_lists_the_new_pages() -> None:
    layout = _read(LAYOUT)
    for slug in ("representation", "evidence", "explore", "related-threads", "contribute"):
        assert f"{{BASE}}{slug}" in layout or f"${{BASE}}{slug}" in layout, f"nav is missing {slug}"
    for slug in ("understand", "what-we-score", "start-here"):
        assert f"${{BASE}}{slug}" not in layout, f"nav still links {slug}"
