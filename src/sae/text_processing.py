"""Sentence splitting, prompt section parsing, and choice parsing."""

import re

from src.formatting.configs import DefaultPromptFormat
from src.parsing import parse_choice

from .activations import CHOICE_SHORT_TERM, CHOICE_LONG_TERM, CHOICE_UNKNOWN, Sentence

MIN_SENTENCE_WORDS = 3


def get_prompt_markers() -> dict[str, str]:
    """Return prompt section markers from DefaultPromptFormat."""
    fmt = DefaultPromptFormat()
    return fmt.get_prompt_section_markers()


def get_response_markers() -> dict[str, str]:
    """Return response markers from DefaultPromptFormat."""
    fmt = DefaultPromptFormat()
    return fmt.get_response_markers()


# ── Sentence splitting ─────────────────────────────────────────────────────


def _split_raw(text: str, min_words: int = MIN_SENTENCE_WORDS) -> list[str]:
    """Split text into sentences, protecting decimals."""
    if not text or not text.strip():
        return []
    protected = re.sub(r"(\d)\.(\d)", r"\1<DECIMAL>\2", text)
    raw = re.split(r"(?<=[.!?;\n])\s+", protected)
    sentences = []
    for s in raw:
        s = s.replace("<DECIMAL>", ".").strip()
        if s and len(s.split()) >= min_words:
            sentences.append(s)
    return sentences


def _prompt_sections(prompt_text: str) -> list[tuple[str, str]]:
    """Split prompt into (section_name, text) pairs using format markers."""
    prompt_markers = get_prompt_markers()
    markers_sorted = sorted(
        prompt_markers.items(),
        key=lambda kv: prompt_text.find(kv[1]),
    )
    sections: list[tuple[str, str]] = []
    for i, (name, marker) in enumerate(markers_sorted):
        start = prompt_text.find(marker)
        if start < 0:
            continue
        if i + 1 < len(markers_sorted):
            next_start = prompt_text.find(markers_sorted[i + 1][1])
            if next_start > start:
                sections.append((name, prompt_text[start:next_start]))
                continue
        sections.append((name, prompt_text[start:]))
    return sections


def split_into_sentences(
    prompt_text: str,
    response_text: str,
    min_words: int = MIN_SENTENCE_WORDS,
) -> list[Sentence]:
    """Split prompt + response into classified Sentence objects.

    Uses DefaultPromptFormat markers to identify:
    - Prompt sections: situation, task, consider, action, format
    - Response sections: choice (around "I select:") and reasoning
      (around "My reasoning:")
    """
    response_markers = get_response_markers()
    choice_prefix = response_markers["choice_prefix"]
    reasoning_prefix = response_markers["reasoning_prefix"]

    sentences: list[Sentence] = []

    # ── Prompt sentences ──
    for section_name, section_text in _prompt_sections(prompt_text):
        for s in _split_raw(section_text, min_words):
            sentences.append(Sentence(text=s, source="prompt", section=section_name))

    # ── Response sentences ──
    if not response_text or not response_text.strip():
        return sentences

    # Strip chat-template artifacts from response
    clean_response = re.sub(r"<\|[^|]*\|>", "", response_text).strip()

    # Locate "I select:" and "My reasoning:" (last occurrences, matching
    # the convention in src.analysis.markers for response parsing).
    choice_pos = clean_response.lower().rfind(choice_prefix.lower())
    reasoning_pos = clean_response.lower().rfind(reasoning_prefix.lower())

    if choice_pos < 0 and reasoning_pos < 0:
        # No markers found — treat everything as reasoning
        for s in _split_raw(clean_response, min_words):
            sentences.append(Sentence(text=s, source="response", section="reasoning"))
        return sentences

    # Determine boundaries
    if choice_pos >= 0 and reasoning_pos >= 0 and reasoning_pos > choice_pos:
        choice_text = clean_response[choice_pos:reasoning_pos]
        reasoning_text = clean_response[reasoning_pos:]
    elif choice_pos >= 0:
        choice_text = clean_response[choice_pos:]
        reasoning_text = ""
    else:
        choice_text = ""
        reasoning_text = clean_response[reasoning_pos:]

    for s in _split_raw(choice_text, min_words):
        sentences.append(Sentence(text=s, source="response", section="choice"))
    for s in _split_raw(reasoning_text, min_words):
        sentences.append(Sentence(text=s, source="response", section="reasoning"))

    return sentences


# ── Choice parsing ─────────────────────────────────────────────────────────

_CHOICE_MAP = {
    "short_term": CHOICE_SHORT_TERM,
    "long_term": CHOICE_LONG_TERM,
    "unknown": CHOICE_UNKNOWN,
}


def parse_llm_choice(response_text: str, short_label: str, long_label: str) -> int:
    """Parse the LLM's choice from a response string."""
    response_markers = get_response_markers()
    choice_prefix = response_markers["choice_prefix"]
    result = parse_choice(response_text, short_label, long_label, choice_prefix)
    return _CHOICE_MAP[result]
