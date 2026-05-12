"""Domain specifications for the position-baseline staircase.

For every supported task we need to answer three structural questions:

  1. Which token positions in the prompt count as "earlier" positions
     (the N+P-analog — the baseline we are trying to beat).
  2. Which position(s) count as the "target probe position"
     (where we are claiming the model has done computation beyond
     what's recoverable from earlier).
  3. How is the classification label defined for this prompt.

A *DomainSpec* captures these three things together with the predicted
gap-sign so we can run pre-registration checks automatically after
experiments finish.

DESIGN PRINCIPLES
-----------------
* Per-position curves are the source of truth. Aggregated baselines
  ("name+params mean-pool", "earlier-max") are *derived* from the
  same per-position pass — there is no separate baseline pipeline.

* Target positions are specified by a *resolver function* on the
  tokenizer-aware prompt, not by raw integer indices. Different
  models tokenize the same prompt differently (Llama BPE vs Gemma
  SentencePiece vs GPT-2 BPE) — resolving against tokens is the
  only correct option.

* "Earlier" positions are all token indices strictly less than the
  earliest target position. This is what makes the staircase a
  *position*-based diagnostic rather than a content-based one.

PRE-REGISTRATION (matches the D6 matrix locked with the user):
  code             → predicted gap: NEGATIVE (workshop result)
  trivia           → predicted gap: NEGATIVE (negative control)
  qa_suggestive    → predicted gap: WEAK / NEAR-ZERO
  qa_neutral       → predicted gap: STRONG POSITIVE  ← within-task discriminator
  rhyme            → predicted gap: STRONG POSITIVE  (Maar's causal ground truth)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


# ──────────────────────────────────────────────────────────────────────
# Predicted-gap sign for the pre-registration matrix
# ──────────────────────────────────────────────────────────────────────
class PredictedGap(str, Enum):
    NEGATIVE = "negative"
    NEAR_ZERO = "near_zero"
    WEAK_POSITIVE = "weak_positive"
    STRONG_POSITIVE = "strong_positive"


# ──────────────────────────────────────────────────────────────────────
# Resolver: tokenizer-aware position finding
# ──────────────────────────────────────────────────────────────────────
@dataclass
class PositionResolver:
    """Resolves a logical position name to a token index for a given prompt.

    The same logical position ("question mark", "newline", "last word
    of first line") sits at different token indices across tokenizers
    and across prompt instances. This dataclass packages the rule.

    Attributes:
        name: Human-readable identifier ("question_mark", "newline",
            "last_word_of_first_line", "colon", "answer_marker", etc.)
        find: Callable(token_strings, token_ids, tokenizer) -> int | None
            Returns the token index, or None if the position cannot be
            located in this particular tokenized prompt.
        description: A one-line description for logging.
    """
    name: str
    find: Callable[..., Optional[int]]
    description: str = ""


# ──────────────────────────────────────────────────────────────────────
# Stable resolver implementations (used across domains)
# ──────────────────────────────────────────────────────────────────────
def _last_token_resolver() -> PositionResolver:
    """The final token of the prompt (excluding padding)."""
    def find(token_strings, token_ids, tokenizer):
        return len(token_ids) - 1
    return PositionResolver(
        name="last_token",
        find=find,
        description="Final non-padding token of the prompt.",
    )


def _find_nth_token_resolver(name: str, target_string: str, occurrence: int = -1) -> PositionResolver:
    """Find the n-th occurrence of a literal target string in token strings.

    occurrence: -1 for last (default), 0 for first, etc.
    Match is by *contains*: a token whose string contains the target.
    For unambiguous single-character markers (?, \\n, :) this is robust.
    """
    def find(token_strings, token_ids, tokenizer):
        hits = [i for i, t in enumerate(token_strings) if target_string in t]
        if not hits:
            return None
        return hits[occurrence] if occurrence != -1 else hits[-1]
    return PositionResolver(
        name=name,
        find=find,
        description=f"Occurrence #{occurrence} of a token containing '{target_string}'.",
    )


def _token_before_resolver(name: str, anchor_resolver: PositionResolver) -> PositionResolver:
    """The token immediately preceding the anchor (anchor_idx - 1)."""
    def find(token_strings, token_ids, tokenizer):
        idx = anchor_resolver.find(token_strings, token_ids, tokenizer)
        if idx is None or idx <= 0:
            return None
        return idx - 1
    return PositionResolver(
        name=name,
        find=find,
        description=f"Token immediately before {anchor_resolver.name}.",
    )


# Pre-canned resolvers for common positions across domains
RESOLVER_QUESTION_MARK = _find_nth_token_resolver(
    "question_mark", "?", occurrence=-1
)
RESOLVER_NEWLINE = _find_nth_token_resolver(
    "newline", "\n", occurrence=-1
)
RESOLVER_COLON = _find_nth_token_resolver(
    "colon", ":", occurrence=-1
)
RESOLVER_LAST_TOKEN = _last_token_resolver()
RESOLVER_LAST_WORD_BEFORE_Q = _token_before_resolver(
    "last_word_before_question_mark", RESOLVER_QUESTION_MARK
)
RESOLVER_LAST_WORD_BEFORE_NL = _token_before_resolver(
    "last_word_before_newline", RESOLVER_NEWLINE
)


# ──────────────────────────────────────────────────────────────────────
# Domain specification
# ──────────────────────────────────────────────────────────────────────
@dataclass
class DomainSpec:
    """Full specification for a probing-staircase domain.

    Attributes:
        domain_id: Stable identifier ("code", "rhyme", "qa_suggestive",
            "qa_neutral", "trivia") — used as a directory/key name.

        target_position_resolvers: Ordered list of candidate target
            positions. The staircase reports per-resolver results; the
            *headline* target position is the one yielding the highest
            probe accuracy (chosen post-hoc but disclosed).

        earlier_region: "all_before_target" means every token strictly
            before the earliest target index counts as an "earlier"
            candidate. Some domains (code) cap this at the signature
            region to match the workshop's N+P definition.

        n_classes: Number of classification labels.

        label_field: Which field of PlanningExample.metadata or
            .target_value carries the class label.

        predicted_gap: From the pre-registration matrix.

        notes: Free-form description for human readers / paper writing.
    """
    domain_id: str
    target_position_resolvers: list[PositionResolver]
    earlier_region: str  # "all_before_target" | "signature_only"
    n_classes: int
    label_field: str  # path like "target_value" or "metadata.return_type"
    predicted_gap: PredictedGap
    notes: str = ""
    extra: dict = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────
# The five locked domains
# ──────────────────────────────────────────────────────────────────────

# 1. CODE — workshop anchor. Predicted negative.
# Target positions: the colon, or the newline right after, or the last
# signature token. All three carry full signature context.
SPEC_CODE = DomainSpec(
    domain_id="code",
    target_position_resolvers=[
        RESOLVER_COLON,
        RESOLVER_NEWLINE,
        RESOLVER_LAST_TOKEN,
    ],
    earlier_region="signature_only",  # cap "earlier" at signature tokens (≤10)
    n_classes=5,  # int / str / bool / list / float
    label_field="target_value",
    predicted_gap=PredictedGap.NEGATIVE,
    notes=(
        "Workshop anchor. Signature N+P region carries return-type info; "
        "probe at target position is not expected to add information. "
        "Empirical: 0/66 layer-model pairs significant after FDR (workshop)."
    ),
)


# 2. RHYME — Maar's primary causally-validated planning domain.
# Target positions: Maar's exact steering positions — last word of
# first line, or newline after first line.
SPEC_RHYME = DomainSpec(
    domain_id="rhyme",
    target_position_resolvers=[
        RESOLVER_LAST_WORD_BEFORE_NL,  # Maar's "last word" steering position
        RESOLVER_NEWLINE,              # Maar's "newline" steering position
    ],
    earlier_region="all_before_target",
    n_classes=10,  # ing / air / ip / oat / ird / ee / ight / ake / ow / it
    label_field="metadata.rhyme_family",
    predicted_gap=PredictedGap.STRONG_POSITIVE,
    notes=(
        "Maar et al. (ICLR 2026) show steering at last_word/newline "
        "causally manipulates the generated rhyme. Forward planning "
        "representations are localized at these positions; earlier "
        "content tokens of the first line do not determine the rhyme "
        "family the model will commit to."
    ),
)


# 3. QA-SUGGESTIVE — content predicts answer. Predicted weak/near-zero.
SPEC_QA_SUGGESTIVE = DomainSpec(
    domain_id="qa_suggestive",
    target_position_resolvers=[
        RESOLVER_QUESTION_MARK,
        RESOLVER_LAST_WORD_BEFORE_Q,
        RESOLVER_NEWLINE,
        RESOLVER_LAST_TOKEN,
    ],
    earlier_region="all_before_target",
    n_classes=2,  # article: a / an   (binary, from the noun's first letter)
    label_field="metadata.article",
    predicted_gap=PredictedGap.NEAR_ZERO,
    notes=(
        "Suggestive questions: surface content predicts the answer "
        "(e.g., 'Who performs in films?' → actor). The probe-baseline "
        "gap should be weak because earlier positions already carry "
        "the determinative info."
    ),
    extra={
        "secondary_task": {  # also probe noun-class (30-way) for richer signal
            "n_classes": 30,
            "label_field": "metadata.noun",
        },
    },
)


# 4. QA-NEUTRAL — KEY EXPERIMENT. Same question text for both nouns in a
# pair → surface CANNOT determine the answer. Causally-validated planning case.
SPEC_QA_NEUTRAL = DomainSpec(
    domain_id="qa_neutral",
    target_position_resolvers=[
        RESOLVER_QUESTION_MARK,
        RESOLVER_LAST_WORD_BEFORE_Q,
        RESOLVER_NEWLINE,
        RESOLVER_LAST_TOKEN,
    ],
    earlier_region="all_before_target",
    n_classes=2,  # article: a / an
    label_field="metadata.article",
    predicted_gap=PredictedGap.STRONG_POSITIVE,
    notes=(
        "Neutral questions in Maar et al.'s setup share IDENTICAL "
        "text across pair members (e.g., 'What organ is essential "
        "for life?' appears under both 'eye' and 'heart'). The model "
        "must plan an answer; surface content cannot distinguish. "
        "This is the within-task discriminator — same model, same "
        "question text, but the diagnostic flips positive."
    ),
    extra={
        "secondary_task": {
            "n_classes": 30,
            "label_field": "metadata.noun",
        },
    },
)


# 5. TRIVIA — our constructed negative control. Predicted negative.
SPEC_TRIVIA = DomainSpec(
    domain_id="trivia",
    target_position_resolvers=[
        RESOLVER_COLON,
        RESOLVER_NEWLINE,
        RESOLVER_LAST_TOKEN,
    ],
    earlier_region="all_before_target",
    n_classes=5,  # capitals / elements / presidents / planets / continents
    label_field="metadata.category",
    predicted_gap=PredictedGap.NEGATIVE,
    notes=(
        "Constructed negative control. Question content fully "
        "determines answer category (e.g., 'The capital of France is' "
        "implies category=capitals). No planning required; probe at "
        "target position should match the question-content baseline."
    ),
)


# Public registry
DOMAINS: dict[str, DomainSpec] = {
    "code":           SPEC_CODE,
    "rhyme":          SPEC_RHYME,
    "qa_suggestive":  SPEC_QA_SUGGESTIVE,
    "qa_neutral":     SPEC_QA_NEUTRAL,
    "trivia":         SPEC_TRIVIA,
}


def get_domain(domain_id: str) -> DomainSpec:
    if domain_id not in DOMAINS:
        raise KeyError(
            f"Unknown domain '{domain_id}'. "
            f"Available: {sorted(DOMAINS.keys())}"
        )
    return DOMAINS[domain_id]


# ──────────────────────────────────────────────────────────────────────
# Helper: resolve target positions for one example
# ──────────────────────────────────────────────────────────────────────
def resolve_target_positions(
    spec: DomainSpec,
    token_strings: list[str],
    token_ids: list[int],
    tokenizer,
) -> dict[str, Optional[int]]:
    """Resolve every candidate target position for this domain on this example.

    Returns:
        Dict mapping resolver name to token index (or None if not found).
        Callers handle None by skipping that resolver for this example.
    """
    return {
        r.name: r.find(token_strings, token_ids, tokenizer)
        for r in spec.target_position_resolvers
    }


def get_earlier_positions(
    spec: DomainSpec,
    target_position: int,
    n_tokens: int,
    signature_end: Optional[int] = None,
) -> list[int]:
    """Return token indices that count as "earlier" candidate positions.

    For code: capped at signature_end (≤10 tokens) to match workshop.
    For all other domains: every position strictly before target_position.
    """
    if spec.earlier_region == "signature_only":
        cap = signature_end if signature_end is not None else min(10, n_tokens)
        return list(range(0, min(cap, target_position)))
    elif spec.earlier_region == "all_before_target":
        return list(range(0, target_position))
    else:
        raise ValueError(f"Unknown earlier_region '{spec.earlier_region}'")


__all__ = [
    "PredictedGap",
    "PositionResolver",
    "DomainSpec",
    "DOMAINS",
    "get_domain",
    "resolve_target_positions",
    "get_earlier_positions",
    # Pre-canned resolvers exposed for the loaders / runners
    "RESOLVER_QUESTION_MARK",
    "RESOLVER_NEWLINE",
    "RESOLVER_COLON",
    "RESOLVER_LAST_TOKEN",
    "RESOLVER_LAST_WORD_BEFORE_Q",
    "RESOLVER_LAST_WORD_BEFORE_NL",
    # Specs (for direct reference)
    "SPEC_CODE",
    "SPEC_RHYME",
    "SPEC_QA_SUGGESTIVE",
    "SPEC_QA_NEUTRAL",
    "SPEC_TRIVIA",
]
