"""Model-agnostic chat-template boundaries.

The prompt body we care about is wrapped by a chat template. To label token
positions we need to know exactly where that wrapper starts and ends. Detecting
it by matching known special tokens is a blacklist, and a blacklist silently
mislabels every model it was not written for.

We ask the tokenizer instead. Templating a sentinel and splitting on it yields
the literal prefix and suffix that model puts around user content, for any model
with a chat template.

The suffix is also where the paper's turn-transition analysis lives. Its tokens
differ per family, so we expose ROLE ALIASES rather than raw indices:

    Qwen3   suffix tokens: <|im_end|>  \\n                  <|im_start|>  assistant
    Llama-3 suffix tokens: <|eot_id|>  <|start_header_id|>  assistant     <|end_header_id|>

`assistant` is relative position 3 on Qwen and 2 on Llama. Comparing
`chat_suffix:3` across the two therefore compares different token types, which
is why callers should ask for `role_assistant` and let this module resolve it.
"""

from __future__ import annotations

from dataclasses import dataclass

# Must survive `| trim` (Llama trims user content) and never appear in a prompt.
_SENTINEL = "ZQXJBODYZQXJ"

# Role aliases, resolved against the decoded suffix tokens.
ROLE_TURN_END = "turn_end"
ROLE_TURN_START = "turn_start"
ROLE_ASSISTANT = "role_assistant"
ROLE_ALIASES = (ROLE_TURN_END, ROLE_TURN_START, ROLE_ASSISTANT)


@dataclass(frozen=True)
class ChatBoundaries:
    """The literal strings a chat template wraps user content in."""

    prefix: str
    suffix: str

    @property
    def has_suffix(self) -> bool:
        return bool(self.suffix)


def get_chat_boundaries(tokenizer) -> ChatBoundaries:
    """Return the exact prefix/suffix this tokenizer's chat template applies.

    Raises ValueError when the tokenizer has no chat template, or when the
    template mangles the sentinel so the split would be ambiguous.
    """
    if getattr(tokenizer, "chat_template", None) is None:
        raise ValueError(
            f"{tokenizer.__class__.__name__} has no chat_template; "
            "position mapping cannot locate the turn boundary."
        )

    templated = tokenizer.apply_chat_template(
        [{"role": "user", "content": _SENTINEL}],
        tokenize=False,
        add_generation_prompt=True,
    )

    if templated.count(_SENTINEL) != 1:
        raise ValueError(
            f"chat template produced {templated.count(_SENTINEL)} copies of the "
            "sentinel; cannot split prefix from suffix unambiguously."
        )

    prefix, suffix = templated.split(_SENTINEL, 1)
    return ChatBoundaries(prefix=prefix, suffix=suffix)


def resolve_role_positions(
    suffix_tokens: list[str], special_tokens: set[str] | None = None
) -> dict[str, int]:
    """Map role aliases to indices within the decoded chat-suffix tokens.

    `special_tokens` should come from the tokenizer (`all_special_tokens` plus
    any added-vocab control tokens). Without it we fall back to a bracket
    heuristic, which only recognises the `<|...|>` convention and so misses
    families like Gemma that use `<start_of_turn>`.

    The role name is identified structurally: within a generation suffix it is
    the only plain-text token. That holds whether the family calls it
    "assistant" (Qwen, Llama) or "model" (Gemma), so nothing is hardcoded.

    Returns only the aliases actually found, so callers must handle absence
    rather than silently receive a wrong index.
    """
    stripped = [t.strip() for t in suffix_tokens]

    def is_special(tok: str) -> bool:
        if special_tokens is not None:
            return tok in special_tokens
        return tok.startswith("<") and tok.endswith(">")

    specials = [i for i, t in enumerate(stripped) if t and is_special(t)]
    plain = [i for i, t in enumerate(stripped) if t and not is_special(t)]

    roles: dict[str, int] = {}
    if plain:
        roles[ROLE_ASSISTANT] = plain[0]
    if specials:
        roles[ROLE_TURN_END] = specials[0]
        assistant_at = roles.get(ROLE_ASSISTANT)
        before = [i for i in specials if assistant_at is None or i < assistant_at]
        if before and before[-1] != roles[ROLE_TURN_END]:
            roles[ROLE_TURN_START] = before[-1]

    return roles


def suffix_special_tokens(tokenizer) -> set[str]:
    """Every token the tokenizer treats as special or control."""
    specials = {str(t) for t in (tokenizer.all_special_tokens or [])}
    added = getattr(tokenizer, "added_tokens_encoder", None) or {}
    specials.update(str(t) for t in added)
    return specials
