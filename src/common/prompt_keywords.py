# Standard prompt format keywords for position references
# These are used by token_positions.py for resolving position specs
# Keys are shorthand names; values are literal strings searched in tokenized prompts.
# Must match text actually present in the rendered prompt (see DefaultPromptFormat).
PROMPT_KEYWORDS = {
    "situation": "SITUATION:",
    "task": "TASK:",
    "consider": "CONSIDER:",
    "action": "ACTION:",
    "format": "FORMAT:",
    "choice_prefix": "I select:",
    "reasoning_prefix": "My reasoning:",
}

# Keywords that should match LAST occurrence (in response, not FORMAT instructions)
LAST_OCCURRENCE_KEYWORDS = {"choice_prefix", "reasoning_prefix"}
