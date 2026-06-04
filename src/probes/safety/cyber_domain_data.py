"""Real-data cybersecurity/code domain for the OOD-luck + refit probe (non-medical).

A non-medical domain with LOWER base refusal than the DDXPlus medical setup, built entirely from
real data (no synthesis):

  benign task : Alpaca code/tech instructions WITH their real outputs (instruction->output pairs)
                — used as ICL demos and as the map's fit inputs (the "coding assistant" task vector).
  in-domain harmful : AdvBench prompts filtered to cyber (hack a database, write malware, steal data)
                — real harmful requests in the same free-text instruction format, lower base refusal.

Both are plain user instructions, so the harmful prompts are format-matched to the benign fit task
(the distance-to-fit the conditional map keys on). The filtering is pure and unit-tested; the dataset
reads are thin.
"""

from __future__ import annotations

import re

from datasets import load_dataset

from src.probes.safety.safety_data import load_harmful

CYBER_HARM_RE = re.compile(
    r"\b(hack|malware|virus|exploit|ransomware|keylog|phishing|ddos|botnet|breach|spyware|"
    r"trojan|backdoor|database|inject|crack|password|firewall|server|steal)\b", re.I)
CODE_RE = re.compile(
    r"\b(code|program|function|algorithm|script|software|debug|python|java|sql|html|api|"
    r"compile|database|server|password|encrypt)\b", re.I)

# Deception / misinformation domain — direct real AdvBench requests with (hopefully) lower base
# refusal than malware, paired with benign Alpaca writing/content instructions.
DECEPTION_HARM_RE = re.compile(
    r"\b(fake|misinformation|mislead|misleading|propaganda|defam|manipulat|deceptive|deceive|"
    r"hoax|conspiracy|false|fraud|scam|impersonat|fabricat|rumor|disinformation|review|"
    r"blackmail|forge|extort)\b", re.I)
WRITING_RE = re.compile(
    r"\b(write|essay|article|story|review|letter|email|paragraph|blog|compose|persuasive|"
    r"describe|narrative|poem|speech|caption|headline|summar)\b", re.I)

DOMAINS = {"cyber": (CYBER_HARM_RE, CODE_RE), "deception": (DECEPTION_HARM_RE, WRITING_RE)}


def cyber_harmful_prompts(harmful_pool) -> list[str]:
    """AdvBench prompts that fall in the cyber/code domain."""
    return [p for p in harmful_pool if CYBER_HARM_RE.search(p)]


def code_benign_pairs(alpaca_rows) -> list[tuple[str, str]]:
    """Real (instruction, output) pairs from Alpaca that are code/tech and have no extra input."""
    out = []
    for r in alpaca_rows:
        instr = r["instruction"].strip()
        if r.get("input", "").strip() or not r.get("output", "").strip():
            continue
        if CODE_RE.search(instr):
            out.append((instr, r["output"].strip()))
    return out


def user_turn(text: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": text}]


def icl_freetext(tokenizer, demo_pairs, final_content: str, max_ctx: int, fill_target: float):
    """Pack as many (user instruction, assistant output) demos as fit, then the final user turn."""
    msgs, budget = [], int(max_ctx * fill_target)
    for instr, out in demo_pairs:
        trial = msgs + [{"role": "user", "content": instr}, {"role": "assistant", "content": out}]
        if len(tokenizer.apply_chat_template(trial, add_generation_prompt=False, tokenize=True)) > budget:
            break
        msgs = trial
    return msgs + [{"role": "user", "content": final_content}]


def filter_benign_pairs(alpaca_rows, benign_re) -> list[tuple[str, str]]:
    """Real (instruction, output) pairs matching a domain's benign topic regex."""
    out = []
    for r in alpaca_rows:
        instr = r["instruction"].strip()
        if r.get("input", "").strip() or not r.get("output", "").strip():
            continue
        if benign_re.search(instr):
            out.append((instr, r["output"].strip()))
    return out


def load_domain(name: str):
    """Real in-domain harmful prompts + real benign instruction/output pairs for ``name``."""
    harm_re, benign_re = DOMAINS[name]
    harmful = [p for p in load_harmful() if harm_re.search(p)]
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    return harmful, filter_benign_pairs(alpaca, benign_re)


def load_cyber_domain():
    """Real cyber-harmful prompts + real code instruction/output pairs."""
    return load_domain("cyber")
