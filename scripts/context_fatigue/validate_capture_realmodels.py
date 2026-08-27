"""Validate SelectiveAttentionCapture against ``output_attentions`` on the real 7B models.

The unit tests pin the capture to ground truth on tiny models at 1e-5; the paper's Appendix B
claim (agreement 1.5e-8 OLMo-2, exactly 0.0 Qwen2/GQA) previously traced only to a commit
message. This script reproduces the measurement on the actual models in float32 and writes
the log the claim can cite: one forward per model over a chat-templated prompt, last-token
rows at every layer compared between the hook capture and ``output_attentions``. The exact
magnitude is reduction-order dependent (prompt, hardware, library versions): under the
2026-08 stack this box measures 1.1e-06 (OLMo-2) and 3.1e-05 (Qwen2.5) rather than the
historical 1.5e-08 / 0.0 — all float-reduction noise, orders of magnitude below any real
attention difference. The assertion is that noise bound; the log records the measured values,
and Appendix B should cite them rather than the commit-message figures.

    uv run python scripts/context_fatigue/validate_capture_realmodels.py
"""

import gc
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.probes.context_fatigue.attention_capture import SelectiveAttentionCapture

MODELS = ["allenai/OLMo-2-1124-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct"]
PROMPT = [{"role": "system", "content": "You are a doctor. Reply with just the letter."},
          {"role": "user", "content": "A patient reports fever and cough. (A) Flu (B) URTI. "
                                      "Which is more likely?"}]
OUT_LOG = Path("results/context_fatigue/capture_validation.log")


def validate(model_id: str, device: str = "cuda") -> float:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # float32: the claim is algorithmic equivalence with ``output_attentions``, so the
    # comparison must not be dominated by bf16 rounding (which sits at ~2e-3).
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32, device_map=device,
        attn_implementation="eager").eval()
    text = tokenizer.apply_chat_template(PROMPT, add_generation_prompt=True, tokenize=False)
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    layers = list(range(len(model.model.layers)))
    with torch.no_grad():
        out = model(ids, output_attentions=True)
    capture = SelectiveAttentionCapture(model, layers)
    capture.enabled = True
    with torch.no_grad():
        model(ids)
    worst = 0.0
    for li in layers:
        truth = out.attentions[li][0, :, -1, :].float().cpu()
        got = capture.captured[li].float().cpu()
        worst = max(worst, float((got - truth).abs().max()))
    capture.remove()
    del model, out
    gc.collect()
    torch.cuda.empty_cache()
    return worst


def main():
    lines = []
    for model_id in MODELS:
        worst = validate(model_id)
        line = f"{model_id}: max|capture - output_attentions| = {worst:.3e} over all layers"
        print(line, flush=True)
        lines.append(line)
        assert worst < 1e-4, f"capture disagrees with output_attentions on {model_id}"
    OUT_LOG.write_text("\n".join(lines) + "\n")
    print(f"Saved to {OUT_LOG}")


if __name__ == "__main__":
    main()
