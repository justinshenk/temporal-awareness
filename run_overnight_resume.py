import json, os, sys, gc
import torch
sys.path.insert(0, '.')

# Load the overnight script functions
exec(open('run_overnight.py').read().split("def main():")[0])

# Load Phase 1a results
all_results = json.load(open(f"{OUTDIR}/overnight_phase1a_domains.json"))
print("Loaded Phase 1a results")

from transformer_lens import HookedTransformer

# Resume Phase 1b: Attn vs MLP
print("\n--- Resuming: Phase 1b ---")
model = HookedTransformer.from_pretrained("EleutherAI/gpt-j-6b", device="cuda", dtype=torch.float16)
model.eval()

attn_mlp_domains = {k: v for k, v in DOMAINS.items() 
                    if k in ["chain_of_thought", "code", "poetry"]}
all_results["gptj_attn_mlp"] = run_attn_mlp(model, "EleutherAI/gpt-j-6b", attn_mlp_domains)
save_intermediate(all_results, "phase1b_attnmlp")

# Phase 1c: Cross-domain transfer
print("\n--- Phase 1c: Cross-domain transfer ---")
domain_seqs = {}
for domain_name, prompts in DOMAINS.items():
    if domain_name in ["chain_of_thought", "code", "free_prose", "poetry"]:
        seqs = []
        for prompt in prompts:
            tokens = model.to_tokens(prompt, prepend_bos=True)
            with torch.no_grad():
                gen = model.generate(tokens, max_new_tokens=N_GEN, temperature=0.0)
            seqs.append({"prompt_len": tokens.shape[1], "full_ids": gen[0].cpu().tolist()})
        domain_seqs[domain_name] = seqs

all_results["gptj_transfer"] = run_cross_domain_transfer(model, "EleutherAI/gpt-j-6b", domain_seqs)
save_intermediate(all_results, "phase1c_transfer")

del model; torch.cuda.empty_cache(); gc.collect()

# Phase 2: Qwen-7B
print("\n--- Phase 2a: Qwen-7B code staircase ---")
model = HookedTransformer.from_pretrained("Qwen/Qwen2.5-7B", device="cuda", dtype=torch.float16)
model.eval()

all_results["qwen7b_code"] = run_code_staircase(model, "Qwen/Qwen2.5-7B", CODE_SIGS)
save_intermediate(all_results, "phase2a_code")

print("\n--- Phase 2b: Qwen-7B domain study ---")
all_results["qwen7b_domains_50"] = run_domain_study(model, "Qwen/Qwen2.5-7B", DOMAINS)
save_intermediate(all_results, "phase2b_domains")

print("\n--- Phase 2c: Qwen-7B attn vs MLP ---")
all_results["qwen7b_attn_mlp"] = run_attn_mlp(model, "Qwen/Qwen2.5-7B",
                                                {"chain_of_thought": DOMAINS["chain_of_thought"]})
save_intermediate(all_results, "phase2c_attnmlp")

del model; torch.cuda.empty_cache(); gc.collect()

save_intermediate(all_results, "complete")
print("\nDONE — OVERNIGHT BULLETPROOFING COMPLETE")
