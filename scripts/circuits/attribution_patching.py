import torch
import einops

from transformer_lens import (
    HookedTransformer,
    ActivationCache,
)

def get_logit_diff(logits, clean_answer_id, corrupted_answer_id):
    if len(logits.shape) == 3:
        # Get final logits only from batch size == 1
        logits = logits[-1, -1, :]
    correct_logit = logits[clean_answer_id]
    incorrect_logit = logits[corrupted_answer_id]
    return correct_logit - incorrect_logit

def logit_metric(logits, clean_baseline, corrupted_baseline, clean_answer, corrupted_answer):
    return (get_logit_diff(logits, clean_answer, corrupted_answer) - corrupted_baseline) / (
        clean_baseline - corrupted_baseline
    )

class AttributionPatching:
    def __init__(self, model_name, clean_prompts, clean_answers, corrupted_prompts, corrupted_answers):
        self.model = HookedTransformer.from_pretrained(model_name)
        self.model.set_use_attn_result(True)
        self.model.set_use_attn_in(True)
        self.model.set_use_hook_mlp_in(False)

        self.clean_tokens = self.model.to_tokens(clean_prompts, prepend_bos=True, padding_side='left')
        self.corrupted_tokens = self.model.to_tokens(corrupted_prompts, prepend_bos=True, padding_side='left')
        print("Clean string 0", self.model.to_string(self.clean_tokens[0]))
        print("Corrupted string 0", self.model.to_string(self.corrupted_tokens[0]))
        self.clean_answer_ids = torch.Tensor([self.model.to_single_token(a) for a in clean_answers]).to(dtype=int)
        self.corrupted_answer_ids = torch.Tensor([self.model.to_single_token(a) for a in corrupted_answers]).to(dtype=int)
        print("Clean answers", self.clean_answer_ids)
        print("Corrupted answers", self.corrupted_answer_ids)

    def get_cache_fwd_and_bwd(self, tokens, metric,
                              clean_baseline, corrupted_baseline,
                              clean_answer, corrupted_answer):
        filter_layers = \
            lambda name: "_input" not in name and "mlp_in" not in name

        self.model.reset_hooks()

        cache = {}
        def forward_cache_hook(act, hook):
            cache[hook.name] = act.detach()
        self.model.add_hook(filter_layers, forward_cache_hook, "fwd")

        grad_cache = {}
        def backward_cache_hook(act, hook):
            grad_cache[hook.name] = act.detach()
        self.model.add_hook(filter_layers, backward_cache_hook, "bwd")

        value = metric(self.model(tokens),
                       clean_baseline, corrupted_baseline,
                       clean_answer, corrupted_answer)
        value.backward()
        self.model.reset_hooks()
        return (
            value.item(),
            ActivationCache(cache, self.model),
            ActivationCache(grad_cache, self.model),
        )

    def patch_residual(self):
        # FIXME: Implement batched version. But it will be memory-costly.
        num_prompts = len(self.clean_tokens)
        clean_logits_top_3 = []
        corrupted_logits_top_3 = []
        clean_logit_diff_avg = 0
        corrupted_logit_diff_avg = 0
        for i in range(0, num_prompts):
            clean_logits, __ = self.model.run_with_cache(self.clean_tokens[i])
            clean_logits_top_3.append(torch.sort(clean_logits[-1, -1, :], descending=True).indices[0:3])
            corrupted_logits, __ = self.model.run_with_cache(self.corrupted_tokens[i])
            corrupted_logits_top_3.append(torch.sort(corrupted_logits[-1, -1, :], descending=True).indices[0:3])

            clean_logit_diff_avg += get_logit_diff(clean_logits, self.clean_answer_ids[i], self.corrupted_answer_ids[i]).item()
            corrupted_logit_diff_avg += get_logit_diff(corrupted_logits, self.clean_answer_ids[i], self.corrupted_answer_ids[i]).item()

        clean_logit_diff_avg /= num_prompts
        corrupted_logit_diff_avg /= num_prompts

        print(f"Clean logit TOP-3: {self.model.to_string(torch.stack(clean_logits_top_3))}")
        print()
        print(f"Corrupted logit TOP-3: {self.model.to_string(torch.stack(corrupted_logits_top_3))}")

        print(f"Clean logit diff: {clean_logit_diff_avg:.4f}")
        print(f"Corrupted logit diff: {corrupted_logit_diff_avg:.4f}")

        residual_attr_avg = 0
        for i in range(0, num_prompts):
            clean_value, clean_cache, clean_grad_cache = self.get_cache_fwd_and_bwd(
                self.clean_tokens[i], logit_metric,
                clean_logit_diff_avg, corrupted_logit_diff_avg,
                self.clean_answer_ids[i], self.corrupted_answer_ids[i]
            )
            corrupted_value, corrupted_cache, corrupted_grad_cache = self.get_cache_fwd_and_bwd(
                self.corrupted_tokens[i], logit_metric,
                clean_logit_diff_avg, corrupted_logit_diff_avg,
                self.clean_answer_ids[i], self.corrupted_answer_ids[i]
            )

            clean_residual, residual_labels = clean_cache.accumulated_resid(
                -1, incl_mid=True, return_labels=True
            )
            corrupted_residual = corrupted_cache.accumulated_resid(
                -1, incl_mid=True, return_labels=False
            )
            corrupted_grad_residual = corrupted_grad_cache.accumulated_resid(
                -1, incl_mid=True, return_labels=False
            )
            residual_attr_avg += einops.reduce(
                corrupted_grad_residual * (clean_residual - corrupted_residual),
                "component batch pos d_model -> component pos",
                "sum",
            )
        residual_attr_avg /= num_prompts

        return residual_attr_avg, residual_labels


    def patch_layer_out(self):
        # FIXME: Implement batched version. But it will be memory-costly.
        num_prompts = len(self.clean_tokens)
        clean_logits_top_3 = []
        corrupted_logits_top_3 = []
        clean_logit_diff_avg = 0
        corrupted_logit_diff_avg = 0
        for i in range(0, num_prompts):
            clean_logits, __ = self.model.run_with_cache(self.clean_tokens[i])
            clean_logits_top_3.append(torch.sort(clean_logits[-1, -1, :], descending=True).indices[0:3])
            corrupted_logits, __ = self.model.run_with_cache(self.corrupted_tokens[i])
            corrupted_logits_top_3.append(torch.sort(corrupted_logits[-1, -1, :], descending=True).indices[0:3])

            clean_logit_diff_avg += get_logit_diff(clean_logits, self.clean_answer_ids[i], self.corrupted_answer_ids[i]).item()
            corrupted_logit_diff_avg += get_logit_diff(corrupted_logits, self.clean_answer_ids[i], self.corrupted_answer_ids[i]).item()

        clean_logit_diff_avg /= num_prompts
        corrupted_logit_diff_avg /= num_prompts

        print(f"Clean logit TOP-3: {self.model.to_string(torch.stack(clean_logits_top_3))}")
        print()
        print(f"Corrupted logit TOP-3: {self.model.to_string(torch.stack(corrupted_logits_top_3))}")

        print(f"Clean logit diff: {clean_logit_diff_avg:.4f}")
        print(f"Corrupted logit diff: {corrupted_logit_diff_avg:.4f}")

        residual_attr_avg = 0
        for i in range(0, num_prompts):
            clean_value, clean_cache, clean_grad_cache = self.get_cache_fwd_and_bwd(
                self.clean_tokens[i], logit_metric,
                clean_logit_diff_avg, corrupted_logit_diff_avg,
                self.clean_answer_ids[i], self.corrupted_answer_ids[i]
            )
            corrupted_value, corrupted_cache, corrupted_grad_cache = self.get_cache_fwd_and_bwd(
                self.corrupted_tokens[i], logit_metric,
                clean_logit_diff_avg, corrupted_logit_diff_avg,
                self.clean_answer_ids[i], self.corrupted_answer_ids[i]
            )

            clean_layer_out, labels = clean_cache.decompose_resid(-1, return_labels=True)
            corrupted_layer_out = corrupted_cache.decompose_resid(-1, return_labels=False)
            corrupted_grad_layer_out = corrupted_grad_cache.decompose_resid(
                 -1, return_labels=False
            )
            layer_out_attr = einops.reduce(
                corrupted_grad_layer_out * (clean_layer_out - corrupted_layer_out),
                "component batch pos d_model -> component pos",
                "sum",
            )
        residual_attr_avg /= num_prompts

        return layer_out_attr, labels
