import torch
import einops
import pandas as pd

from transformer_lens import (
    HookedTransformer,
    ActivationCache,
    patching
)

def get_logit_diff(logits, clean_answer_id, corrupted_answer_id):
    if len(logits.shape) == 3:
        # Get final logits only from batch size == 1
        logits = logits[-1, -1, :]
    correct_logit = logits[clean_answer_id]
    incorrect_logit = logits[corrupted_answer_id]
    return (correct_logit - incorrect_logit)

def logit_metric(logits, clean_baseline, corrupted_baseline, clean_answer, corrupted_answer):
    return (get_logit_diff(logits, clean_answer, corrupted_answer) - corrupted_baseline) / (
        clean_baseline - corrupted_baseline
    )

class Patching:
    def __init__(self, model_name, clean_prompts, clean_answers, corrupted_prompts, corrupted_answers):
        self.model = HookedTransformer.from_pretrained(model_name)
        self.model.set_use_attn_result(True)
        self.is_qwen = False
        if "Qwen" in model_name:
            self.model.set_use_split_qkv_input(True)
            self.is_qwen = True
        else:
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

        clean_tokens_ticks = self.clean_tokens[0].cpu()
        self.first_prompt_as_ticks = [f"{i}, {self.model.to_single_str_token(int(t))}" for i, t in enumerate(clean_tokens_ticks)]

        self.baselines_ready = False

    def __precalculate_baselines__(self):
        if not self.baselines_ready:
            num_prompts = len(self.clean_tokens)
            self.clean_logits_top_3 = []
            self.corrupted_logits_top_3 = []
            self.clean_baseline = 0
            self.corrupted_baseline = 0
            for i in range(0, num_prompts):
                clean_logits, __ = self.model.run_with_cache(self.clean_tokens[i])
                self.clean_logits_top_3.append(torch.sort(clean_logits[-1, -1, :], descending=True).indices[0:3])
                corrupted_logits, __ = self.model.run_with_cache(self.corrupted_tokens[i])
                self.corrupted_logits_top_3.append(torch.sort(corrupted_logits[-1, -1, :], descending=True).indices[0:3])

                self.clean_baseline += get_logit_diff(clean_logits, self.clean_answer_ids[i], self.corrupted_answer_ids[i]).item()
                self.corrupted_baseline += get_logit_diff(corrupted_logits, self.clean_answer_ids[i], self.corrupted_answer_ids[i]).item()

            self.clean_baseline /= num_prompts
            self.corrupted_baseline /= num_prompts

            self.baselines_ready = True

        print(f"Clean logit TOP-3: {self.model.to_string(torch.stack(self.clean_logits_top_3))}")
        print()
        print(f"Corrupted logit TOP-3: {self.model.to_string(torch.stack(self.corrupted_logits_top_3))}")
        print()
        print()
        
        print(f"Clean logit diff: {self.clean_baseline:.4f}")
        print(f"Corrupted logit diff: {self.corrupted_baseline:.4f}")

    def append_ticks(self, patch_metrics, prompt_number, is_clean=True):
        assert(prompt_number < len(self.clean_tokens))
        if is_clean:
            ticks = self.clean_tokens[prompt_number].cpu()
        else:
            ticks = self.corrupted_tokens[prompt_number].cpu()
        prompt_as_ticks = [f"{i}, {self.model.to_single_str_token(int(t))}" for i, t in enumerate(ticks)]
        df = pd.DataFrame(patch_metrics.cpu(), columns=prompt_as_ticks)
        return df

class ActivationPatching(Patching):
    def __init__(self, model_name, clean_prompts, clean_answers, corrupted_prompts, corrupted_answers):
        super().__init__(model_name, clean_prompts, clean_answers, corrupted_prompts, corrupted_answers)
        self.caches_and_baselines_ready = False

    def __precalculate_caches_and_baselines__(self):
        super().__precalculate_baselines__()
        if not self.caches_and_baselines_ready:
            __, self.clean_cache = self.model.run_with_cache(self.clean_tokens)
            self.caches_and_baselines_ready = True

    def __patch__(self, layer_specific_algorithm):
        # Precalculate caches and baselines if not yet:
        self.__precalculate_caches_and_baselines__()
        assert(self.caches_and_baselines_ready)

        # Define answer_token_indices needed for logit_metric function
        answer_token_indices = torch.tensor(
            [
                [self.clean_answer_ids[i], self.corrupted_answer_ids[i]]
                for i in range(len(self.clean_answer_ids))
            ],
            device=self.model.cfg.device,
        ).to(dtype=int)

        # Implement batched version of logit_metric that uses defined variables:
        def __inner_get_logit_diff__(logits):
            if len(logits.shape) == 3:
                # Get final logits only
                logits = logits[:, -1, :]
            correct_logits = logits.gather(1, answer_token_indices[:, 0].unsqueeze(1))
            incorrect_logits = logits.gather(1, answer_token_indices[:, 1].unsqueeze(1))
            return (correct_logits - incorrect_logits).mean()

        def __inner_logit_metric__(logits):
            return (__inner_get_logit_diff__(logits) - self.corrupted_baseline) / (
                self.clean_baseline - self.corrupted_baseline
            )

        # for batch..
        every_block_act_patch_result = layer_specific_algorithm(
            self.model, self.corrupted_tokens, self.clean_cache, __inner_logit_metric__)

        df = pd.DataFrame(every_block_act_patch_result.cpu(), columns=self.first_prompt_as_ticks)
        return df

    def patch_residual(self):
        return self.__patch__(patching.get_act_patch_resid_pre)

    def patch_layer_out(self):
        raise NotImplementedError()

    def patch_attn_out(self):
        return self.__patch__(patching.get_act_patch_attn_out)

class AttributionPatching(Patching):
    def __init__(self, model_name, clean_prompts, clean_answers, corrupted_prompts, corrupted_answers):
        super().__init__(model_name, clean_prompts, clean_answers, corrupted_prompts, corrupted_answers)
        self.caches_and_baselines_ready = False

    def get_cache_fwd_and_bwd(self, tokens, metric,
                              clean_baseline, corrupted_baseline,
                              clean_answer, corrupted_answer):
        if self.is_qwen:
            filter_layers = \
                lambda name: "_input" not in name and "mlp_in" not in name and "attn_in" not in name
        else:
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

    def __precalculate_caches_and_baselines__(self):
        super().__precalculate_baselines__()
        if not self.caches_and_baselines_ready:
            num_prompts = len(self.clean_tokens)
            self.clean_caches = []
            self.corrupted_caches = []
            self.corrupted_grad_caches = []
            for i in range(0, num_prompts):
                clean_value, clean_cache, clean_grad_cache = self.get_cache_fwd_and_bwd(
                    self.clean_tokens[i], logit_metric,
                    self.clean_baseline, self.corrupted_baseline,
                    self.clean_answer_ids[i], self.corrupted_answer_ids[i]
                )

                self.clean_caches.append(clean_cache)

                corrupted_value, corrupted_cache, corrupted_grad_cache = self.get_cache_fwd_and_bwd(
                    self.corrupted_tokens[i], logit_metric,
                    self.clean_baseline, self.corrupted_baseline,
                    self.clean_answer_ids[i], self.corrupted_answer_ids[i]
                )

                self.corrupted_caches.append(corrupted_cache)
                self.corrupted_grad_caches.append(corrupted_grad_cache)

            self.caches_and_baselines_ready = True

    def __patch__(self, layer_specific_algorithm):
        # Precalculate caches and baselines if not yet:
        self.__precalculate_caches_and_baselines__()
        assert(self.caches_and_baselines_ready)

        num_prompts = len(self.clean_tokens)
        attr_avg = 0
        for i in range(0, num_prompts):
            attr, labels = layer_specific_algorithm(self.clean_caches[i],
                                                    self.corrupted_caches[i],
                                                    self.corrupted_grad_caches[i])
            attr_avg += attr
        attr_avg /= num_prompts

        df = pd.DataFrame(attr_avg.cpu(), columns=self.first_prompt_as_ticks)
        return df, labels

    def patch_residual(self):
        def residual_algorithm(clean_cache, corrupted_cache, corrupted_grad_cache):
            clean_residual, residual_labels = clean_cache.accumulated_resid(
                -1, incl_mid=True, return_labels=True
            )
            corrupted_residual = corrupted_cache.accumulated_resid(
                -1, incl_mid=True, return_labels=False
            )
            corrupted_grad_residual = corrupted_grad_cache.accumulated_resid(
                -1, incl_mid=True, return_labels=False
            )
            residual_attr = einops.reduce(
                corrupted_grad_residual * (clean_residual - corrupted_residual),
                "component batch pos d_model -> component pos",
                "sum",
            )
            return residual_attr, residual_labels
        return self.__patch__(residual_algorithm)

    def patch_layer_out(self):
        def layer_out_algorithm(clean_cache, corrupted_cache, corrupted_grad_cache):
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
            return layer_out_attr, labels

        return self.__patch__(layer_out_algorithm)

    def patch_attn_out(self):
        def attn_out_algorithm(clean_cache, corrupted_cache, corrupted_grad_cache):
            labels = [i for i in range(0, self.model.cfg.n_layers)]
            clean_atten_out = clean_cache.stack_activation("attn_out")
            corrupted_atten_out = corrupted_cache.stack_activation("attn_out")
            corrupted_grad_atten_out = corrupted_grad_cache.stack_activation("attn_out")
            head_out_attr = einops.reduce(
                corrupted_grad_atten_out * (clean_atten_out - corrupted_atten_out),
                "layer batch pos d_model -> layer pos",
                "sum",
            )
            return head_out_attr, labels
        return self.__patch__(attn_out_algorithm)
