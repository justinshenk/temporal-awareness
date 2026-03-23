import torch
import einops
import pandas as pd
from enum import Enum

from transformer_lens import (
    HookedTransformer,
    ActivationCache,
    patching
)

import gc

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
    class Technique(Enum):
        DENOISING = 0,
        NOISING = 1
        DENOISING_CUSTOM = 2
        NOISING_CUSTOM = 3,
        DENOISING_BOTH_LOGPROBS = 4,
        NOISING_BOTH_LOGPROBS = 5,
        DENOISING_BOTH_LOGPROBS_CUSTOM = 6,
        NOISING_BOTH_LOGPROBS_CUSTOM = 7

    class Metric(Enum):
        LOGIT_DIFF = 0,
        LOGIT = 1,
        LOGPROB = 2

    class Viz(Enum):
        UP_MEANS_HIGH_ATTRIBUTION = 0,
        READER_FRIENDLY = 1

    def __init__(self, model_name, clean_prompts, clean_answers, corrupted_prompts, corrupted_answers,
                 metric_type=Metric.LOGIT_DIFF,
                 technique_type=Technique.DENOISING,
                 viz_type=Viz.UP_MEANS_HIGH_ATTRIBUTION):
        super().__init__(model_name, clean_prompts, clean_answers, corrupted_prompts, corrupted_answers)
        self.caches_and_baselines_ready = False
        self.metric_type = metric_type
        self.technique_type = technique_type
        self.viz_type = viz_type

        # Define answer_token_indices needed for logit_metric function
        answer_token_indices = torch.tensor(
            [
                [self.clean_answer_ids[i], self.corrupted_answer_ids[i]]
                for i in range(len(self.clean_answer_ids))
            ],
            device=self.model.cfg.device,
        ).to(dtype=int)

        self.inner_metric = None
        if (self.technique_type == ActivationPatching.Technique.DENOISING_BOTH_LOGPROBS or
            self.technique_type == ActivationPatching.Technique.NOISING_BOTH_LOGPROBS or
            self.technique_type == ActivationPatching.Technique.DENOISING_BOTH_LOGPROBS_CUSTOM or
            self.technique_type == ActivationPatching.Technique.NOISING_BOTH_LOGPROBS_CUSTOM):
            def __inner_get_both_logprobs__(logits):
                if len(logits.shape) == 3:
                    # Get final logits only
                    logits = logits[:, -1, :]
                logits_logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
                correct_logprobs = logits_logprobs.gather(1, answer_token_indices[:, 0].unsqueeze(1))
                incorrect_logprobs = logits_logprobs.gather(1, answer_token_indices[:, 1].unsqueeze(1))
                return correct_logprobs.mean(), incorrect_logprobs.mean()
            self.inner_metric = __inner_get_both_logprobs__
        elif (self.metric_type == ActivationPatching.Metric.LOGIT_DIFF):
            # Implement batched version of logit_metric that uses defined variables:
            def __inner_get_logit_diff__(logits):
                if len(logits.shape) == 3:
                    # Get final logits only
                    logits = logits[:, -1, :]
                correct_logits = logits.gather(1, answer_token_indices[:, 0].unsqueeze(1))
                incorrect_logits = logits.gather(1, answer_token_indices[:, 1].unsqueeze(1))
                return (correct_logits - incorrect_logits).mean()
            self.inner_metric = __inner_get_logit_diff__
        elif (self.metric_type == ActivationPatching.Metric.LOGIT):
            def __inner_get_logit__(logits):
                if len(logits.shape) == 3:
                    # Get final logits only
                    logits = logits[:, -1, :]
                correct_logits = logits.gather(1, answer_token_indices[:, 0].unsqueeze(1))
                return correct_logits.mean()
            self.inner_metric = __inner_get_logit__
        elif (self.metric_type == ActivationPatching.Metric.LOGPROB):
            def __inner_get_logprob__(logits):
                if len(logits.shape) == 3:
                    # Get final logits only
                    logits = logits[:, -1, :]
                logits_logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
                correct_logprobs = logits_logprobs.gather(1, answer_token_indices[:, 0].unsqueeze(1))
                return correct_logprobs.mean()
            self.inner_metric = __inner_get_logprob__

    # TODO: Is removing gradients save for Activation Patching (not for Attribution Patching)?
    def __precalculate_caches_and_baselines__(self):
        if not self.baselines_ready:
            num_prompts = len(self.clean_tokens)
            self.clean_logits_top_3 = []
            self.corrupted_logits_top_3 = []
            batched_clean_logits = []
            batched_corrupted_logits = []

            # Try run without cache
            for i in range(0, num_prompts):
                clean_logits, clean_cache = self.model.run_with_cache(self.clean_tokens[i])
                del clean_cache
                gc.collect()
                self.clean_logits_top_3.append(torch.sort(clean_logits[-1, -1, :], descending=True).indices[0:3])
                batched_clean_logits.append(clean_logits)
            if (self.technique_type == ActivationPatching.Technique.DENOISING_BOTH_LOGPROBS or
                self.technique_type == ActivationPatching.Technique.NOISING_BOTH_LOGPROBS or
                self.technique_type == ActivationPatching.Technique.DENOISING_BOTH_LOGPROBS_CUSTOM or
                self.technique_type == ActivationPatching.Technique.NOISING_BOTH_LOGPROBS_CUSTOM):
                self.clean_q_clean_a_bsl, self.clean_q_corrupted_a_bsl = self.inner_metric(torch.cat(batched_clean_logits))
                self.clean_q_clean_a_bsl = self.clean_q_clean_a_bsl.item()
                self.clean_q_corrupted_a_bsl = self.clean_q_corrupted_a_bsl.item()
            else:
                self.clean_q_clean_a_bsl = self.inner_metric(torch.cat(batched_clean_logits)).item()
            del batched_clean_logits
            gc.collect()

            for i in range(0, num_prompts):
                corrupted_logits, corrupted_cache = self.model.run_with_cache(self.corrupted_tokens[i])
                del corrupted_cache
                gc.collect()
                self.corrupted_logits_top_3.append(torch.sort(corrupted_logits[-1, -1, :], descending=True).indices[0:3])
                batched_corrupted_logits.append(corrupted_logits)
            if (self.technique_type == ActivationPatching.Technique.DENOISING_BOTH_LOGPROBS or
                self.technique_type == ActivationPatching.Technique.NOISING_BOTH_LOGPROBS or 
                self.technique_type == ActivationPatching.Technique.DENOISING_BOTH_LOGPROBS_CUSTOM or
                self.technique_type == ActivationPatching.Technique.NOISING_BOTH_LOGPROBS_CUSTOM):
                self.corrupted_q_clean_a_bsl, self.corrupted_q_corrupted_a_bsl = self.inner_metric(torch.cat(batched_corrupted_logits))
                self.corrupted_q_clean_a_bsl = self.corrupted_q_clean_a_bsl.item()
                self.corrupted_q_corrupted_a_bsl = self.corrupted_q_corrupted_a_bsl.item()
            else:      
                self.corrupted_q_clean_a_bsl = self.inner_metric(torch.cat(batched_corrupted_logits)).item()
            del batched_corrupted_logits
            gc.collect()

            self.baselines_ready = True

        print(f"Clean logit TOP-3: {self.model.to_string(torch.stack(self.clean_logits_top_3))}")
        print()
        print(f"Corrupted logit TOP-3: {self.model.to_string(torch.stack(self.corrupted_logits_top_3))}")
        print()
        print()

        if (self.technique_type == ActivationPatching.Technique.DENOISING_BOTH_LOGPROBS or
            self.technique_type == ActivationPatching.Technique.NOISING_BOTH_LOGPROBS or
            self.technique_type == ActivationPatching.Technique.DENOISING_BOTH_LOGPROBS_CUSTOM or
            self.technique_type == ActivationPatching.Technique.NOISING_BOTH_LOGPROBS_CUSTOM):
            print(f"Clean(clean) baseline metric: {self.clean_q_clean_a_bsl:.4f}")
            print(f"Corrupted(clean) baseline metric: {self.corrupted_q_clean_a_bsl:.4f}")
            print(f"Clean(corrupted) baseline metric: {self.clean_q_corrupted_a_bsl:.4f}")
            print(f"Corrupted(corrupted) baseline metric: {self.corrupted_q_corrupted_a_bsl:.4f}")
        else:
            print(f"Clean baseline metric: {self.clean_q_clean_a_bsl:.4f}")
            print(f"Corrupted baseline metric: {self.corrupted_q_clean_a_bsl:.4f}")

        if not self.caches_and_baselines_ready:
            __, self.clean_cache = self.model.run_with_cache(self.clean_tokens)
            __, self.corrupted_cache = self.model.run_with_cache(self.corrupted_tokens)
            self.caches_and_baselines_ready = True

    def __patch__(self, layer_specific_algorithm):
        # Precalculate caches and baselines if not yet:
        self.__precalculate_caches_and_baselines__()
        assert(self.caches_and_baselines_ready)

        if (self.technique_type == self.Technique.DENOISING):
            # for batch..
            def __inner_logit_metric__(logits):
                return (self.inner_metric(logits) - self.corrupted_q_clean_a_bsl) / (
                    self.clean_q_clean_a_bsl - self.corrupted_q_clean_a_bsl
                )
            act_patch_result = layer_specific_algorithm(
                self.model, self.corrupted_tokens, self.clean_cache, __inner_logit_metric__)
        elif (self.technique_type == self.Technique.DENOISING_CUSTOM):
            # for batch..
            def __inner_logit_metric__(logits):
                return (self.inner_metric(logits) - self.corrupted_q_clean_a_bsl)

            act_patch_result = layer_specific_algorithm(
                self.model, self.corrupted_tokens, self.clean_cache, __inner_logit_metric__)
        elif (self.technique_type == self.Technique.NOISING):
            # For Noising: basically do the same, but:
            # run corrupted and cache it first and then patch the clean.
            # We need to inject corrupted patches into clean run.
            # Metric: how much clean answer is broken. The more it severed,
            #         the more layer was needed for it.
            if (self.viz_type == ActivationPatching.Viz.UP_MEANS_HIGH_ATTRIBUTION):
                def __inner_logit_metric__(logits):
                    return (self.clean_q_clean_a_bsl - self.inner_metric(logits)) / (
                        self.clean_q_clean_a_bsl - self.corrupted_q_clean_a_bsl
                    )
                act_patch_result = layer_specific_algorithm(
                    self.model, self.clean_tokens, self.corrupted_cache, __inner_logit_metric__)
            elif (self.viz_type == ActivationPatching.Viz.READER_FRIENDLY):
                # Version #2 with plots looking downwards if logprob decreases.
                if (self.metric_type == ActivationPatching.Metric.LOGIT_DIFF):
                    def __inner_logit_metric__(logits):
                        return (self.clean_q_clean_a_bsl - self.inner_metric(logits)) / (
                            self.clean_q_clean_a_bsl - self.corrupted_q_clean_a_bsl
                        )
                    act_patch_result = layer_specific_algorithm(
                        self.model, self.clean_tokens, self.corrupted_cache, __inner_logit_metric__)
                else:
                    def __inner_logit_metric__(logits):
                        return - (self.inner_metric(logits) - self.clean_q_clean_a_bsl) / (
                            self.corruped_q_clean_a_bsl - self.clean_q_clean_a_bsl
                        )
                    act_patch_result = layer_specific_algorithm(
                        self.model, self.clean_tokens, self.corrupted_cache, __inner_logit_metric__)
        elif (self.technique_type == self.Technique.NOISING_CUSTOM):
            # Clean prompt has a positive difference.
            # Corrupted prompt has a negative difference.
            if (self.viz_type == ActivationPatching.Viz.UP_MEANS_HIGH_ATTRIBUTION):
                def __inner_logit_metric__(logits):
                    return (self.clean_q_clean_a_bsl - self.inner_metric(logits))

                act_patch_result = layer_specific_algorithm(
                    self.model, self.clean_tokens, self.corrupted_cache, __inner_logit_metric__)
            elif (self.viz_type == ActivationPatching.Viz.READER_FRIENDLY):
                # Version #2 with plots looking downwards if logprob decreases.
                if (self.metric_type == ActivationPatching.Metric.LOGIT_DIFF):
                    def __inner_logit_metric__(logits):
                        return (self.clean_q_clean_a_bsl - self.inner_metric(logits))
                    act_patch_result = layer_specific_algorithm(
                        self.model, self.clean_tokens, self.corrupted_cache, __inner_logit_metric__)
                else:
                    def __inner_logit_metric__(logits):
                        return (self.inner_metric(logits) - self.clean_q_clean_a_bsl)
                    act_patch_result = layer_specific_algorithm(
                        self.model, self.clean_tokens, self.corrupted_cache, __inner_logit_metric__)
        elif (self.technique_type == ActivationPatching.Technique.DENOISING_BOTH_LOGPROBS):
            if (self.viz_type == ActivationPatching.Viz.UP_MEANS_HIGH_ATTRIBUTION):
                def __inner_logit_metric_for_clean__(logits):
                    return (self.inner_metric(logits)[0] - self.corrupted_q_clean_a_bsl) / (
                        self.clean_q_clean_a_bsl - self.corrupted_q_clean_a_bsl
                    )
                def __inner_logit_metric_for_corrupted__(logits):
                    return (self.inner_metric(logits)[1] - self.corrupted_q_corrupted_a_bsl) / (
                        self.clean_q_corrupted_a_bsl - self.corrupted_q_corrupted_a_bsl
                    )
                act_patch_result_clean_logprob = layer_specific_algorithm(
                    self.model, self.corrupted_tokens, self.clean_cache, __inner_logit_metric_for_clean__)
                act_patch_result_clean_logprob_df = pd.DataFrame(act_patch_result_clean_logprob.cpu(), columns=self.first_prompt_as_ticks)
                act_patch_result_corr_logprob = layer_specific_algorithm(
                    self.model, self.corrupted_tokens, self.clean_cache, __inner_logit_metric_for_corrupted__)
                act_patch_result_corr_logprob_df = pd.DataFrame(act_patch_result_corr_logprob.cpu(), columns=self.first_prompt_as_ticks)
                return act_patch_result_clean_logprob_df, act_patch_result_corr_logprob_df
            elif (self.viz_type == ActivationPatching.Viz.READER_FRIENDLY):
                # Version #2 to show downwards contribution if logprob decreases
                def __inner_logit_metric_for_clean__(logits):
                    return (self.inner_metric(logits)[0] - self.corrupted_q_clean_a_bsl) / (
                        self.clean_q_clean_a_bsl - self.corrupted_q_clean_a_bsl
                    )
                def __inner_logit_metric_for_corrupted__(logits):
                    return  - (self.inner_metric(logits)[1] - self.corrupted_q_corrupted_a_bsl) / (
                        self.clean_q_corrupted_a_bsl - self.corrupted_q_corrupted_a_bsl
                    )
                act_patch_result_clean_logprob = layer_specific_algorithm(
                    self.model, self.corrupted_tokens, self.clean_cache, __inner_logit_metric_for_clean__)
                act_patch_result_clean_logprob_df = pd.DataFrame(act_patch_result_clean_logprob.cpu(), columns=self.first_prompt_as_ticks)
                act_patch_result_corr_logprob = layer_specific_algorithm(
                    self.model, self.corrupted_tokens, self.clean_cache, __inner_logit_metric_for_corrupted__)
                act_patch_result_corr_logprob_df = pd.DataFrame(act_patch_result_corr_logprob.cpu(), columns=self.first_prompt_as_ticks)
                return act_patch_result_clean_logprob_df, act_patch_result_corr_logprob_df
        elif (self.technique_type == ActivationPatching.Technique.NOISING_BOTH_LOGPROBS):
            if (self.viz_type == ActivationPatching.Viz.UP_MEANS_HIGH_ATTRIBUTION):
                def __inner_logit_metric__(logits):
                    return (self.clean_q_clean_a_bsl - self.inner_metric(logits)[0]) / (
                        self.clean_q_clean_a_bsl - self.corrupted_q_clean_a_bsl
                    )
                def __anti_inner_logit_metric__(logits):
                    return (self.clean_q_corrupted_a_bsl - self.inner_metric(logits)[1]) / (
                        self.clean_q_corrupted_a_bsl - self.corrupted_q_corrupted_a_bsl
                    )
                act_patch_result_clean_logprob = layer_specific_algorithm(
                    self.model, self.corrupted_tokens, self.clean_cache, __inner_logit_metric_for_clean__)
                act_patch_result_clean_logprob_df = pd.DataFrame(act_patch_result_clean_logprob.cpu(), columns=self.first_prompt_as_ticks)
                act_patch_result_corr_logprob = layer_specific_algorithm(
                    self.model, self.corrupted_tokens, self.clean_cache, __inner_logit_metric_for_corrupted__)
                act_patch_result_corr_logprob_df = pd.DataFrame(act_patch_result_corr_logprob.cpu(), columns=self.first_prompt_as_ticks)
                return act_patch_result_clean_logprob_df, act_patch_result_corr_logprob_df
            elif (self.viz_type == ActivationPatching.Viz.READER_FRIENDLY):
                # Version #2 to show downwards contribution if logprob decreases:
                def __inner_logit_metric_for_clean__(logits):
                    return - (self.inner_metric(logits)[0] - self.clean_q_clean_a_bsl) / (
                        self.corrupted_q_clean_a_bsl - self.clean_q_clean_a_bsl
                    )
                def __inner_logit_metric_for_corrupted__(logits):
                    return (self.clean_q_corrupted_a_bsl - self.inner_metric(logits)[1]) / (
                        self.clean_q_corrupted_a_bsl - self.corrupted_q_corrupted_a_bsl
                    )
                act_patch_result_clean_logprob = layer_specific_algorithm(
                    self.model, self.clean_tokens, self.corrupted_cache, __inner_logit_metric_for_clean__)
                act_patch_result_clean_logprob_df = pd.DataFrame(act_patch_result_clean_logprob.cpu(), columns=self.first_prompt_as_ticks)
                act_patch_result_corr_logprob = layer_specific_algorithm(
                    self.model, self.clean_tokens, self.corrupted_cache, __inner_logit_metric_for_corrupted__)
                act_patch_result_corr_logprob_df = pd.DataFrame(act_patch_result_corr_logprob.cpu(), columns=self.first_prompt_as_ticks)
                return act_patch_result_clean_logprob_df, act_patch_result_corr_logprob_df
        elif (self.technique_type == ActivationPatching.Technique.DENOISING_BOTH_LOGPROBS_CUSTOM):
            assert(self.viz_type == ActivationPatching.Viz.READER_FRIENDLY)
            def __inner_logit_metric_for_clean__(logits):
                return (self.inner_metric(logits)[0] - self.corrupted_q_clean_a_bsl)

            def __inner_logit_metric_for_corrupted__(logits):
                return (self.inner_metric(logits)[1] - self.corrupted_q_corrupted_a_bsl)

            act_patch_result_clean_logprob = layer_specific_algorithm(
                self.model, self.corrupted_tokens, self.clean_cache, __inner_logit_metric_for_clean__)
            act_patch_result_clean_logprob_df = pd.DataFrame(act_patch_result_clean_logprob.cpu(), columns=self.first_prompt_as_ticks)
            act_patch_result_corr_logprob = layer_specific_algorithm(
                self.model, self.corrupted_tokens, self.clean_cache, __inner_logit_metric_for_corrupted__)
            act_patch_result_corr_logprob_df = pd.DataFrame(act_patch_result_corr_logprob.cpu(), columns=self.first_prompt_as_ticks)
            return act_patch_result_clean_logprob_df, act_patch_result_corr_logprob_df
        elif (self.technique_type == ActivationPatching.Technique.NOISING_BOTH_LOGPROBS_CUSTOM):
            assert(self.viz_type == ActivationPatching.Viz.READER_FRIENDLY)
            def __inner_logit_metric_for_clean__(logits):
                return (self.inner_metric(logits)[0] - self.clean_q_clean_a_bsl)

            def __inner_logit_metric_for_corrupted__(logits):
                return (self.inner_metric(logits)[1] - self.clean_q_corrupted_a_bsl)

            act_patch_result_clean_logprob = layer_specific_algorithm(
                self.model, self.clean_tokens, self.corrupted_cache, __inner_logit_metric_for_clean__)
            act_patch_result_clean_logprob_df = pd.DataFrame(act_patch_result_clean_logprob.cpu(), columns=self.first_prompt_as_ticks)
            act_patch_result_corr_logprob = layer_specific_algorithm(
                self.model, self.clean_tokens, self.corrupted_cache, __inner_logit_metric_for_corrupted__)
            act_patch_result_corr_logprob_df = pd.DataFrame(act_patch_result_corr_logprob.cpu(), columns=self.first_prompt_as_ticks)
            return act_patch_result_clean_logprob_df, act_patch_result_corr_logprob_df
        else:
            raise Exception("Unknown patching technique type is sent!")

        df = pd.DataFrame(act_patch_result.cpu(), columns=self.first_prompt_as_ticks)
        return df

    def patch_residual(self):
        return self.__patch__(patching.get_act_patch_resid_pre)

    def patch_layer_out(self):
        raise NotImplementedError()

    def patch_attn_out(self):
        return self.__patch__(patching.get_act_patch_attn_out)

    def patch_mlp_out(self):
        return self.__patch__(patching.get_act_patch_mlp_out)

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

    def patch_mlp_out(self):
        raise NotImplementedError()
