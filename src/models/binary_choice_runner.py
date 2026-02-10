"""Binary choice runner for preference experiments.

Extends ModelRunner with specialized binary choice methods.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from math import prod
from typing import Any, Optional, Union

import torch

from ..common.types import SchemaClass
from .backends import ModelBackend
from .model_runner import ModelRunner


@dataclass
class BinaryChoice(SchemaClass):
    # Minimal
    choice_idx: int

    # Prob details
    prob_trajectory: tuple[list[float], list[float]] | None = None
    divergent_probs: tuple[float, float] | None = None
    label_probs: tuple[float, float] | None = None
    response_probs: tuple[float, float] | None = None
    perplexities: tuple[float, float] | None = None
    divergent_token_id_position: int | None = None


@dataclass
class BinaryChoiceWithData(BinaryChoice):
    # Full Detail
    labels: tuple[str, str] | None = None
    response_texts: tuple[str, str] | None = None
    response_token_ids: tuple[int, int] | None = None

    # Extra Data
    intervention: Union[Intervention, list[Intervention]] | None = None
    internals: tuple[dict] | None = None

    def without_data(self) -> BinaryChoice:
        return self.as_base()


class BinaryChoiceRunner(ModelRunner):
    """High-level runner for binary choice preference experiments.

    Inherits all functionality from ModelRunner and adds binary choice
    probability computation methods.
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        backend: ModelBackend = ModelBackend.TRANSFORMERLENS,
    ):
        """Initialize BinaryChoiceRunner.

        Args:
            model_name: HuggingFace model name
            device: Device to run on (default: auto-detect)
            dtype: Data type for model weights
            backend: Which backend to use for inference
        """
        super().__init__(model_name, device, dtype, backend)

    ############################
    ######### MAIN API #########
    ############################

    def choose(
        self,
        prompt: str,  # e.g. 'Task:...' := "Task:Make choice... Labels: '<a>', '<b>', Response_Format: 'I choose:[chosen_label]...' ..."
        choice_prefix: str,  # e.g. 'I choose:'
        labels: tuple[str, str],  # e.g. ['<a>', '(<b>']
        # Advanced Options
        with_cache: bool = False,
        intervention: Optional[Union[Intervention, list[Intervention]]] = None,
        names_filter: Optional[callable] = None,
        past_kv_cache: Any = None,
    ) -> BinaryChoice:
        label_a = labels[0]  # e.g. "<a>"
        label_b = labels[1]  # e.g. "<b>"

        response_text_a = choice_prefix + labels[0]  # e.g. 'I choose:<a>'
        response_text_b = choice_prefix + labels[0]  # e.g. 'I choose:<b>'

        #############################
        ######### ENCODING ##########
        #############################

        # OPTION 1
        formatted_prompt = self._apply_chat_template(prompt)
        response_token_ids_a = self.tokenizer.encode(
            formatted_prompt + response_text_a, add_special_tokens=False
        )
        response_token_ids_b = self.tokenizer.encode(
            formatted_prompt + response_text_b, add_special_tokens=False
        )

        option_1 = self.tokenizer.decode(response_token_ids_a)

        # OPTION 2
        prompt_token_ids = self.tokenizer.encode(
            self._apply_chat_template(prompt), add_special_tokens=True
        )
        response_token_ids_a = prompt_token_ids + self.tokenizer.encode(
            response_text_a, add_special_tokens=False
        )
        response_token_ids_b = prompt_token_ids + self.tokenizer.encode(
            response_text_b, add_special_tokens=False
        )

        option_2 = self.tokenizer.decode(response_token_ids_a)

        print("\n\n\n")
        print("ENCODING_OPTIONS")
        print("\n")
        print(option_1)
        print("\n")
        print(option_2)
        print("\n\n\n")

        # start_a, end_a = self.get_label_start_end_pos(response_token_ids_a, choice_prefix, label_a)
        # start_b, end_b = self.get_label_start_end_pos(response_token_ids_a, choice_prefix, label_b)

        div_pos = get_divergent_token_id_position(
            response_token_ids_a, response_token_ids_a
        )
        start_a, end_a = div_pos, div_pos + len(choice_prefix)
        start_b, end_b = div_pos, div_pos + len(choice_prefix)

        #############################
        ######### INFERENCE #########
        #############################

        internals_cache_a = None
        internals_cache_b = None

        if intervention and with_cache:
            prob_trajectory_a, internals_cache_a = (
                self.get_prob_trajectory_with_intervention_and_cache(
                    response_token_ids_a
                )
            )  # [ ..., p('a' | prompt + choice_prefix), ... ] , act_cache
            prob_trajectory_b, internals_cache_b = (
                self.get_prob_trajectory_with_intervention_and_cache(
                    response_token_ids_b
                )
            )  # [ ..., p('b' | prompt + choice_prefix), ... ] , act_cache
        elif intervention:
            prob_trajectory_a = self.get_prob_trajectory_with_intervention(
                response_token_ids_a
            )  # [ ..., p('a' | prompt + choice_prefix), ... ]
            prob_trajectory_b = self.get_prob_trajectory_with_intervention(
                response_token_ids_b
            )  # [ ..., p('b' | prompt + choice_prefix), ... ]
        elif with_cache:
            prob_trajectory_a, internals_cache_a = self.get_prob_trajectory_with_cache(
                response_token_ids_a
            )  # [ ..., p('a' | prompt + choice_prefix), ... ] , act_cache
            prob_trajectory_b, internals_cache_b = self.get_prob_trajectory_with_cache(
                response_token_ids_b
            )  # [ ..., p('b' | prompt + choice_prefix), ... ] , act_cache
        else:
            prob_trajectory_a = self.get_prob_trajectory(
                response_token_ids_a,
            )  # [ ..., p('a' | prompt + choice_prefix), ... ]
            prob_trajectory_b = self.get_prob_trajectory(
                response_token_ids_b,
            )  # [ ..., p('b' | prompt + choice_prefix), ... ]

        if internals_cache_a and internals_cache_b:
            internals = (internals_cache_a, internals_cache_b)
        else:
            internals = None

        #############################
        ######## PROB RESULTS #######
        #############################

        divergent_token_id_position = get_divergent_token_id_position(
            response_token_ids_a, response_token_ids_b
        )  # e.g. pos of 'x' in: (choice_prefix + '<x>') := 'I choose:<x>'

        divergent_prob_a = prob_trajectory_a[
            divergent_token_id_position
        ]  # p('a' | prompt + choice_prefix + "<")
        divergent_prob_b = prob_trajectory_b[
            divergent_token_id_position
        ]  # p('b' | prompt + choice_prefix + "<")

        label_prob_a = prod(
            prob_trajectory_a[start_a:end_a]
        )  # p('<a>' | prompt + choice_prefix)
        label_prob_b = prod(
            prob_trajectory_b[start_b:end_b]
        )  # p('<b>' | prompt + choice_prefix)

        response_prob_a = prod(prob_trajectory_a)  # p(choice_prefix + '<a>' | prompt)
        response_prob_b = prod(prob_trajectory_b)  # p(choice_prefix + '<b> | prompt)

        perplexity_a = perplexity(
            prob_trajectory_a
        )  # p(choice_prefix + '<a>' | prompt)
        perplexity_b = perplexity(prob_trajectory_b)  # p('I choose:<b>' | 'Task:...')

        #############################
        ########### CHOICE ##########
        #############################

        choice_idx = 0

        choice = BinaryChoiceWithData(
            # Minimal
            choice_idx=choice_idx,
            # Prob details
            prob_trajectory=(prob_trajectory_a, prob_trajectory_b),
            divergent_probs=(divergent_prob_a, divergent_prob_b),
            label_probs=(label_prob_a, label_prob_b),
            response_probs=(response_prob_a, response_prob_b),
            perplexities=(perplexity_a, perplexity_b),
            divergent_token_id_position=divergent_token_id_position,
            # Full Detail
            labels=labels,
            response_texts=(response_text_a, response_text_b),
            response_token_ids=(response_token_ids_a, response_token_ids_b),
            # Extra Data
            intervention=intervention,
            internals=None,
        )

        return choice

    ############################
    ######### FAST API #########
    ############################

    # TODO(me, person): Think about this much later

    ############################
    ######### UTILITIES ########
    ############################


############################
##### HELPER FUNCTIONS #####
############################


def perplexity(token_probs):
    if any(p == 0 for p in token_probs):
        return float("inf")
    log_probs = [math.log(p) for p in token_probs]
    return math.exp(-sum(log_probs) / len(log_probs))


def get_divergent_token_id_position(ids_a: list[int], ids_b: list[int]):
    divergent_token_id_position = 0
    for i in range(min(len(ids_a), len(ids_b))):
        if ids_a[i] != ids_b[i]:
            divergent_token_id_position = i
            break
    else:
        divergent_token_id_position = min(len(ids_a), len(ids_b))
    return divergent_token_id_position


def parse_choice_from_generated_response(
    response: str,
    short_label: str,
    long_label: str,
    choice_prefix: str,
) -> str:
    """
    Parse choice from model response.

    Looks for pattern: "<choice_prefix> <label>"
    Returns: "short_term", "long_term", or "unknown"
    """
    response_lower = response.lower().strip()
    prefix_lower = choice_prefix.lower()

    labels = [short_label, long_label]
    labels_stripped = [label.rstrip(".)") for label in labels]
    all_variants = set(label.lower() for label in labels + labels_stripped)
    labels_pattern = "|".join(
        re.escape(label) for label in sorted(all_variants, key=len, reverse=True)
    )

    pattern = rf"{re.escape(prefix_lower)}\s*({labels_pattern})"
    match = re.search(pattern, response_lower)

    if match:
        matched = match.group(1)
        if matched in (short_label.lower(), short_label.rstrip(".)").lower()):
            return 1
        elif matched in (long_label.lower(), long_label.rstrip(".)").lower()):
            return 0

    return -1
