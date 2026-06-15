"""Tests for the matched commonsense LoRA arm (CPU, no network).

The training script reuses the LoReFT data path (``encode_examples`` / ``collate_left_padded``), so
the only new pure logic to pin is ``build_lora_config`` — that the YAML ``lora:`` block maps to a
causal-LM ``LoraConfig`` with the LLM-Adapters fields the ReFT paper benchmarks against. The label
identity with the LoReFT arm (the matched-data control) is exercised here via the shared collate.
"""

from __future__ import annotations

import torch

from scripts.attribution.train_lora_commonsense import build_lora_config
from scripts.attribution.train_loreft_commonsense import collate_left_padded


def test_build_lora_config_maps_yaml_block():
    lcfg = {"rank": 32, "alpha": 64, "dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]}
    cfg = build_lora_config(lcfg)
    assert cfg.r == 32
    assert cfg.lora_alpha == 64
    assert cfg.lora_dropout == 0.05
    assert set(cfg.target_modules) == set(lcfg["target_modules"])
    assert cfg.task_type == "CAUSAL_LM"
    assert cfg.bias == "none"


def test_collate_labels_mask_prompt_and_pad():
    """Response-only CE: prompt and left-pad positions are -100, response+eos tokens are kept.

    This is the same collate the LoReFT arm uses, so the supervised signal is byte-identical across
    the two methods — the matched-data control for the activation-similarity comparison.
    """
    # two items: ids = prompt(plen) + response; plen marks the prompt boundary
    batch = [{"ids": [1, 2, 3, 10, 11], "plen": 3},      # response tokens at positions 3,4
             {"ids": [1, 2, 20], "plen": 2}]              # response token at position 2
    ids, mask, _locs, labels = collate_left_padded(batch, pad_id=0, n_prefix=7, n_suffix=7,
                                                   device="cpu")
    width = ids.shape[1]
    assert width == 5
    # item 1 (no left pad): labels = [-100,-100,-100,10,11]
    assert labels[0].tolist() == [-100, -100, -100, 10, 11]
    # item 2 (left-padded by 2): pad + prompt are -100, response token 20 kept
    assert labels[1].tolist() == [-100, -100, -100, -100, 20]
    # attention mask zeros exactly the left pad of the shorter item
    assert mask[1].tolist() == [0, 0, 1, 1, 1]
    assert torch.is_tensor(labels)
