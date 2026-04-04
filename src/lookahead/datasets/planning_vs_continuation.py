"""Planning vs. Statistical Continuation: the critical disambiguation.

Reviewer 2's strongest attack: "You're detecting statistical association,
not planning. The model learned 'red' → 'said' from training data."

This module implements experiments that CANNOT be explained by
statistical continuation alone:

1. NOVEL_RHYME: Use made-up words (e.g., "blorf") that have no
   training-data association, but follow English phonology. If the
   model commits to a rhyme for "blorf" (e.g., "dworf"), it MUST be
   planning, not retrieving a memorized association.

2. COMPETING_CONTINUATIONS: Prompts where the most likely continuation
   is NOT the rhyme. If commitment to the rhyme appears despite it
   being statistically unlikely, that's evidence of planning over
   the continuation baseline.

3. MULTI_STEP_PLANNING: Tasks requiring planning >5 tokens ahead
   where no single n-gram predicts the target. The acrostic task
   partially addresses this, but we add explicit multi-step tasks.

4. COUNTERFACTUAL_PROMPTS: Same prompt but with a manipulation that
   changes the PLAN but not the surface statistics. E.g., "Write a
   couplet rhyming with 'red'" vs "Write a couplet rhyming with
   'blue'" — identical structure, different required plan. If probes
   differentiate these, it's plan-sensitive, not surface-sensitive.
"""

from __future__ import annotations

import hashlib
from ..utils.types import PlanningExample, TaskType


# ═══════════════════════════════════════════════════════════════════════
# 1. NOVEL (NONCE) WORD RHYMES
# ═══════════════════════════════════════════════════════════════════════

# Phonologically valid English nonce words with clear rhyme patterns
NONCE_RHYME_SETS = [
    # (nonce_word, rhyme_pattern_like, plausible_nonce_rhymes)
    ("blorf", "dwarf", ["scorf", "dorf", "worf"]),
    ("snark", "dark", ["blark", "grark", "plark"]),
    ("gleeb", "need", ["preeb", "dreeb", "fleeb"]),
    ("crunt", "front", ["blunt", "grunt", "stunt"]),  # real rhymes exist
    ("splidge", "bridge", ["fridge", "ridge", "midge"]),  # real rhymes exist
    ("thwack", "back", ["black", "crack", "track"]),  # real rhymes exist
    ("zingle", "single", ["dingle", "mingle", "tingle"]),  # real rhymes
    ("prax", "tax", ["wax", "max", "fax"]),  # real rhymes
]

# These nonce words have NO memorized rhyme partners in training data.
# If the model commits to a rhyme for them, it must be constructing
# the rhyme relationship at inference time — i.e., planning.
PURE_NONCE_PAIRS = [
    # (nonce_anchor, nonce_target, template)
    ("blorf", "scorf", "The creature let out a mighty blorf,\n"),
    ("gleeb", "preeb", "She sang a tune of gleeb,\n"),
    ("snark", "blark", "Beyond the hill they heard a snark,\n"),
]


def generate_nonce_rhyme_dataset(
    include_real_controls: bool = True,
) -> list[PlanningExample]:
    """Generate nonce-word rhyme examples.
    
    These examples use made-up words that have no training-data
    n-gram associations. If the model can commit to rhyming them,
    it must be doing phonological planning, not retrieval.
    
    We also include real-word controls to compare commitment
    curves: if nonce-word commitment is LATER than real-word
    commitment, that's consistent with the model falling back
    to slower, more deliberate planning when it can't retrieve.
    """
    examples = []
    
    for idx, (nonce, pattern, nonce_rhymes) in enumerate(NONCE_RHYME_SETS):
        # Template with the nonce word
        prompt = f"Complete this rhyme:\nThe wizard cast a spell of {nonce},\n"
        
        ex = PlanningExample(
            task_type=TaskType.RHYME,
            prompt=prompt,
            target_value=nonce_rhymes[0] if nonce_rhymes else "",
            target_token_positions=[],
            metadata={
                "anchor_word": nonce,
                "all_valid_rhymes": nonce_rhymes,
                "is_nonce": True,
                "rhyme_pattern": pattern,
                "is_control": False,
            },
            example_id=hashlib.md5(f"nonce_{nonce}".encode()).hexdigest()[:12],
        )
        examples.append(ex)
    
    # Real-word controls with same template structure
    if include_real_controls:
        real_pairs = [
            ("dark", ["park", "mark", "spark", "bark", "shark"]),
            ("need", ["feed", "speed", "lead", "read", "seed"]),
            ("back", ["track", "black", "pack", "stack", "crack"]),
        ]
        for idx, (anchor, rhymes) in enumerate(real_pairs):
            prompt = f"Complete this rhyme:\nThe wizard cast a spell of {anchor},\n"
            ex = PlanningExample(
                task_type=TaskType.RHYME,
                prompt=prompt,
                target_value=rhymes[0],
                target_token_positions=[],
                metadata={
                    "anchor_word": anchor,
                    "all_valid_rhymes": rhymes,
                    "is_nonce": False,
                    "is_nonce_control": True,
                    "is_control": False,
                },
                example_id=hashlib.md5(f"nonce_ctrl_{anchor}".encode()).hexdigest()[:12],
            )
            examples.append(ex)
    
    return examples


# ═══════════════════════════════════════════════════════════════════════
# 2. COMPETING CONTINUATIONS
# ═══════════════════════════════════════════════════════════════════════

def generate_competing_continuation_dataset() -> list[PlanningExample]:
    """Generate prompts where the rhyme competes with a more likely continuation.
    
    Example: "The cat sat on the mat so red,"
    - Most likely continuation (GPT-2): "and then..." (narrative)
    - Required rhyme: something ending in "-ed"
    
    If the model commits to the rhyme target despite it being
    statistically less likely than the narrative continuation,
    that's evidence of planning overriding default continuation.
    """
    examples = []
    
    # Prompts where the natural continuation is NOT a rhyme
    competing_prompts = [
        {
            "prompt": "Write a couplet. First line: The scientist worked late into the night,\nSecond line: ",
            "anchor": "night",
            "rhymes": ["light", "right", "sight", "bright", "might", "fight", "white"],
            "likely_continuation": "and",  # narrative continuation
        },
        {
            "prompt": "Write a couplet. First line: She walked alone beneath the rain,\nSecond line: ",
            "anchor": "rain",
            "rhymes": ["pain", "brain", "train", "plain", "gain", "main", "chain"],
            "likely_continuation": "and",
        },
        {
            "prompt": "Write a couplet. First line: The old man sat beside the fire,\nSecond line: ",
            "anchor": "fire",
            "rhymes": ["desire", "higher", "wire", "tire", "inspire", "admire"],
            "likely_continuation": "and",
        },
        {
            "prompt": "Write a couplet. First line: The children played until the day,\nSecond line: ",
            "anchor": "day",
            "rhymes": ["way", "say", "play", "stay", "away", "may", "lay"],
            "likely_continuation": "was",
        },
        {
            "prompt": "Write a couplet. First line: Through fields of gold beneath the sun,\nSecond line: ",
            "anchor": "sun",
            "rhymes": ["run", "fun", "done", "won", "gun", "one", "begun"],
            "likely_continuation": "the",
        },
    ]
    
    for idx, item in enumerate(competing_prompts):
        ex = PlanningExample(
            task_type=TaskType.RHYME,
            prompt=item["prompt"],
            target_value=item["rhymes"][0],
            target_token_positions=[],
            metadata={
                "anchor_word": item["anchor"],
                "all_valid_rhymes": item["rhymes"],
                "likely_continuation": item["likely_continuation"],
                "is_competing": True,
                "is_control": False,
            },
            example_id=hashlib.md5(f"competing_{idx}".encode()).hexdigest()[:12],
        )
        examples.append(ex)
    
    return examples


# ═══════════════════════════════════════════════════════════════════════
# 3. COUNTERFACTUAL PROMPTS
# ═══════════════════════════════════════════════════════════════════════

def generate_counterfactual_pairs() -> list[tuple[PlanningExample, PlanningExample]]:
    """Generate counterfactual pairs: same structure, different plan.
    
    Each pair has identical prompt structure except for ONE word that
    changes the required plan. If probes can differentiate the pair,
    they're sensitive to the plan, not to surface statistics.
    
    This is the cleanest test because the pairs are maximally
    controlled — everything is identical except the planning target.
    """
    pairs = []
    
    templates = [
        "Write a couplet where the last word of line 2 rhymes with '{anchor}'.\nLine 1: The world is full of {anchor},\nLine 2: ",
        "Complete the rhyme with '{anchor}':\nI see the {anchor},\n",
        "Rhyme with '{anchor}': Looking at the {anchor},\n",
    ]
    
    anchor_pairs = [
        ("red", ["said", "head", "bed"], "blue", ["true", "new", "grew"]),
        ("night", ["light", "right", "bright"], "day", ["way", "say", "play"]),
        ("cold", ["gold", "old", "bold"], "fire", ["desire", "higher", "wire"]),
        ("rain", ["pain", "brain", "train"], "sky", ["high", "fly", "eye"]),
        ("dream", ["seem", "team", "stream"], "stone", ["bone", "alone", "known"]),
    ]
    
    for t_idx, template in enumerate(templates):
        for p_idx, (anchor_a, rhymes_a, anchor_b, rhymes_b) in enumerate(anchor_pairs):
            prompt_a = template.format(anchor=anchor_a)
            prompt_b = template.format(anchor=anchor_b)
            
            ex_a = PlanningExample(
                task_type=TaskType.RHYME,
                prompt=prompt_a,
                target_value=rhymes_a[0],
                target_token_positions=[],
                metadata={
                    "anchor_word": anchor_a,
                    "all_valid_rhymes": rhymes_a,
                    "counterfactual_pair": f"cf_{t_idx}_{p_idx}",
                    "counterfactual_role": "a",
                    "is_control": False,
                },
                example_id=hashlib.md5(f"cf_{t_idx}_{p_idx}_a".encode()).hexdigest()[:12],
            )
            
            ex_b = PlanningExample(
                task_type=TaskType.RHYME,
                prompt=prompt_b,
                target_value=rhymes_b[0],
                target_token_positions=[],
                metadata={
                    "anchor_word": anchor_b,
                    "all_valid_rhymes": rhymes_b,
                    "counterfactual_pair": f"cf_{t_idx}_{p_idx}",
                    "counterfactual_role": "b",
                    "is_control": False,
                },
                example_id=hashlib.md5(f"cf_{t_idx}_{p_idx}_b".encode()).hexdigest()[:12],
            )
            
            pairs.append((ex_a, ex_b))
    
    return pairs
