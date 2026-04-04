"""Rhyme completion dataset for planning detection.

Generates couplets where the model must commit to a rhyme word before
producing it. The key property: when the model sees line 1 end with
a word (e.g., "red"), it must plan the rhyme target (e.g., "said", "head")
*before* generating the tokens that lead to it in line 2.

Ground truth: We construct couplets where the rhyme word is known,
so we can probe for its presence in activations at each position.

Controls:
- Minimal pairs: same line 1, different line 2 endings
- Non-rhyming pairs: same structure, no rhyme constraint
- Multiple valid rhymes: cases where several rhyme targets are plausible
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Optional

from ..utils.types import PlanningExample, TaskType


# Curated rhyme sets with frequency tiers
# Each entry: (line1_end_word, [common_rhymes], [uncommon_rhymes])
# We separate common/uncommon to test whether models commit to
# high-frequency rhymes earlier than low-frequency ones.
RHYME_SETS = [
    ("red", ["said", "head", "bed", "dead", "fed", "led", "read", "spread", "bread", "thread"],
            ["stead", "shed", "sled", "wed", "dread", "fled", "shred"]),
    ("blue", ["true", "new", "few", "knew", "through", "too", "do", "grew", "drew", "flew"],
             ["hue", "clue", "due", "brew", "crew", "dew", "stew"]),
    ("day", ["way", "say", "play", "stay", "away", "may", "pay", "lay", "ray", "gray"],
            ["bay", "clay", "decay", "delay", "display", "essay", "okay", "pray", "spray", "stray"]),
    ("night", ["light", "right", "sight", "bright", "fight", "might", "white", "flight", "tight", "write"],
              ["bite", "height", "kite", "knight", "quite", "slight", "spite"]),
    ("love", ["above", "dove", "of", "shove", "glove"],
             ["thereof"]),
    ("heart", ["start", "part", "art", "smart", "apart", "dart", "chart", "cart"],
              ["depart", "impart", "restart"]),
    ("sky", ["high", "fly", "eye", "by", "why", "try", "die", "lie", "cry", "dry"],
            ["buy", "deny", "guy", "nearby", "reply", "shy", "supply", "tie"]),
    ("time", ["rhyme", "climb", "crime", "dime", "mime", "prime", "chime", "lime"],
             ["paradigm", "sublime", "thyme"]),
    ("tree", ["free", "see", "be", "me", "three", "agree", "key", "sea", "knee", "flee"],
             ["decree", "degree", "guarantee", "referee"]),
    ("fire", ["desire", "higher", "wire", "tire", "inspire", "admire", "entire", "hire"],
             ["acquire", "conspire", "expire", "perspire", "require"]),
    ("cold", ["old", "gold", "told", "hold", "bold", "fold", "sold", "rolled"],
             ["behold", "controlled", "enrolled", "unfold", "withhold"]),
    ("rain", ["pain", "main", "brain", "gain", "train", "plain", "chain", "remain", "explain", "contain"],
             ["abstain", "campaign", "complain", "domain", "obtain", "sustain"]),
    ("moon", ["soon", "tune", "June", "noon", "spoon", "boon", "balloon", "cartoon"],
             ["bassoon", "cocoon", "lagoon", "maroon", "platoon", "saloon"]),
    ("stone", ["bone", "alone", "known", "own", "phone", "tone", "grown", "zone", "throne", "blown"],
              ["cologne", "cyclone", "hormone", "ozone", "postpone"]),
    ("dream", ["seem", "team", "stream", "scheme", "cream", "beam", "gleam", "theme", "extreme"],
              ["esteem", "redeem", "regime", "supreme"]),
]


# Couplet templates — line 1 sets up the rhyme, line 2 must complete it
# {word} = line1 end word, model generates line 2
COUPLET_TEMPLATES = [
    # Simple — model just needs to complete the rhyme
    "Roses are {adj}, violets are {word},\n",
    "The {noun} was shining {word},\n",
    "I walked along and saw something {word},\n",
    "The wind was blowing through the {word},\n",
    "She whispered softly, filled with {word},\n",
]

# Line 1 endings with adjective/noun fillers
TEMPLATE_FILLERS = {
    "red": {"adj": "red", "noun": "sun"},
    "blue": {"adj": "blue", "noun": "sky"},
    "day": {"adj": "gray", "noun": "light of day"},  # NOTE: "gray" not "day" in adj slot
    "night": {"adj": "bright", "noun": "dark of night"},
    "cold": {"adj": "cold", "noun": "winter cold"},
}


def _make_id(task_type: str, template_idx: int, rhyme_set_idx: int, variant: str) -> str:
    """Create deterministic example ID."""
    raw = f"{task_type}_{template_idx}_{rhyme_set_idx}_{variant}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def generate_rhyme_dataset(
    n_per_rhyme_set: int = 5,
    include_controls: bool = True,
    seed: int = 42,
) -> list[PlanningExample]:
    """Generate rhyme completion examples with ground-truth targets.
    
    For each rhyme set:
    1. Generate couplet prompts where line 1 ends with the anchor word
    2. Label the expected rhyme targets (common + uncommon)
    3. If include_controls: add non-rhyming prompts as negative controls
    
    Args:
        n_per_rhyme_set: Number of prompt variants per rhyme set
        include_controls: Whether to include non-rhyming control examples
        seed: Random seed for reproducibility
        
    Returns:
        List of PlanningExample with rhyme targets
    """
    import random
    rng = random.Random(seed)
    
    examples = []
    
    for rs_idx, (anchor, common_rhymes, uncommon_rhymes) in enumerate(RHYME_SETS):
        all_rhymes = common_rhymes + uncommon_rhymes
        
        # Build prompts using simple templates
        # We use straightforward couplet starts rather than complex templates
        # to keep the task clean and the commitment signal interpretable
        prompts = _build_rhyme_prompts(anchor, n_per_rhyme_set, rng)
        
        for p_idx, prompt in enumerate(prompts):
            # Primary target: most common rhyme
            primary_target = common_rhymes[0]
            
            ex = PlanningExample(
                task_type=TaskType.RHYME,
                prompt=prompt,
                target_value=primary_target,
                target_token_positions=[],  # filled during activation extraction
                metadata={
                    "anchor_word": anchor,
                    "all_valid_rhymes": all_rhymes,
                    "common_rhymes": common_rhymes,
                    "uncommon_rhymes": uncommon_rhymes,
                    "is_control": False,
                },
                example_id=_make_id("rhyme", p_idx, rs_idx, "primary"),
            )
            examples.append(ex)
            
            # Contrastive variant: same prompt but different expected rhyme
            if len(common_rhymes) > 1:
                alt_target = common_rhymes[1]
                ex_alt = PlanningExample(
                    task_type=TaskType.RHYME,
                    prompt=prompt,
                    target_value=alt_target,
                    target_token_positions=[],
                    metadata={
                        "anchor_word": anchor,
                        "all_valid_rhymes": all_rhymes,
                        "common_rhymes": common_rhymes,
                        "uncommon_rhymes": uncommon_rhymes,
                        "is_control": False,
                        "is_contrastive": True,
                        "contrastive_pair_id": ex.example_id,
                    },
                    example_id=_make_id("rhyme", p_idx, rs_idx, "contrastive"),
                )
                examples.append(ex_alt)
        
        # Control: non-rhyming completions with same structure
        if include_controls:
            for p_idx, prompt in enumerate(prompts[:2]):
                # Replace the rhyming context with a non-rhyming one
                control_prompt = prompt.replace(
                    anchor,
                    "something"  # generic word, no rhyme constraint
                )
                ex_ctrl = PlanningExample(
                    task_type=TaskType.RHYME,
                    prompt=control_prompt,
                    target_value="",  # no specific target
                    target_token_positions=[],
                    metadata={
                        "anchor_word": "something",
                        "is_control": True,
                        "control_type": "no_rhyme_constraint",
                    },
                    example_id=_make_id("rhyme_ctrl", p_idx, rs_idx, "control"),
                )
                examples.append(ex_ctrl)
    
    return examples


def _build_rhyme_prompts(
    anchor: str,
    n: int,
    rng,
) -> list[str]:
    """Build n couplet prompts ending with the anchor word.
    
    Uses simple, naturalistic couplet structures that GPT-2 is likely
    to have seen during training, to maximize the chance of observing
    genuine planning behavior rather than confusion.
    """
    prompts = []
    
    # Template 1: Classic "Roses are X" structure
    prompts.append(f"Roses are {anchor}, violets are blue,\n")
    
    # Template 2: "The X was Y" narrative start
    prompts.append(f"The night was dark, the moon was {anchor},\n")
    
    # Template 3: Instruction-style
    prompts.append(f"Write a couplet that rhymes with {anchor}:\nThe world is full of things that are {anchor},\n")
    
    # Template 4: Poetry completion
    prompts.append(f"Complete this poem:\nI looked up at the sky so {anchor},\n")
    
    # Template 5: Simple fill-in
    prompts.append(f"Finish the rhyme: The cat sat on the mat so {anchor},\n")
    
    # Template 6: With explicit rhyme instruction
    prompts.append(f"Write the second line of a couplet. The first line ends with '{anchor}'.\nFirst line ends with: {anchor}\nSecond line: ")
    
    # Return up to n prompts, cycling if needed
    result = []
    for i in range(n):
        result.append(prompts[i % len(prompts)])
    
    return result


def generate_minimal_rhyme_pairs(
    n_pairs: int = 50,
) -> list[tuple[PlanningExample, PlanningExample]]:
    """Generate minimal pairs for contrastive analysis.
    
    Each pair: same prompt structure, different anchor word → different rhyme target.
    This controls for everything except the rhyme commitment.
    
    Example pair:
        A: "The sky was _red_, \\n" → expects "said"/"head"/etc.
        B: "The sky was _blue_, \\n" → expects "true"/"new"/etc.
    """
    pairs = []
    
    # Use rhyme sets in pairs
    for i in range(0, min(len(RHYME_SETS) - 1, n_pairs * 2), 2):
        anchor_a, rhymes_a, _ = RHYME_SETS[i]
        anchor_b, rhymes_b, _ = RHYME_SETS[i + 1]
        
        # Same template, different anchor
        template = "The world was full of things so {anchor},\n"
        
        ex_a = PlanningExample(
            task_type=TaskType.RHYME,
            prompt=template.format(anchor=anchor_a),
            target_value=rhymes_a[0],
            target_token_positions=[],
            metadata={
                "anchor_word": anchor_a,
                "all_valid_rhymes": rhymes_a,
                "pair_type": "minimal",
            },
            example_id=_make_id("rhyme_pair", i, 0, "a"),
        )
        
        ex_b = PlanningExample(
            task_type=TaskType.RHYME,
            prompt=template.format(anchor=anchor_b),
            target_value=rhymes_b[0],
            target_token_positions=[],
            metadata={
                "anchor_word": anchor_b,
                "all_valid_rhymes": rhymes_b,
                "pair_type": "minimal",
            },
            example_id=_make_id("rhyme_pair", i, 0, "b"),
        )
        
        pairs.append((ex_a, ex_b))
        
        if len(pairs) >= n_pairs:
            break
    
    return pairs


def save_dataset(examples: list[PlanningExample], path: str | Path) -> None:
    """Save dataset to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        "metadata": {
            "task_type": "rhyme",
            "n_examples": len(examples),
            "n_controls": sum(1 for e in examples if e.metadata.get("is_control")),
            "n_contrastive": sum(1 for e in examples if e.metadata.get("is_contrastive")),
        },
        "examples": [
            {
                "task_type": ex.task_type.value,
                "prompt": ex.prompt,
                "target_value": ex.target_value,
                "target_token_positions": ex.target_token_positions,
                "metadata": ex.metadata,
                "example_id": ex.example_id,
            }
            for ex in examples
        ],
    }
    
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_dataset(path: str | Path) -> list[PlanningExample]:
    """Load dataset from JSON."""
    with open(path) as f:
        data = json.load(f)
    
    return [
        PlanningExample(
            task_type=TaskType(ex["task_type"]),
            prompt=ex["prompt"],
            target_value=ex["target_value"],
            target_token_positions=ex["target_token_positions"],
            metadata=ex["metadata"],
            example_id=ex["example_id"],
        )
        for ex in data["examples"]
    ]
