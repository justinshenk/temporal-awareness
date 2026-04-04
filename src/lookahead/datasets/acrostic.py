"""Acrostic dataset for planning detection.

In an acrostic, the first letters of each line spell out a hidden word.
This is ideal for studying lookahead because:

1. The model must plan the *next* word's first letter before generating it
2. The commitment can be measured letter-by-letter as the acrostic progresses
3. We can control difficulty by varying how many letters are already revealed

Key experimental design:
- We give the model partial acrostics and measure when it commits to the target word
- Progressive reveal: give 1 letter, 2 letters, ..., n-1 letters
- At each stage, probe whether the model has committed to a specific completion
"""

from __future__ import annotations

import hashlib
from typing import Optional

from ..utils.types import PlanningExample, TaskType


# Target words for acrostics, grouped by length and difficulty
# Shorter words → easier planning task (fewer future commitments)
ACROSTIC_TARGETS = {
    # 3-letter: easiest — model needs to plan just 2-3 tokens ahead
    3: ["CAT", "DOG", "SUN", "RUN", "FLY", "JOY", "SKY", "RED", "BIG", "OLD"],
    # 4-letter: moderate
    4: ["LOVE", "HOPE", "STAR", "FIRE", "MOON", "RAIN", "TREE", "WIND", "GOLD", "SONG"],
    # 5-letter: harder — requires more sustained planning
    5: ["DREAM", "LIGHT", "PEACE", "STORM", "OCEAN", "BRAVE", "NIGHT", "SWEET", "MAGIC", "WORLD"],
    # 6-letter: hardest for GPT-2 scale models
    6: ["NATURE", "GARDEN", "SPIRIT", "WONDER", "SUMMER", "WINTER"],
}


def _make_acrostic_prompt(
    target: str,
    n_revealed: int,
    style: str = "poem",
) -> str:
    """Build a prompt with n_revealed lines of the acrostic already present.
    
    Args:
        target: The full acrostic word (e.g., "DREAM")
        n_revealed: How many lines are already given (0 = just instruction)
        style: "poem" for poetic lines, "list" for simple word lists
    """
    # Pre-built line completions for each letter
    # These are simple, naturalistic completions that start with the required letter
    LETTER_LINES = {
        "A": "Above the hills the sun arose",
        "B": "Beneath the stars we lay our heads",
        "C": "Calling out across the field",
        "D": "Dancing shadows on the wall",
        "E": "Every moment counts for something",
        "F": "Flying high above the clouds",
        "G": "Gentle breezes in the spring",
        "H": "Hoping for a better day",
        "I": "In the quiet of the night",
        "J": "Just beyond the garden gate",
        "K": "Keeping watch until the dawn",
        "L": "Laughter fills the evening air",
        "M": "Mountains rising in the mist",
        "N": "Nothing lasts but memories",
        "O": "Over fields of golden wheat",
        "P": "Peaceful waters flow downstream",
        "Q": "Quietly the snow begins",
        "R": "Running through the morning dew",
        "S": "Sailing on a silver sea",
        "T": "Through the forest dark and deep",
        "U": "Under skies of endless blue",
        "V": "Voices carry on the wind",
        "W": "Wandering through ancient woods",
        "X": "Xenial warmth from hearth and home",
        "Y": "Yearning for the days of old",
        "Z": "Zephyrs whisper through the leaves",
    }
    
    instruction = f'Write an acrostic poem where the first letter of each line spells "{target}".\n\n'
    
    lines = []
    for i in range(n_revealed):
        letter = target[i]
        lines.append(LETTER_LINES.get(letter, f"{letter}ight shines upon the path"))
    
    if lines:
        return instruction + "\n".join(lines) + "\n"
    else:
        return instruction


def generate_acrostic_dataset(
    word_lengths: list[int] | None = None,
    n_per_word: int = 1,
    include_progressive: bool = True,
    seed: int = 42,
) -> list[PlanningExample]:
    """Generate acrostic planning examples.
    
    For each target word, generates:
    1. Full prompt (just instruction, 0 lines revealed)
    2. If include_progressive: prompts with 1, 2, ..., n-1 lines revealed
       This creates a "progressive reveal" series to track commitment
    
    Args:
        word_lengths: Which word lengths to include (default: all)
        n_per_word: Number of style variants per word
        include_progressive: Whether to generate progressive reveal series
        seed: Random seed
        
    Returns:
        List of PlanningExample
    """
    if word_lengths is None:
        word_lengths = [3, 4, 5]  # Skip 6-letter by default (too hard for GPT-2)
    
    examples = []
    
    for length in word_lengths:
        words = ACROSTIC_TARGETS.get(length, [])
        
        for word in words:
            if include_progressive:
                # Generate the progressive series:
                # 0 revealed → 1 revealed → ... → (n-1) revealed
                for n_revealed in range(len(word)):
                    next_letter = word[n_revealed]
                    # The "target" at this stage is the next letter
                    # and ultimately the full remaining word
                    remaining = word[n_revealed:]
                    
                    prompt = _make_acrostic_prompt(word, n_revealed)
                    
                    ex = PlanningExample(
                        task_type=TaskType.ACROSTIC,
                        prompt=prompt,
                        target_value=next_letter,
                        target_token_positions=[],  # filled during extraction
                        metadata={
                            "full_word": word,
                            "word_length": length,
                            "n_revealed": n_revealed,
                            "remaining_letters": remaining,
                            "next_letter": next_letter,
                            "is_progressive": True,
                            "progress_fraction": n_revealed / len(word),
                        },
                        example_id=hashlib.md5(
                            f"acrostic_{word}_{n_revealed}".encode()
                        ).hexdigest()[:12],
                    )
                    examples.append(ex)
            else:
                # Just the full instruction, 0 lines revealed
                prompt = _make_acrostic_prompt(word, 0)
                
                ex = PlanningExample(
                    task_type=TaskType.ACROSTIC,
                    prompt=prompt,
                    target_value=word[0],
                    target_token_positions=[],
                    metadata={
                        "full_word": word,
                        "word_length": length,
                        "n_revealed": 0,
                        "remaining_letters": word,
                        "next_letter": word[0],
                        "is_progressive": False,
                    },
                    example_id=hashlib.md5(
                        f"acrostic_{word}_full".encode()
                    ).hexdigest()[:12],
                )
                examples.append(ex)
    
    return examples


def generate_acrostic_minimal_pairs(
    n_pairs: int = 30,
) -> list[tuple[PlanningExample, PlanningExample]]:
    """Generate minimal pairs for contrastive probing.
    
    Each pair: same number of revealed lines, same prompt structure,
    but different target words → different next letter.
    
    E.g., "DREAM" with 2 lines revealed (D, R) → next letter "E"
    vs.   "DRINK" with 2 lines revealed (D, R) → next letter "I"
    """
    pairs = []
    
    # Find words that share prefixes but diverge
    prefix_groups: dict[str, list[str]] = {}
    for length in [4, 5]:
        for word in ACROSTIC_TARGETS.get(length, []):
            for prefix_len in range(1, len(word)):
                prefix = word[:prefix_len]
                if prefix not in prefix_groups:
                    prefix_groups[prefix] = []
                prefix_groups[prefix].append(word)
    
    # Find pairs that share a prefix but have different next letter
    for prefix, words in prefix_groups.items():
        if len(words) < 2:
            continue
        
        # Check that the next letters differ
        next_letters = set(w[len(prefix)] for w in words if len(w) > len(prefix))
        if len(next_letters) < 2:
            continue
        
        w1, w2 = words[0], words[1]
        if w1[len(prefix)] == w2[len(prefix)]:
            continue
        
        n_revealed = len(prefix)
        
        prompt_a = _make_acrostic_prompt(w1, n_revealed)
        prompt_b = _make_acrostic_prompt(w2, n_revealed)
        
        ex_a = PlanningExample(
            task_type=TaskType.ACROSTIC,
            prompt=prompt_a,
            target_value=w1[n_revealed],
            target_token_positions=[],
            metadata={
                "full_word": w1,
                "n_revealed": n_revealed,
                "shared_prefix": prefix,
                "pair_type": "minimal",
            },
            example_id=hashlib.md5(
                f"acrostic_pair_{w1}_{w2}_{n_revealed}".encode()
            ).hexdigest()[:12],
        )
        
        ex_b = PlanningExample(
            task_type=TaskType.ACROSTIC,
            prompt=prompt_b,
            target_value=w2[n_revealed],
            target_token_positions=[],
            metadata={
                "full_word": w2,
                "n_revealed": n_revealed,
                "shared_prefix": prefix,
                "pair_type": "minimal",
            },
            example_id=hashlib.md5(
                f"acrostic_pair_{w2}_{w1}_{n_revealed}".encode()
            ).hexdigest()[:12],
        )
        
        pairs.append((ex_a, ex_b))
        
        if len(pairs) >= n_pairs:
            break
    
    return pairs
