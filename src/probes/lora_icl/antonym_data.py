"""Antonym word-pair dataset for the function-vector vs LoRA "two routes" experiment.

Antonyms are the canonical function-vector task (Todd et al. ICLR 2024; Hendel et al. EMNLP
2023): in the bare ``word: antonym`` format the model cannot know *which* function is wanted
without the demonstrations, so ICL carries a real, recoverable signal (unlike DDXPlus, where
prior knowledge makes the demo labels inert). The pairs are curated common single-word antonyms;
``antonym_split`` gives a deterministic train (LoRA) / held-out (FV + eval) partition.
"""

from __future__ import annotations

ANTONYM_PAIRS: list[tuple[str, str]] = [
    ("hot", "cold"), ("big", "small"), ("tall", "short"), ("fast", "slow"),
    ("happy", "sad"), ("good", "bad"), ("light", "heavy"), ("bright", "dark"),
    ("high", "low"), ("open", "shut"), ("up", "down"), ("left", "right"),
    ("hard", "easy"), ("wet", "dry"), ("clean", "dirty"), ("full", "empty"),
    ("rich", "poor"), ("young", "old"), ("fresh", "stale"), ("strong", "weak"),
    ("thick", "thin"), ("wide", "narrow"), ("deep", "shallow"), ("loud", "quiet"),
    ("sharp", "blunt"), ("smooth", "rough"), ("sweet", "sour"), ("near", "far"),
    ("early", "late"), ("true", "false"), ("win", "lose"), ("buy", "sell"),
    ("push", "pull"), ("give", "take"), ("love", "hate"), ("war", "peace"),
    ("day", "night"), ("black", "white"), ("yes", "no"), ("more", "less"),
    ("many", "few"), ("begin", "end"), ("first", "last"), ("best", "worst"),
    ("top", "bottom"), ("front", "back"), ("over", "under"), ("above", "below"),
    ("north", "south"), ("east", "west"), ("male", "female"), ("boy", "girl"),
    ("king", "queen"), ("brother", "sister"), ("accept", "reject"), ("allow", "forbid"),
    ("arrive", "depart"), ("asleep", "awake"), ("attack", "defend"), ("beautiful", "ugly"),
    ("brave", "afraid"), ("build", "destroy"), ("calm", "angry"), ("cheap", "expensive"),
    ("clever", "stupid"), ("come", "go"), ("cry", "laugh"), ("dangerous", "safe"),
    ("dead", "alive"), ("deny", "admit"), ("difficult", "simple"), ("dull", "interesting"),
    ("enemy", "friend"), ("enter", "exit"), ("evening", "morning"), ("expand", "shrink"),
    ("fail", "succeed"), ("fall", "rise"), ("find", "lose"), ("float", "sink"),
    ("forget", "remember"), ("frequent", "rare"), ("future", "past"), ("generous", "selfish"),
    ("giant", "dwarf"), ("guilty", "innocent"), ("healthy", "sick"), ("hide", "show"),
    ("joy", "sorrow"), ("lead", "follow"), ("learn", "teach"), ("leave", "stay"),
    ("lock", "unlock"), ("lucky", "unlucky"), ("major", "minor"), ("mean", "kind"),
    ("modern", "ancient"), ("neat", "messy"), ("never", "always"), ("normal", "strange"),
    ("often", "seldom"), ("polite", "rude"), ("positive", "negative"), ("private", "public"),
    ("rapid", "sluggish"), ("rare", "common"), ("raw", "cooked"), ("real", "fake"),
    ("save", "spend"), ("scatter", "gather"), ("separate", "join"), ("shrink", "grow"),
    ("single", "married"), ("sit", "stand"), ("start", "stop"), ("straight", "crooked"),
    ("tame", "wild"), ("tight", "loose"), ("vacant", "occupied"), ("victory", "defeat"),
    ("visible", "hidden"), ("warm", "cool"), ("wealth", "poverty"), ("wise", "foolish"),
]


def antonym_split(n_train: int) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Deterministic train (first ``n_train``) / held-out (rest) split of the pairs."""
    return ANTONYM_PAIRS[:n_train], ANTONYM_PAIRS[n_train:]
