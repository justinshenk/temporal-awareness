"""Trivia dataset — negative control for the staircase diagnostic.

DESIGN
------
We need a 5-class task where the answer is fully determined by surface
content of the question. The probe at the target position should NOT
beat the earlier-position baseline because all positions of the prompt
already carry enough information to classify.

This mirrors the code task in structure:
    code task:    5 return types × ~100 signatures   → predict return type
    trivia task:  5 question types × 100 prompts      → predict question type

The classification label is the *question category*, not the answer.
This is intentional: the category is recoverable from the question's
surface words ("capital", "symbol", "currency", etc.), so even a BoW
probe should get near-perfect accuracy and the target-position probe
should provide no additional information.

Same prompt format as Maar's QA dataset for cross-task comparability:
    "Question: {q}\\nAnswer:"  (no few-shot prefix — we don't need to
    pin a/an article patterns here)
"""

from __future__ import annotations

import hashlib
import random
from typing import Literal

from ..utils.types import PlanningExample, TaskType


# ──────────────────────────────────────────────────────────────────────
# Category content
# ──────────────────────────────────────────────────────────────────────

CAPITALS = [
    # 100 country-capital pairs
    ("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"),
    ("Spain", "Madrid"), ("Portugal", "Lisbon"), ("Greece", "Athens"),
    ("Russia", "Moscow"), ("Poland", "Warsaw"), ("Sweden", "Stockholm"),
    ("Norway", "Oslo"), ("Finland", "Helsinki"), ("Denmark", "Copenhagen"),
    ("Netherlands", "Amsterdam"), ("Belgium", "Brussels"), ("Austria", "Vienna"),
    ("Switzerland", "Bern"), ("Ireland", "Dublin"), ("Iceland", "Reykjavik"),
    ("Hungary", "Budapest"), ("Romania", "Bucharest"), ("Bulgaria", "Sofia"),
    ("Croatia", "Zagreb"), ("Serbia", "Belgrade"), ("Albania", "Tirana"),
    ("Ukraine", "Kyiv"), ("Belarus", "Minsk"), ("Lithuania", "Vilnius"),
    ("Latvia", "Riga"), ("Estonia", "Tallinn"), ("Czechia", "Prague"),
    ("Slovakia", "Bratislava"), ("Slovenia", "Ljubljana"), ("Moldova", "Chisinau"),
    ("Japan", "Tokyo"), ("China", "Beijing"), ("India", "Delhi"),
    ("Pakistan", "Islamabad"), ("Bangladesh", "Dhaka"), ("Nepal", "Kathmandu"),
    ("Thailand", "Bangkok"), ("Vietnam", "Hanoi"), ("Indonesia", "Jakarta"),
    ("Malaysia", "Kuala Lumpur"), ("Philippines", "Manila"), ("Singapore", "Singapore"),
    ("South Korea", "Seoul"), ("North Korea", "Pyongyang"), ("Mongolia", "Ulaanbaatar"),
    ("Kazakhstan", "Astana"), ("Uzbekistan", "Tashkent"), ("Turkmenistan", "Ashgabat"),
    ("Iran", "Tehran"), ("Iraq", "Baghdad"), ("Turkey", "Ankara"),
    ("Syria", "Damascus"), ("Lebanon", "Beirut"), ("Jordan", "Amman"),
    ("Israel", "Jerusalem"), ("Saudi Arabia", "Riyadh"), ("Yemen", "Sanaa"),
    ("Egypt", "Cairo"), ("Libya", "Tripoli"), ("Tunisia", "Tunis"),
    ("Algeria", "Algiers"), ("Morocco", "Rabat"), ("Sudan", "Khartoum"),
    ("Ethiopia", "Addis Ababa"), ("Kenya", "Nairobi"), ("Uganda", "Kampala"),
    ("Tanzania", "Dodoma"), ("Nigeria", "Abuja"), ("Ghana", "Accra"),
    ("Senegal", "Dakar"), ("Mali", "Bamako"), ("Cameroon", "Yaounde"),
    ("Angola", "Luanda"), ("Mozambique", "Maputo"), ("Zambia", "Lusaka"),
    ("Zimbabwe", "Harare"), ("Botswana", "Gaborone"), ("Namibia", "Windhoek"),
    ("Madagascar", "Antananarivo"), ("Canada", "Ottawa"), ("Mexico", "Mexico City"),
    ("Cuba", "Havana"), ("Jamaica", "Kingston"), ("Haiti", "Port-au-Prince"),
    ("Guatemala", "Guatemala City"), ("Honduras", "Tegucigalpa"), ("Panama", "Panama City"),
    ("Colombia", "Bogota"), ("Venezuela", "Caracas"), ("Ecuador", "Quito"),
    ("Peru", "Lima"), ("Bolivia", "La Paz"), ("Chile", "Santiago"),
    ("Argentina", "Buenos Aires"), ("Uruguay", "Montevideo"), ("Paraguay", "Asuncion"),
    ("Brazil", "Brasilia"), ("Australia", "Canberra"), ("New Zealand", "Wellington"),
]
CAPITALS = CAPITALS[:100]
assert len(CAPITALS) == 100


ELEMENTS = [
    # 100 element-symbol pairs (first 100 by atomic number)
    ("Hydrogen", "H"), ("Helium", "He"), ("Lithium", "Li"), ("Beryllium", "Be"),
    ("Boron", "B"), ("Carbon", "C"), ("Nitrogen", "N"), ("Oxygen", "O"),
    ("Fluorine", "F"), ("Neon", "Ne"), ("Sodium", "Na"), ("Magnesium", "Mg"),
    ("Aluminum", "Al"), ("Silicon", "Si"), ("Phosphorus", "P"), ("Sulfur", "S"),
    ("Chlorine", "Cl"), ("Argon", "Ar"), ("Potassium", "K"), ("Calcium", "Ca"),
    ("Scandium", "Sc"), ("Titanium", "Ti"), ("Vanadium", "V"), ("Chromium", "Cr"),
    ("Manganese", "Mn"), ("Iron", "Fe"), ("Cobalt", "Co"), ("Nickel", "Ni"),
    ("Copper", "Cu"), ("Zinc", "Zn"), ("Gallium", "Ga"), ("Germanium", "Ge"),
    ("Arsenic", "As"), ("Selenium", "Se"), ("Bromine", "Br"), ("Krypton", "Kr"),
    ("Rubidium", "Rb"), ("Strontium", "Sr"), ("Yttrium", "Y"), ("Zirconium", "Zr"),
    ("Niobium", "Nb"), ("Molybdenum", "Mo"), ("Technetium", "Tc"), ("Ruthenium", "Ru"),
    ("Rhodium", "Rh"), ("Palladium", "Pd"), ("Silver", "Ag"), ("Cadmium", "Cd"),
    ("Indium", "In"), ("Tin", "Sn"), ("Antimony", "Sb"), ("Tellurium", "Te"),
    ("Iodine", "I"), ("Xenon", "Xe"), ("Cesium", "Cs"), ("Barium", "Ba"),
    ("Lanthanum", "La"), ("Cerium", "Ce"), ("Praseodymium", "Pr"), ("Neodymium", "Nd"),
    ("Promethium", "Pm"), ("Samarium", "Sm"), ("Europium", "Eu"), ("Gadolinium", "Gd"),
    ("Terbium", "Tb"), ("Dysprosium", "Dy"), ("Holmium", "Ho"), ("Erbium", "Er"),
    ("Thulium", "Tm"), ("Ytterbium", "Yb"), ("Lutetium", "Lu"), ("Hafnium", "Hf"),
    ("Tantalum", "Ta"), ("Tungsten", "W"), ("Rhenium", "Re"), ("Osmium", "Os"),
    ("Iridium", "Ir"), ("Platinum", "Pt"), ("Gold", "Au"), ("Mercury", "Hg"),
    ("Thallium", "Tl"), ("Lead", "Pb"), ("Bismuth", "Bi"), ("Polonium", "Po"),
    ("Astatine", "At"), ("Radon", "Rn"), ("Francium", "Fr"), ("Radium", "Ra"),
    ("Actinium", "Ac"), ("Thorium", "Th"), ("Protactinium", "Pa"), ("Uranium", "U"),
    ("Neptunium", "Np"), ("Plutonium", "Pu"), ("Americium", "Am"), ("Curium", "Cm"),
    ("Berkelium", "Bk"), ("Californium", "Cf"), ("Einsteinium", "Es"), ("Fermium", "Fm"),
]
ELEMENTS = ELEMENTS[:100]
assert len(ELEMENTS) == 100


CURRENCIES = [
    # 100 country-currency pairs
    ("Japan", "Yen"), ("United Kingdom", "Pound"), ("Switzerland", "Franc"),
    ("Russia", "Ruble"), ("India", "Rupee"), ("Pakistan", "Rupee"),
    ("China", "Yuan"), ("Vietnam", "Dong"), ("Thailand", "Baht"),
    ("South Korea", "Won"), ("North Korea", "Won"), ("Singapore", "Dollar"),
    ("Malaysia", "Ringgit"), ("Indonesia", "Rupiah"), ("Philippines", "Peso"),
    ("Australia", "Dollar"), ("New Zealand", "Dollar"), ("Canada", "Dollar"),
    ("Mexico", "Peso"), ("Argentina", "Peso"), ("Chile", "Peso"),
    ("Colombia", "Peso"), ("Uruguay", "Peso"), ("Cuba", "Peso"),
    ("Brazil", "Real"), ("Peru", "Sol"), ("Venezuela", "Bolivar"),
    ("Egypt", "Pound"), ("Lebanon", "Pound"), ("Syria", "Pound"),
    ("Sudan", "Pound"), ("South Sudan", "Pound"), ("Israel", "Shekel"),
    ("Turkey", "Lira"), ("Iran", "Rial"), ("Iraq", "Dinar"),
    ("Kuwait", "Dinar"), ("Bahrain", "Dinar"), ("Jordan", "Dinar"),
    ("Saudi Arabia", "Riyal"), ("Qatar", "Riyal"), ("Yemen", "Riyal"),
    ("Oman", "Rial"), ("Morocco", "Dirham"), ("Tunisia", "Dinar"),
    ("Algeria", "Dinar"), ("Libya", "Dinar"), ("Ethiopia", "Birr"),
    ("Kenya", "Shilling"), ("Tanzania", "Shilling"), ("Uganda", "Shilling"),
    ("Nigeria", "Naira"), ("Ghana", "Cedi"), ("Senegal", "Franc"),
    ("Cameroon", "Franc"), ("Mali", "Franc"), ("Madagascar", "Ariary"),
    ("Zambia", "Kwacha"), ("Malawi", "Kwacha"), ("Angola", "Kwanza"),
    ("Mozambique", "Metical"), ("Zimbabwe", "Dollar"), ("South Africa", "Rand"),
    ("Namibia", "Dollar"), ("Botswana", "Pula"), ("Lesotho", "Loti"),
    ("Eswatini", "Lilangeni"), ("Sweden", "Krona"), ("Norway", "Krone"),
    ("Denmark", "Krone"), ("Iceland", "Krona"), ("Czechia", "Koruna"),
    ("Poland", "Zloty"), ("Hungary", "Forint"), ("Romania", "Leu"),
    ("Moldova", "Leu"), ("Bulgaria", "Lev"), ("Albania", "Lek"),
    ("Serbia", "Dinar"), ("Macedonia", "Denar"), ("Bosnia", "Mark"),
    ("Ukraine", "Hryvnia"), ("Belarus", "Ruble"), ("Georgia", "Lari"),
    ("Armenia", "Dram"), ("Azerbaijan", "Manat"), ("Turkmenistan", "Manat"),
    ("Kazakhstan", "Tenge"), ("Uzbekistan", "Som"), ("Kyrgyzstan", "Som"),
    ("Tajikistan", "Somoni"), ("Mongolia", "Tugrik"), ("Afghanistan", "Afghani"),
    ("Nepal", "Rupee"), ("Sri Lanka", "Rupee"), ("Bangladesh", "Taka"),
    ("Myanmar", "Kyat"), ("Laos", "Kip"), ("Cambodia", "Riel"),
    ("Bhutan", "Ngultrum"), ("Fiji", "Dollar"), ("Samoa", "Tala"),
    ("Iceland", "Krona"),
]
CURRENCIES = CURRENCIES[:100]
assert len(CURRENCIES) == 100


LANGUAGES = [
    # 100 country-official-language pairs
    ("France", "French"), ("Germany", "German"), ("Italy", "Italian"),
    ("Spain", "Spanish"), ("Portugal", "Portuguese"), ("Greece", "Greek"),
    ("Russia", "Russian"), ("Poland", "Polish"), ("Sweden", "Swedish"),
    ("Norway", "Norwegian"), ("Finland", "Finnish"), ("Denmark", "Danish"),
    ("Netherlands", "Dutch"), ("Iceland", "Icelandic"), ("Hungary", "Hungarian"),
    ("Romania", "Romanian"), ("Bulgaria", "Bulgarian"), ("Croatia", "Croatian"),
    ("Serbia", "Serbian"), ("Albania", "Albanian"), ("Ukraine", "Ukrainian"),
    ("Belarus", "Belarusian"), ("Lithuania", "Lithuanian"), ("Latvia", "Latvian"),
    ("Estonia", "Estonian"), ("Czechia", "Czech"), ("Slovakia", "Slovak"),
    ("Slovenia", "Slovenian"), ("Moldova", "Romanian"), ("Japan", "Japanese"),
    ("China", "Chinese"), ("India", "Hindi"), ("Pakistan", "Urdu"),
    ("Bangladesh", "Bengali"), ("Nepal", "Nepali"), ("Thailand", "Thai"),
    ("Vietnam", "Vietnamese"), ("Indonesia", "Indonesian"), ("Malaysia", "Malay"),
    ("Philippines", "Filipino"), ("South Korea", "Korean"), ("North Korea", "Korean"),
    ("Mongolia", "Mongolian"), ("Kazakhstan", "Kazakh"), ("Uzbekistan", "Uzbek"),
    ("Turkmenistan", "Turkmen"), ("Iran", "Persian"), ("Iraq", "Arabic"),
    ("Turkey", "Turkish"), ("Syria", "Arabic"), ("Lebanon", "Arabic"),
    ("Jordan", "Arabic"), ("Saudi Arabia", "Arabic"), ("Yemen", "Arabic"),
    ("Egypt", "Arabic"), ("Libya", "Arabic"), ("Tunisia", "Arabic"),
    ("Algeria", "Arabic"), ("Morocco", "Arabic"), ("Sudan", "Arabic"),
    ("Ethiopia", "Amharic"), ("Somalia", "Somali"), ("Kenya", "Swahili"),
    ("Uganda", "Swahili"), ("Tanzania", "Swahili"), ("Rwanda", "Kinyarwanda"),
    ("Nigeria", "English"), ("Ghana", "English"), ("Senegal", "French"),
    ("Mali", "French"), ("Cameroon", "French"), ("Madagascar", "Malagasy"),
    ("Angola", "Portuguese"), ("Mozambique", "Portuguese"), ("South Africa", "English"),
    ("Zimbabwe", "English"), ("Botswana", "English"), ("Namibia", "English"),
    ("Canada", "English"), ("Mexico", "Spanish"), ("Cuba", "Spanish"),
    ("Jamaica", "English"), ("Haiti", "French"), ("Guatemala", "Spanish"),
    ("Honduras", "Spanish"), ("Panama", "Spanish"), ("Colombia", "Spanish"),
    ("Venezuela", "Spanish"), ("Ecuador", "Spanish"), ("Peru", "Spanish"),
    ("Bolivia", "Spanish"), ("Chile", "Spanish"), ("Argentina", "Spanish"),
    ("Uruguay", "Spanish"), ("Paraguay", "Spanish"), ("Brazil", "Portuguese"),
    ("Australia", "English"), ("New Zealand", "English"), ("Ireland", "Irish"),
    ("Wales", "Welsh"), ("Scotland", "English"), ("Catalonia", "Catalan"),
    ("Quebec", "French"),
]
LANGUAGES = LANGUAGES[:100]
assert len(LANGUAGES) == 100


CONTINENTS = [
    # 100 country-continent pairs
    ("France", "Europe"), ("Germany", "Europe"), ("Italy", "Europe"),
    ("Spain", "Europe"), ("Portugal", "Europe"), ("Greece", "Europe"),
    ("Russia", "Europe"), ("Poland", "Europe"), ("Sweden", "Europe"),
    ("Norway", "Europe"), ("Finland", "Europe"), ("Denmark", "Europe"),
    ("Netherlands", "Europe"), ("Belgium", "Europe"), ("Austria", "Europe"),
    ("Switzerland", "Europe"), ("Ireland", "Europe"), ("Iceland", "Europe"),
    ("Hungary", "Europe"), ("Romania", "Europe"), ("Bulgaria", "Europe"),
    ("Croatia", "Europe"), ("Serbia", "Europe"), ("Ukraine", "Europe"),
    ("Lithuania", "Europe"), ("Latvia", "Europe"), ("Estonia", "Europe"),
    ("Czechia", "Europe"), ("Slovakia", "Europe"), ("Slovenia", "Europe"),
    ("Japan", "Asia"), ("China", "Asia"), ("India", "Asia"),
    ("Pakistan", "Asia"), ("Bangladesh", "Asia"), ("Nepal", "Asia"),
    ("Thailand", "Asia"), ("Vietnam", "Asia"), ("Indonesia", "Asia"),
    ("Malaysia", "Asia"), ("Philippines", "Asia"), ("Singapore", "Asia"),
    ("South Korea", "Asia"), ("North Korea", "Asia"), ("Mongolia", "Asia"),
    ("Kazakhstan", "Asia"), ("Uzbekistan", "Asia"), ("Iran", "Asia"),
    ("Iraq", "Asia"), ("Turkey", "Asia"), ("Syria", "Asia"),
    ("Lebanon", "Asia"), ("Jordan", "Asia"), ("Israel", "Asia"),
    ("Saudi Arabia", "Asia"), ("Yemen", "Asia"), ("Sri Lanka", "Asia"),
    ("Egypt", "Africa"), ("Libya", "Africa"), ("Tunisia", "Africa"),
    ("Algeria", "Africa"), ("Morocco", "Africa"), ("Sudan", "Africa"),
    ("Ethiopia", "Africa"), ("Kenya", "Africa"), ("Uganda", "Africa"),
    ("Tanzania", "Africa"), ("Nigeria", "Africa"), ("Ghana", "Africa"),
    ("Senegal", "Africa"), ("Mali", "Africa"), ("Cameroon", "Africa"),
    ("Angola", "Africa"), ("Mozambique", "Africa"), ("Zambia", "Africa"),
    ("Zimbabwe", "Africa"), ("Botswana", "Africa"), ("Namibia", "Africa"),
    ("Madagascar", "Africa"), ("South Africa", "Africa"), ("Somalia", "Africa"),
    ("Canada", "North America"), ("Mexico", "North America"), ("Cuba", "North America"),
    ("Jamaica", "North America"), ("Haiti", "North America"), ("Guatemala", "North America"),
    ("Honduras", "North America"), ("Panama", "North America"), ("Nicaragua", "North America"),
    ("Costa Rica", "North America"), ("Colombia", "South America"), ("Venezuela", "South America"),
    ("Ecuador", "South America"), ("Peru", "South America"), ("Bolivia", "South America"),
    ("Chile", "South America"), ("Argentina", "South America"), ("Uruguay", "South America"),
    ("Paraguay", "South America"), ("Brazil", "South America"), ("Australia", "Oceania"),
]
CONTINENTS = CONTINENTS[:100]
assert len(CONTINENTS) == 100


# ──────────────────────────────────────────────────────────────────────
# Templates — distinctive surface markers make this a clean negative control
# ──────────────────────────────────────────────────────────────────────

# Each template ends with "Answer:" to match Maar's QA prompt format
# exactly, so within-example position structure (last_token, newline,
# answer-marker) is comparable across QA and trivia.

CATEGORY_TEMPLATES = {
    "capitals":   "Question: The capital of {x} is what?\nAnswer:",
    "elements":   "Question: The chemical symbol for {x} is what?\nAnswer:",
    "currencies": "Question: The currency of {x} is what?\nAnswer:",
    "languages":  "Question: The official language of {x} is what?\nAnswer:",
    "continents": "Question: On which continent is {x}?\nAnswer:",
}


# ──────────────────────────────────────────────────────────────────────
# Public loader
# ──────────────────────────────────────────────────────────────────────

CATEGORIES = ("capitals", "elements", "currencies", "languages", "continents")


def _make_id(*parts) -> str:
    raw = "::".join(str(p) for p in parts)
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def load_trivia(
    split: Literal["train", "test", "all"] = "all",
    seed: int = 42,
    test_frac: float = 0.2,
) -> list[PlanningExample]:
    """Generate the trivia negative-control dataset.

    500 examples total: 5 categories × 100 each.

    Args:
        split: "train" (80%), "test" (20%), or "all" (everything).
            Split is stratified by category, deterministic by seed.
        seed: Random seed for the train/test split.
        test_frac: Test fraction (default 0.2).

    Returns:
        List of PlanningExample with metadata.category as the
        classification target.
    """
    if split not in ("train", "test", "all"):
        raise ValueError(f"split must be train/test/all, got {split!r}")

    rng = random.Random(seed)
    examples: list[PlanningExample] = []

    sources = {
        "capitals":   CAPITALS,
        "elements":   ELEMENTS,
        "currencies": CURRENCIES,
        "languages":  LANGUAGES,
        "continents": CONTINENTS,
    }

    for category in CATEGORIES:
        items = list(sources[category])
        template = CATEGORY_TEMPLATES[category]

        if split != "all":
            shuffled = items[:]
            rng.shuffle(shuffled)
            n_test = int(round(len(shuffled) * test_frac))
            if split == "test":
                items = shuffled[:n_test]
            else:  # train
                items = shuffled[n_test:]

        for item_idx, (x, answer) in enumerate(items):
            prompt = template.format(x=x)
            ex = PlanningExample(
                task_type=TaskType.CODE_RETURN,  # placeholder (see maar_data.py note)
                prompt=prompt,
                target_value=category,
                target_token_positions=[],
                metadata={
                    "source": "trivia_negative_control",
                    "split": split,
                    "category": category,
                    "subject": x,
                    "expected_answer": answer,
                    "is_control": False,
                },
                example_id=_make_id("trivia", category, x),
            )
            examples.append(ex)
    return examples


def summarize(examples: list[PlanningExample]) -> dict:
    from collections import Counter
    cat_counts = Counter(e.metadata.get("category") for e in examples)
    return {
        "n": len(examples),
        "kind": "trivia",
        "categories": dict(cat_counts),
        "n_categories": len(cat_counts),
    }


__all__ = ["CATEGORIES", "CATEGORY_TEMPLATES", "load_trivia", "summarize"]
