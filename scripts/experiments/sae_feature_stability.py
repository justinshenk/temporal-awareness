#!/usr/bin/env python3
"""
SAE Feature Stability Under Distribution Shift

Core experiment for the "When Probes Fail" paper direction.

Research question: Do SAE-discovered temporal features remain stable
under systematic distribution shift? How does SAE probe degradation
compare to raw activation probe degradation?

Distribution shift conditions tested:
  1. Domain shift: financial → medical → personal → environmental
  2. Register shift: formal → casual → academic → conversational
  3. Negation: "not short-term" should map to long-term
  4. Implicit-only: no explicit temporal keywords
  5. Paraphrase: same meaning, different surface form
  6. Cross-dataset: trained on minimal pairs, tested on full CAA pairs

Methodology:
  - Train SAE probes and activation probes on in-distribution data
    (temporal_scope_pairs_minimal.json)
  - Identify top-k discriminative SAE latents
  - For each shift condition, measure:
    (a) Probe accuracy (SAE and raw activation)
    (b) Feature activation overlap (Jaccard similarity of active latents)
    (c) Mean activation magnitude change
    (d) Probe confidence calibration (ECE)
  - Produce degradation curves across conditions

Based on Gemma-2-2B + Gemma Scope SAEs (same as sae_temporal_probing.py).

Usage:
    # CPU (slow but works locally)
    python scripts/experiments/sae_feature_stability.py

    # GPU (recommended for full experiment)
    python scripts/experiments/sae_feature_stability.py --device cuda

    # Quick sanity check
    python scripts/experiments/sae_feature_stability.py --quick

Author: Adrian Sadik
Date: 2026-02-26
"""

import argparse
import json
import re
import random
import pickle
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, log_loss,
    precision_score, recall_score, f1_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# Optional W&B integration
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# transformer_lens + sae_lens for Gemma Scope SAEs
from transformer_lens import HookedTransformer
from sae_lens import SAE as SaeLensSAE


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "gemma-2-2b"
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
TOP_K_LATENTS = 64  # matches existing sae_temporal_probing.py

# Layers to probe — middle-ish layers encode semantics best in Gemma-2-2b (26 layers)
DEFAULT_LAYERS = [6, 13, 20, 24]
QUICK_LAYERS = [13]  # single layer for quick sanity checks

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "sae_feature_stability"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ShiftCondition:
    """A single distribution shift condition with its dataset."""
    name: str
    description: str
    prompts: list  # list[str]
    labels: np.ndarray  # 0=immediate, 1=long_term


@dataclass
class ProbeMetrics:
    """Metrics for a single probe evaluation."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    log_loss_value: float
    n_samples: int
    # Calibration
    mean_confidence: float  # mean predicted probability of chosen class
    ece: float  # expected calibration error


@dataclass
class FeatureStabilityMetrics:
    """Measures how SAE feature activations change under shift."""
    jaccard_similarity: float  # overlap of active latent indices
    activation_magnitude_ratio: float  # mean|shifted| / mean|original|
    top_k_overlap: float  # fraction of top-k latents that remain in top-k
    cosine_similarity: float  # cos(mean_activation_original, mean_activation_shifted)


@dataclass
class LayerResult:
    """Results for one layer across all shift conditions."""
    layer: int
    top_k_indices: list  # discriminative latent indices
    in_distribution: dict  # {"sae_probe": ProbeMetrics, "act_probe": ProbeMetrics}
    shift_results: dict  # {condition_name: {sae_probe, act_probe, feature_stability}}


# ---------------------------------------------------------------------------
# Dataset generation: distribution shift conditions
# ---------------------------------------------------------------------------

def load_in_distribution_data(dataset_path: Path) -> tuple[list, np.ndarray]:
    """Load the minimal pairs dataset (in-distribution training data)."""
    with open(dataset_path) as f:
        data = json.load(f)

    pairs = data["pairs"]
    prompts, labels = [], []
    for pair in pairs:
        prompts.append(pair["immediate"])
        labels.append(0)
        prompts.append(pair["long_term"])
        labels.append(1)

    return prompts, np.array(labels)


def load_caa_dataset(dataset_path: Path) -> tuple[list, np.ndarray]:
    """Load a CAA-format dataset (question + answer)."""
    with open(dataset_path) as f:
        data = json.load(f)

    pairs = data["pairs"]
    prompts, labels = [], []
    for pair in pairs:
        question = pair["question"]
        # Strip (A)/(B) prefix from answers
        imm = re.sub(r"^\s*\([AB]\)\s*", "", pair["immediate"])
        lt = re.sub(r"^\s*\([AB]\)\s*", "", pair["long_term"])

        prompts.append(f"{question} {imm}")
        labels.append(0)
        prompts.append(f"{question} {lt}")
        labels.append(1)

    return prompts, np.array(labels)


def generate_domain_shift_data() -> ShiftCondition:
    """
    Domain shift: temporal reasoning in specific professional domains.
    Training data uses generic temporal words; these use domain-specific framing.
    """
    pairs = [
        # Financial domain
        {"immediate": "Liquidate the position before market close today",
         "long_term": "Hold for dividend compounding over the next decade"},
        {"immediate": "Cover the margin call this afternoon",
         "long_term": "Build a diversified retirement portfolio over 30 years"},
        {"immediate": "Sell at the current bid price",
         "long_term": "Dollar-cost average into index funds for decades"},
        {"immediate": "Execute the arbitrage before the spread closes",
         "long_term": "Establish generational wealth through trust structures"},
        {"immediate": "Meet this quarter's earnings target",
         "long_term": "Invest in R&D for market leadership in 2035"},

        # Medical domain
        {"immediate": "Administer epinephrine for anaphylaxis now",
         "long_term": "Design a longitudinal study tracking outcomes over 20 years"},
        {"immediate": "Stop the acute hemorrhage",
         "long_term": "Develop preventive screening programs for future generations"},
        {"immediate": "Prescribe antibiotics for the current infection",
         "long_term": "Address antibiotic resistance through decade-long stewardship"},
        {"immediate": "Stabilize the patient's vitals this hour",
         "long_term": "Implement chronic disease management for lifelong wellness"},
        {"immediate": "Triage incoming casualties",
         "long_term": "Build healthcare infrastructure for the next century"},

        # Environmental domain
        {"immediate": "Deploy boom barriers to contain the oil spill today",
         "long_term": "Transition energy systems to renewables over the next 30 years"},
        {"immediate": "Evacuate residents from the flood zone now",
         "long_term": "Restore wetland ecosystems for multi-generational resilience"},
        {"immediate": "Extinguish the wildfire threatening homes this week",
         "long_term": "Reforest degraded land for carbon sequestration over centuries"},
        {"immediate": "Issue an air quality advisory for today",
         "long_term": "Phase out fossil fuel combustion by mid-century"},
        {"immediate": "Distribute clean water to affected areas immediately",
         "long_term": "Design water infrastructure that serves communities for 50 years"},

        # Personal/relationships domain
        {"immediate": "Apologize right now to de-escalate the argument",
         "long_term": "Build communication patterns that strengthen the relationship for life"},
        {"immediate": "Help your child with tonight's homework",
         "long_term": "Cultivate their love of learning across their entire childhood"},
        {"immediate": "Comfort your friend who is upset today",
         "long_term": "Be the kind of friend who shows up consistently for decades"},
        {"immediate": "Fix the leaking pipe before dinner",
         "long_term": "Renovate the house to last another generation"},
        {"immediate": "Get through today's workload",
         "long_term": "Design a career path that brings fulfillment over a lifetime"},
    ]

    prompts, labels = [], []
    for p in pairs:
        prompts.append(p["immediate"])
        labels.append(0)
        prompts.append(p["long_term"])
        labels.append(1)

    return ShiftCondition(
        name="domain_shift",
        description="Professional domain-specific temporal framing (financial, medical, environmental, personal)",
        prompts=prompts,
        labels=np.array(labels),
    )


def generate_register_shift_data() -> ShiftCondition:
    """
    Register shift: same temporal concepts in formal vs casual language.
    Tests whether probes are robust to surface-level style changes.
    """
    pairs = [
        # Formal/academic register
        {"immediate": "The exigency of the present circumstance necessitates immediate intervention",
         "long_term": "A longitudinal strategic framework optimizing for multi-decadal outcomes is warranted"},
        {"immediate": "Pursuant to the urgency of current operational demands",
         "long_term": "In consideration of intergenerational equity and enduring institutional resilience"},
        {"immediate": "The acute phase of this crisis requires contemporaneous action",
         "long_term": "Sustainable development paradigms must transcend temporal myopia"},

        # Casual/conversational register
        {"immediate": "We gotta deal with this right now, no time to waste",
         "long_term": "Let's think about where we wanna be in like ten or twenty years"},
        {"immediate": "Just patch it up real quick and move on",
         "long_term": "Take our time and build something that'll actually last forever"},
        {"immediate": "Drop everything and handle this ASAP",
         "long_term": "Slow and steady wins the race, think about the long game"},

        # Technical/engineering register
        {"immediate": "Deploy hotfix to production, rollback if P0 regression detected within SLA",
         "long_term": "Architect for horizontal scalability with 10-year capacity planning"},
        {"immediate": "Patch the CVE before end of sprint",
         "long_term": "Design zero-trust infrastructure for next-generation threat models"},
        {"immediate": "Reduce p99 latency below threshold by EOD",
         "long_term": "Invest in platform reliability that compounds over product generations"},

        # Poetic/literary register
        {"immediate": "Seize the fleeting moment before it dissolves like morning dew",
         "long_term": "Plant seeds in the garden of tomorrow that bloom across centuries"},
        {"immediate": "The fire burns brightest in its first spark",
         "long_term": "The river carves the canyon grain by grain across eons"},
        {"immediate": "Strike while the iron glows with present heat",
         "long_term": "Weave threads into a tapestry that outlasts its maker"},

        # Bureaucratic register
        {"immediate": "Per directive 7.3.1, expedite processing of pending item by close of business",
         "long_term": "Pursuant to the 2030 Strategic Vision, establish enduring institutional frameworks"},
        {"immediate": "Action required: resolve outstanding ticket before daily standup",
         "long_term": "Roadmap item: build organizational capability for sustained multi-year growth"},
    ]

    prompts, labels = [], []
    for p in pairs:
        prompts.append(p["immediate"])
        labels.append(0)
        prompts.append(p["long_term"])
        labels.append(1)

    return ShiftCondition(
        name="register_shift",
        description="Same temporal concepts across formal, casual, technical, poetic, and bureaucratic registers",
        prompts=prompts,
        labels=np.array(labels),
    )


def generate_negation_data() -> ShiftCondition:
    """
    Negation: "not short-term" → long-term, "not long-term" → short-term.
    Tests whether probes understand compositional temporal semantics.
    """
    pairs = [
        # Negated immediate → should be long-term
        {"prompt": "This is not an immediate priority but a long-term investment",
         "label": 1},
        {"prompt": "We should not focus on short-term gains here",
         "label": 1},
        {"prompt": "Not a quick fix, but rather a lasting transformation",
         "label": 1},
        {"prompt": "This isn't urgent; it requires patient, sustained effort over years",
         "label": 1},
        {"prompt": "Don't rush this — the value comes from decades of compounding",
         "label": 1},
        {"prompt": "Not for today or tomorrow, but for the generations that follow",
         "label": 1},
        {"prompt": "Resist the temptation to act now; this needs a multi-year strategy",
         "label": 1},
        {"prompt": "The opposite of reactive: building proactive systems for the future",
         "label": 1},

        # Negated long-term → should be immediate
        {"prompt": "This is not a long-term strategy; we need action right now",
         "label": 0},
        {"prompt": "Don't think decades ahead here; focus on what we can do today",
         "label": 0},
        {"prompt": "Not for future generations, but for the people in this room right now",
         "label": 0},
        {"prompt": "We shouldn't plan for years; this needs to be resolved this afternoon",
         "label": 0},
        {"prompt": "Forget the strategic vision; handle the crisis in front of us",
         "label": 0},
        {"prompt": "Not sustainable and gradual, but urgent and immediate",
         "label": 0},
        {"prompt": "The long game doesn't apply here; we need results this week",
         "label": 0},
        {"prompt": "Don't build for permanence; ship the temporary workaround now",
         "label": 0},
    ]

    prompts = [p["prompt"] for p in pairs]
    labels = np.array([p["label"] for p in pairs])

    return ShiftCondition(
        name="negation",
        description="Negated temporal expressions testing compositional understanding",
        prompts=prompts,
        labels=labels,
    )


def generate_paraphrase_data() -> ShiftCondition:
    """
    Paraphrase: creative/unusual ways to express temporal concepts.
    No overlap with training vocabulary.
    """
    pairs = [
        # Immediate paraphrases
        {"prompt": "Before the clock strikes the next hour", "label": 0},
        {"prompt": "Between now and your next breath", "label": 0},
        {"prompt": "In the blink of an eye, before anything else", "label": 0},
        {"prompt": "While the coffee is still hot", "label": 0},
        {"prompt": "Before the ink dries on this page", "label": 0},
        {"prompt": "With the velocity of a reflex", "label": 0},
        {"prompt": "At the speed of necessity", "label": 0},
        {"prompt": "Without pausing to reconsider", "label": 0},

        # Long-term paraphrases
        {"prompt": "When our great-great-grandchildren read about this era", "label": 1},
        {"prompt": "On geological timescales of civilizational change", "label": 1},
        {"prompt": "Through the lifetimes of those yet to draw first breath", "label": 1},
        {"prompt": "Across the arc of epochs yet unnamed", "label": 1},
        {"prompt": "In the fullness of time that dwarfs a single human span", "label": 1},
        {"prompt": "As the seasons turn and the trees grow tall and die and grow again", "label": 1},
        {"prompt": "Over the slow accumulation of compounding wisdom", "label": 1},
        {"prompt": "Through patient iteration that spans the turning of many calendars", "label": 1},
    ]

    prompts = [p["prompt"] for p in pairs]
    labels = np.array([p["label"] for p in pairs])

    return ShiftCondition(
        name="paraphrase",
        description="Creative/unusual temporal expressions with no training vocabulary overlap",
        prompts=prompts,
        labels=labels,
    )


def generate_ambiguous_data() -> ShiftCondition:
    """
    Ambiguous: genuinely ambiguous temporal expressions.
    Used for calibration analysis — well-calibrated probes should show
    ~50% confidence on these.
    """
    pairs = [
        {"prompt": "Do what needs to be done", "label": 0},  # could be either
        {"prompt": "Consider the consequences", "label": 1},  # could be either
        {"prompt": "Make the right choice", "label": 0},
        {"prompt": "Think carefully about this", "label": 1},
        {"prompt": "Take appropriate action", "label": 0},
        {"prompt": "Respond proportionally", "label": 0},
        {"prompt": "Plan accordingly", "label": 1},
        {"prompt": "Proceed with caution", "label": 1},
        {"prompt": "Handle this wisely", "label": 0},
        {"prompt": "Act with intention", "label": 1},
    ]

    prompts = [p["prompt"] for p in pairs]
    labels = np.array([p["label"] for p in pairs])

    return ShiftCondition(
        name="ambiguous",
        description="Genuinely ambiguous temporal expressions (for calibration analysis)",
        prompts=prompts,
        labels=labels,
    )


def load_all_shift_conditions(data_dir: Path) -> dict[str, ShiftCondition]:
    """Load/generate all distribution shift conditions."""
    conditions = {}

    # Generated conditions
    conditions["domain_shift"] = generate_domain_shift_data()
    conditions["register_shift"] = generate_register_shift_data()
    conditions["negation"] = generate_negation_data()
    conditions["paraphrase"] = generate_paraphrase_data()
    conditions["ambiguous"] = generate_ambiguous_data()

    # File-based conditions
    implicit_path = data_dir / "raw" / "temporal_scope" / "temporal_scope_implicit.json"
    if implicit_path.exists():
        prompts, labels = load_caa_dataset(implicit_path)
        conditions["implicit_only"] = ShiftCondition(
            name="implicit_only",
            description="No explicit temporal keywords, only semantic/contextual cues",
            prompts=prompts,
            labels=labels,
        )

    clean_path = data_dir / "raw" / "temporal_scope" / "temporal_scope_clean.json"
    if clean_path.exists():
        prompts, labels = load_caa_dataset(clean_path)
        conditions["cross_dataset_clean"] = ShiftCondition(
            name="cross_dataset_clean",
            description="Full CAA-format dataset with pure temporal markers (no semantic cues)",
            prompts=prompts,
            labels=labels,
        )

    caa_path = data_dir / "raw" / "temporal_scope" / "temporal_scope_caa.json"
    if caa_path.exists():
        prompts, labels = load_caa_dataset(caa_path)
        conditions["cross_dataset_caa"] = ShiftCondition(
            name="cross_dataset_caa",
            description="Original CAA dataset (50 explicit temporal pairs)",
            prompts=prompts,
            labels=labels,
        )

    return conditions


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

def extract_activations(
    model: HookedTransformer,
    sae: SaeLensSAE,
    prompts: list[str],
    layer: int,
    batch_size: int = 16,
    device: str = "cpu",
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract raw residual stream activations and SAE-encoded latents.

    Returns:
        raw_activations: (n_prompts, d_model)
        sae_latents: (n_prompts, d_sae)
    """
    hook_name = f"blocks.{layer}.hook_resid_post"
    all_raw, all_sae = [], []

    iterator = range(0, len(prompts), batch_size)
    if verbose:
        iterator = tqdm(iterator, desc=f"Layer {layer}")

    for i in iterator:
        batch = prompts[i : i + batch_size]
        with torch.no_grad():
            _, cache = model.run_with_cache(
                batch,
                names_filter=[hook_name],
                stop_at_layer=layer + 1,
            )

        acts = cache[hook_name][:, -1, :]  # last token: (batch, d_model)
        sae_out = sae.encode(acts)  # (batch, d_sae)

        all_raw.append(acts.detach().float().cpu().numpy())
        all_sae.append(sae_out.detach().float().cpu().numpy())

    return np.concatenate(all_raw), np.concatenate(all_sae)


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

def select_discriminative_latents(
    sae_latents: np.ndarray,
    labels: np.ndarray,
    k: int = TOP_K_LATENTS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Select top-k SAE latents with highest absolute mean difference between classes.

    Returns:
        top_k_indices: (k,) indices of most discriminative latents
        signed_diff: (d_sae,) signed difference (mean_longterm - mean_immediate)
    """
    mean_imm = sae_latents[labels == 0].mean(axis=0)
    mean_lt = sae_latents[labels == 1].mean(axis=0)
    signed_diff = mean_lt - mean_imm
    abs_diff = np.abs(signed_diff)
    top_k = np.argsort(abs_diff)[-k:][::-1]
    return top_k, signed_diff


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.sum() / len(y_true) * abs(bin_acc - bin_conf)
    return ece


def evaluate_probe(
    probe: LogisticRegression,
    X: np.ndarray,
    y: np.ndarray,
) -> ProbeMetrics:
    """Evaluate a trained probe on a dataset."""
    y_pred = probe.predict(X)
    y_prob = probe.predict_proba(X)

    # Confidence = predicted probability of the predicted class
    confidence = np.max(y_prob, axis=1)

    # Probability of the positive class for ECE
    pos_prob = y_prob[:, 1]

    return ProbeMetrics(
        accuracy=accuracy_score(y, y_pred),
        precision=precision_score(y, y_pred, zero_division=0),
        recall=recall_score(y, y_pred, zero_division=0),
        f1=f1_score(y, y_pred, zero_division=0),
        log_loss_value=log_loss(y, y_prob, labels=[0, 1]),
        n_samples=len(y),
        mean_confidence=float(confidence.mean()),
        ece=compute_ece(y, pos_prob),
    )


def compute_feature_stability(
    original_sae_latents: np.ndarray,
    shifted_sae_latents: np.ndarray,
    top_k_indices: np.ndarray,
    k: int = TOP_K_LATENTS,
) -> FeatureStabilityMetrics:
    """
    Measure how SAE feature activations change between original and shifted data.
    """
    # Active latent sets (non-zero activations averaged across samples)
    orig_mean = np.abs(original_sae_latents).mean(axis=0)
    shift_mean = np.abs(shifted_sae_latents).mean(axis=0)

    orig_active = set(np.argsort(orig_mean)[-k:][::-1])
    shift_active = set(np.argsort(shift_mean)[-k:][::-1])

    # Jaccard similarity of active latent sets
    intersection = len(orig_active & shift_active)
    union = len(orig_active | shift_active)
    jaccard = intersection / union if union > 0 else 0.0

    # Top-k overlap: fraction of training top-k that remain in shifted top-k
    top_k_set = set(top_k_indices.tolist())
    shift_top_k = set(np.argsort(shift_mean)[-k:][::-1].tolist())
    top_k_overlap = len(top_k_set & shift_top_k) / len(top_k_set) if len(top_k_set) > 0 else 0.0

    # Activation magnitude ratio
    orig_mag = np.abs(original_sae_latents[:, top_k_indices]).mean()
    shift_mag = np.abs(shifted_sae_latents[:, top_k_indices]).mean()
    mag_ratio = shift_mag / orig_mag if orig_mag > 0 else 0.0

    # Cosine similarity of mean activation vectors
    orig_vec = original_sae_latents[:, top_k_indices].mean(axis=0)
    shift_vec = shifted_sae_latents[:, top_k_indices].mean(axis=0)
    cos_sim = np.dot(orig_vec, shift_vec) / (
        np.linalg.norm(orig_vec) * np.linalg.norm(shift_vec) + 1e-10
    )

    return FeatureStabilityMetrics(
        jaccard_similarity=float(jaccard),
        activation_magnitude_ratio=float(mag_ratio),
        top_k_overlap=float(top_k_overlap),
        cosine_similarity=float(cos_sim),
    )


# ---------------------------------------------------------------------------
# Main experiment pipeline
# ---------------------------------------------------------------------------

def _wandb_active() -> bool:
    """Check if W&B is available and a run is active."""
    return HAS_WANDB and wandb.run is not None


def _log_layer_metrics(layer: int, in_dist: dict, shift_results: dict):
    """Log all metrics for a single layer to W&B."""
    if not _wandb_active():
        return

    # In-distribution metrics
    for probe_type in ("sae_probe", "act_probe"):
        m = in_dist[probe_type]
        prefix = f"layer_{layer}/in_dist/{probe_type}"
        wandb.log({
            f"{prefix}/accuracy": m.accuracy,
            f"{prefix}/precision": m.precision,
            f"{prefix}/recall": m.recall,
            f"{prefix}/f1": m.f1,
            f"{prefix}/log_loss": m.log_loss_value,
            f"{prefix}/ece": m.ece,
            f"{prefix}/mean_confidence": m.mean_confidence,
        })

    # Shift condition metrics
    for cond_name, cond_data in shift_results.items():
        for probe_type in ("sae_probe", "act_probe"):
            m = cond_data[probe_type]
            prefix = f"layer_{layer}/{cond_name}/{probe_type}"
            wandb.log({
                f"{prefix}/accuracy": m.accuracy,
                f"{prefix}/f1": m.f1,
                f"{prefix}/log_loss": m.log_loss_value,
                f"{prefix}/ece": m.ece,
                f"{prefix}/mean_confidence": m.mean_confidence,
            })

            # Also log accuracy drop from in-distribution
            in_acc = in_dist[probe_type].accuracy
            wandb.log({
                f"{prefix}/accuracy_drop": in_acc - m.accuracy,
            })

        # Feature stability
        if "feature_stability" in cond_data:
            fs = cond_data["feature_stability"]
            prefix = f"layer_{layer}/{cond_name}/feature_stability"
            wandb.log({
                f"{prefix}/jaccard": fs.jaccard_similarity,
                f"{prefix}/top_k_overlap": fs.top_k_overlap,
                f"{prefix}/cosine_sim": fs.cosine_similarity,
                f"{prefix}/magnitude_ratio": fs.activation_magnitude_ratio,
            })

    # Summary table row for this layer
    table_data = []
    for cond_name, cond_data in shift_results.items():
        row = {
            "layer": layer,
            "condition": cond_name,
            "sae_accuracy": cond_data["sae_probe"].accuracy,
            "act_accuracy": cond_data["act_probe"].accuracy,
            "sae_accuracy_drop": in_dist["sae_probe"].accuracy - cond_data["sae_probe"].accuracy,
            "act_accuracy_drop": in_dist["act_probe"].accuracy - cond_data["act_probe"].accuracy,
        }
        if "feature_stability" in cond_data:
            fs = cond_data["feature_stability"]
            row.update({
                "top_k_overlap": fs.top_k_overlap,
                "cosine_sim": fs.cosine_similarity,
            })
        table_data.append(row)

    wandb.log({
        f"layer_{layer}/summary": wandb.Table(
            columns=list(table_data[0].keys()),
            data=[list(r.values()) for r in table_data],
        )
    })


def run_single_layer(
    model: HookedTransformer,
    sae: SaeLensSAE,
    layer: int,
    train_prompts: list[str],
    train_labels: np.ndarray,
    shift_conditions: dict[str, ShiftCondition],
    device: str = "cpu",
    batch_size: int = 16,
) -> LayerResult:
    """Run the full stability experiment for a single layer."""

    print(f"\n{'#' * 70}")
    print(f"# LAYER {layer}")
    print(f"{'#' * 70}")

    # 1. Extract in-distribution activations
    print("\n[1/4] Extracting in-distribution activations...")
    raw_acts, sae_latents = extract_activations(
        model, sae, train_prompts, layer, batch_size=batch_size, device=device,
    )

    # 2. Select discriminative latents & train probes
    print("[2/4] Training probes...")
    top_k_indices, signed_diff = select_discriminative_latents(sae_latents, train_labels)

    X_sae = sae_latents[:, top_k_indices]
    X_raw = raw_acts

    X_sae_train, X_sae_test, y_train, y_test = train_test_split(
        X_sae, train_labels, test_size=0.2, random_state=42, stratify=train_labels,
    )
    X_raw_train, X_raw_test, _, _ = train_test_split(
        X_raw, train_labels, test_size=0.2, random_state=42, stratify=train_labels,
    )

    sae_probe = LogisticRegression(max_iter=5000, random_state=42)
    sae_probe.fit(X_sae_train, y_train)

    act_probe = LogisticRegression(max_iter=5000, random_state=42)
    act_probe.fit(X_raw_train, y_train)

    # In-distribution metrics
    in_dist = {
        "sae_probe": evaluate_probe(sae_probe, X_sae_test, y_test),
        "act_probe": evaluate_probe(act_probe, X_raw_test, y_test),
    }
    print(f"  In-dist SAE probe acc: {in_dist['sae_probe'].accuracy:.3f}")
    print(f"  In-dist Act probe acc: {in_dist['act_probe'].accuracy:.3f}")

    # 3. Evaluate under each shift condition
    print("[3/4] Evaluating under distribution shift...")
    shift_results = {}

    for cond_name, cond in shift_conditions.items():
        print(f"\n  --- {cond_name} ({len(cond.prompts)} samples) ---")

        shift_raw, shift_sae = extract_activations(
            model, sae, cond.prompts, layer,
            batch_size=batch_size, device=device, verbose=False,
        )

        # SAE probe evaluation
        X_shift_sae = shift_sae[:, top_k_indices]
        sae_metrics = evaluate_probe(sae_probe, X_shift_sae, cond.labels)

        # Activation probe evaluation
        act_metrics = evaluate_probe(act_probe, shift_raw, cond.labels)

        # Feature stability
        stability = compute_feature_stability(sae_latents, shift_sae, top_k_indices)

        shift_results[cond_name] = {
            "sae_probe": sae_metrics,
            "act_probe": act_metrics,
            "feature_stability": stability,
        }

        print(f"    SAE acc: {sae_metrics.accuracy:.3f}  |  Act acc: {act_metrics.accuracy:.3f}")
        print(f"    Feature overlap: {stability.top_k_overlap:.3f}  |  Cosine sim: {stability.cosine_similarity:.3f}")

    # 4. Compile results & log to W&B
    print("[4/4] Compiling results...")
    _log_layer_metrics(layer, in_dist, shift_results)

    return LayerResult(
        layer=layer,
        top_k_indices=top_k_indices.tolist(),
        in_distribution=in_dist,
        shift_results=shift_results,
    )


def run_experiment(
    layers: list[int],
    device: str = "cpu",
    batch_size: int = 16,
    output_dir: Optional[Path] = None,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
) -> list[LayerResult]:
    """Run the full SAE feature stability experiment."""

    if output_dir is None:
        output_dir = RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialize W&B
    if HAS_WANDB and wandb_project:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run_name or f"sae-stability-{timestamp}",
            config={
                "model": MODEL_NAME,
                "sae_release": SAE_RELEASE,
                "top_k_latents": TOP_K_LATENTS,
                "layers": layers,
                "device": device,
                "batch_size": batch_size,
            },
            tags=["sae-feature-stability", "probe-degradation"],
        )
        print(f"W&B run: {wandb.run.url}")
    elif HAS_WANDB and wandb_project is None:
        print("W&B available but no --wandb-project specified. Logging disabled.")
    else:
        print("W&B not installed. Logging disabled.")

    print("=" * 70)
    print("SAE FEATURE STABILITY EXPERIMENT")
    print("=" * 70)
    print(f"Timestamp: {timestamp}")
    print(f"Model: {MODEL_NAME}")
    print(f"SAE Release: {SAE_RELEASE}")
    print(f"Layers: {layers}")
    print(f"Top-k latents: {TOP_K_LATENTS}")
    print(f"Device: {device}")

    # Load data
    print("\n--- Loading datasets ---")
    train_path = DATA_DIR / "raw" / "temporal_scope_pairs_minimal.json"
    train_prompts, train_labels = load_in_distribution_data(train_path)
    print(f"In-distribution: {len(train_prompts)} samples")

    shift_conditions = load_all_shift_conditions(DATA_DIR)
    for name, cond in shift_conditions.items():
        print(f"  {name}: {len(cond.prompts)} samples")

    # Load model
    print(f"\n--- Loading {MODEL_NAME} ---")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
    print(f"  Layers: {model.cfg.n_layers}, d_model: {model.cfg.d_model}")

    # Run per layer
    all_results = []
    for layer in layers:
        # Load SAE for this layer
        sae_id = f"layer_{layer}/width_16k/canonical"
        try:
            sae = SaeLensSAE.from_pretrained(
                release=SAE_RELEASE, sae_id=sae_id, device=device,
            )
        except Exception as e:
            print(f"Failed to load SAE for layer {layer}: {e}")
            continue

        result = run_single_layer(
            model, sae, layer,
            train_prompts, train_labels,
            shift_conditions,
            device=device,
            batch_size=batch_size,
        )
        all_results.append(result)

        # Free SAE memory
        del sae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save results
    save_results(all_results, shift_conditions, output_dir, timestamp)
    plot_degradation_curves(all_results, output_dir, timestamp)

    # Upload plots and results to W&B
    if _wandb_active():
        # Log plots as images
        for plot_name in ("accuracy_degradation", "feature_stability", "accuracy_vs_stability"):
            plot_path = output_dir / f"{plot_name}_{timestamp}.png"
            if plot_path.exists():
                wandb.log({f"plots/{plot_name}": wandb.Image(str(plot_path))})

        # Log results JSON as artifact
        artifact = wandb.Artifact(
            f"stability-results-{timestamp}",
            type="experiment-results",
            description="SAE feature stability experiment results",
        )
        results_path = output_dir / f"stability_results_{timestamp}.json"
        if results_path.exists():
            artifact.add_file(str(results_path))
        wandb.log_artifact(artifact)

        wandb.finish()

    return all_results


# ---------------------------------------------------------------------------
# Results serialization
# ---------------------------------------------------------------------------

def _metrics_to_dict(m) -> dict:
    """Convert ProbeMetrics or FeatureStabilityMetrics to dict."""
    if hasattr(m, "__dataclass_fields__"):
        return asdict(m)
    return m


def save_results(
    results: list[LayerResult],
    conditions: dict[str, ShiftCondition],
    output_dir: Path,
    timestamp: str,
):
    """Save experiment results to JSON."""
    serializable = {
        "config": {
            "model": MODEL_NAME,
            "sae_release": SAE_RELEASE,
            "top_k_latents": TOP_K_LATENTS,
            "timestamp": timestamp,
        },
        "conditions": {
            name: {
                "description": cond.description,
                "n_samples": len(cond.prompts),
            }
            for name, cond in conditions.items()
        },
        "layer_results": [],
    }

    for r in results:
        layer_data = {
            "layer": r.layer,
            "top_k_indices": r.top_k_indices,
            "in_distribution": {
                k: _metrics_to_dict(v) for k, v in r.in_distribution.items()
            },
            "shift_results": {},
        }
        for cond_name, cond_data in r.shift_results.items():
            layer_data["shift_results"][cond_name] = {
                k: _metrics_to_dict(v) for k, v in cond_data.items()
            }
        serializable["layer_results"].append(layer_data)

    out_path = output_dir / f"stability_results_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Also save as latest
    latest_path = output_dir / "stability_results_latest.json"
    with open(latest_path, "w") as f:
        json.dump(serializable, f, indent=2)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_degradation_curves(
    results: list[LayerResult],
    output_dir: Path,
    timestamp: str,
):
    """Generate the key plots for the paper."""

    if not results:
        print("No results to plot.")
        return

    # --- Plot 1: Accuracy degradation per condition ---
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5), squeeze=False)

    for idx, r in enumerate(results):
        ax = axes[0, idx]

        cond_names = list(r.shift_results.keys())
        sae_accs = [r.shift_results[c]["sae_probe"].accuracy for c in cond_names]
        act_accs = [r.shift_results[c]["act_probe"].accuracy for c in cond_names]

        x = np.arange(len(cond_names))
        width = 0.35

        ax.bar(x - width / 2, sae_accs, width, label="SAE Probe", color="steelblue", alpha=0.85)
        ax.bar(x + width / 2, act_accs, width, label="Act Probe", color="coral", alpha=0.85)

        # In-distribution reference lines
        ax.axhline(r.in_distribution["sae_probe"].accuracy, color="steelblue",
                    linestyle="--", alpha=0.5, label="SAE in-dist")
        ax.axhline(r.in_distribution["act_probe"].accuracy, color="coral",
                    linestyle="--", alpha=0.5, label="Act in-dist")
        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.3)

        ax.set_xlabel("Shift Condition")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Layer {r.layer}")
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("_", "\n") for c in cond_names],
                           fontsize=7, rotation=45, ha="right")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, loc="lower left")

    plt.suptitle("Probe Accuracy Under Distribution Shift", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / f"accuracy_degradation_{timestamp}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved accuracy degradation plot.")

    # --- Plot 2: Feature stability metrics ---
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5), squeeze=False)

    for idx, r in enumerate(results):
        ax = axes[0, idx]

        cond_names = [c for c in r.shift_results if c != "ambiguous"]
        if not cond_names:
            continue

        jaccard = [r.shift_results[c]["feature_stability"].jaccard_similarity for c in cond_names]
        overlap = [r.shift_results[c]["feature_stability"].top_k_overlap for c in cond_names]
        cosine = [r.shift_results[c]["feature_stability"].cosine_similarity for c in cond_names]

        x = np.arange(len(cond_names))
        width = 0.25

        ax.bar(x - width, jaccard, width, label="Jaccard sim", color="teal", alpha=0.85)
        ax.bar(x, overlap, width, label="Top-k overlap", color="darkorange", alpha=0.85)
        ax.bar(x + width, cosine, width, label="Cosine sim", color="mediumpurple", alpha=0.85)

        ax.set_xlabel("Shift Condition")
        ax.set_ylabel("Similarity Score")
        ax.set_title(f"Layer {r.layer} — SAE Feature Stability")
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("_", "\n") for c in cond_names],
                           fontsize=7, rotation=45, ha="right")
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=7)

    plt.suptitle("SAE Feature Activation Stability Under Shift", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / f"feature_stability_{timestamp}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved feature stability plot.")

    # --- Plot 3: Accuracy vs Feature Stability scatter ---
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))
    markers = ["o", "s", "^", "D"]

    for idx, r in enumerate(results):
        cond_names = [c for c in r.shift_results if c != "ambiguous"]
        for c in cond_names:
            acc = r.shift_results[c]["sae_probe"].accuracy
            overlap = r.shift_results[c]["feature_stability"].top_k_overlap
            ax.scatter(
                overlap, acc,
                c=[colors[idx]], marker=markers[idx % len(markers)],
                s=80, alpha=0.8, edgecolors="black", linewidths=0.5,
                label=f"L{r.layer}" if c == cond_names[0] else "",
            )
            ax.annotate(c.replace("_", "\n"), (overlap, acc),
                        fontsize=6, textcoords="offset points", xytext=(5, 5))

    ax.set_xlabel("Top-k Feature Overlap")
    ax.set_ylabel("SAE Probe Accuracy")
    ax.set_title("Accuracy vs Feature Stability")
    ax.legend()
    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(0.3, 1.05)
    plt.tight_layout()
    fig.savefig(output_dir / f"accuracy_vs_stability_{timestamp}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved accuracy vs stability scatter plot.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SAE Feature Stability Under Distribution Shift")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"],
                        help="Device for computation")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: single layer, smaller batch")
    parser.add_argument("--layers", nargs="+", type=int, default=None,
                        help="Specific layers to probe (default: 6 13 20 24)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for activation extraction")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="W&B project name (enables logging)")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="W&B entity/team (default: your default entity)")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="W&B run name (default: auto-generated)")

    args = parser.parse_args()

    if args.layers:
        layers = args.layers
    elif args.quick:
        layers = QUICK_LAYERS
    else:
        layers = DEFAULT_LAYERS

    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR

    batch_size = 4 if args.quick else args.batch_size

    results = run_experiment(
        layers=layers,
        device=args.device,
        batch_size=batch_size,
        output_dir=output_dir,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
    )

    # Final summary
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE — SUMMARY")
    print("=" * 70)

    for r in results:
        print(f"\nLayer {r.layer}:")
        print(f"  In-dist:  SAE={r.in_distribution['sae_probe'].accuracy:.3f}  "
              f"Act={r.in_distribution['act_probe'].accuracy:.3f}")
        for cond_name, cond_data in r.shift_results.items():
            sae_acc = cond_data["sae_probe"].accuracy
            act_acc = cond_data["act_probe"].accuracy
            overlap = cond_data["feature_stability"].top_k_overlap
            sae_drop = r.in_distribution["sae_probe"].accuracy - sae_acc
            act_drop = r.in_distribution["act_probe"].accuracy - act_acc
            print(f"  {cond_name:25s}  SAE={sae_acc:.3f} (Δ{-sae_drop:+.3f})  "
                  f"Act={act_acc:.3f} (Δ{-act_drop:+.3f})  "
                  f"overlap={overlap:.3f}")


if __name__ == "__main__":
    main()
