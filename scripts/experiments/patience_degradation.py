#!/usr/bin/env python3
"""
Patience & Compliance Degradation — RQ3

Research question: When models are given repetitive or extended task sequences,
does performance and compliance degrade in ways detectable in activations
BEFORE surfacing behaviorally?

Sub-questions:
  - Is compliance degradation correlated with loss of structural coherence?
  - Do individual neurons encode semantic features, or is the behavior
    correlative (distributed)?
  - Is patience degradation domain-general, or does it interact with task
    stakes?

This experiment:
  1. Feeds models repetitive/extended task sequences of varying length
  2. Measures behavioral quality (compliance, correctness, coherence) at
     each repetition
  3. Extracts activations at each repetition and tracks SAE feature drift
  4. Tests whether activation changes PRECEDE behavioral degradation
  5. Analyzes whether degradation is neuron-level or distributed

Methodology:
  - Generate repetitive task sequences at different repetition counts
    (1, 5, 10, 15, 20 reps)
  - For each repetition count, extract activations
  - Train probes on low-repetition data, test on high-repetition data
  - Measure: behavioral metrics, probe accuracy, feature drift,
    leading-indicator analysis

Based on Gemma-2-2B + Gemma Scope SAEs.

Usage:
    python scripts/experiments/patience_degradation.py --quick
    python scripts/experiments/patience_degradation.py --device cuda
    python scripts/experiments/patience_degradation.py --device cuda --wandb-project patience-deg

Author: Adrian Sadik
Date: 2026-03-11
"""

import argparse
import json
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path


class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types in JSON serialization."""
    def default(self, obj):
        import numpy as np
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
from typing import Optional

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# Optional W&B
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from transformer_lens import HookedTransformer
from sae_lens import SAE as SaeLensSAE


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TOP_K_LATENTS = 64

# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    "gemma-2-2b": {
        "sae_release": "gemma-scope-2b-pt-res-canonical",
        "sae_id_template": "layer_{layer}/width_16k/canonical",
        "default_layers": [6, 13, 20, 24],
        "quick_layers": [13],
        "n_layers": 26,
    },
    "gpt2": {
        "sae_release": "gpt2-small-res-jb",
        "sae_id_template": "blocks.{layer}.hook_resid_pre",
        "default_layers": [2, 5, 8, 10],
        "quick_layers": [5],
        "n_layers": 12,
    },
    "pythia-70m": {
        "sae_release": "pythia-70m-deduped-res-sm",
        "sae_id_template": "blocks.{layer}.hook_resid_post",
        "default_layers": [1, 2, 3, 4],
        "quick_layers": [3],
        "n_layers": 6,
    },
    "Qwen/Qwen2.5-3B-Instruct": {
        "sae_release": None,  # No pre-trained SAEs available
        "sae_id_template": None,
        "default_layers": [4, 12, 20, 28, 34],
        "quick_layers": [20],
        "n_layers": 36,
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "sae_release": None,  # No pre-trained SAEs available
        "sae_id_template": None,
        "default_layers": [4, 10, 16, 22, 28],
        "quick_layers": [16],
        "n_layers": 32,
    },
}

MODEL_NAME = "gemma-2-2b"
SAE_RELEASE = MODEL_CONFIGS[MODEL_NAME]["sae_release"]
DEFAULT_LAYERS = MODEL_CONFIGS[MODEL_NAME]["default_layers"]
QUICK_LAYERS = MODEL_CONFIGS[MODEL_NAME]["quick_layers"]

# Repetition counts to test
REPETITION_COUNTS = [1, 3, 5, 8, 12, 16, 20]
QUICK_REPETITION_COUNTS = [1, 5, 10]

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "patience_degradation"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RepetitionSample:
    """A single sample at a given repetition level."""
    prompt: str
    label: int  # 0=immediate, 1=long_term
    repetition_count: int
    task_domain: str
    task_stakes: str  # "low", "medium", "high"


@dataclass
class RepetitionLevel:
    """All samples at one repetition count."""
    repetition_count: int
    samples: list  # list[RepetitionSample]
    prompts: list  # list[str]
    labels: np.ndarray


@dataclass
class RepetitionMetrics:
    """Metrics at a single repetition level."""
    repetition_count: int
    n_samples: int
    # Probe performance (SAE probe is Optional — None when no SAE available)
    sae_probe_accuracy: Optional[float]
    act_probe_accuracy: float
    sae_probe_f1: Optional[float]
    act_probe_f1: float
    # Feature drift from rep=1 (based on SAE features if available, else raw acts)
    cosine_sim_to_baseline: float
    jaccard_to_baseline: float
    magnitude_ratio: float
    # Neuron-level analysis (computed on SAE features if available, else raw acts)
    top_neuron_concentration: float  # how much variance top neurons explain
    feature_entropy: float  # entropy of activation distribution (higher = more distributed)
    # Structural coherence (measured by probe confidence)
    mean_confidence: float
    confidence_std: float


@dataclass
class DegradationResult:
    """Results for one layer, one domain/stakes combination."""
    layer: int
    domain: str
    stakes: str
    repetition_metrics: list  # list[RepetitionMetrics]
    # Summary statistics
    degradation_onset_rep: int  # rep count where accuracy drops > 5%
    behavioral_precursor_gap: int  # how many reps early activation drift starts
    is_domain_general: bool  # does same pattern hold across domains
    neuron_vs_distributed: str  # "neuron-level" or "distributed"


# ---------------------------------------------------------------------------
# Repetitive task sequence generation
# ---------------------------------------------------------------------------

def _repeat_task_prefix(base_prompt: str, n_reps: int) -> str:
    """
    Build a prompt simulating n repetitions of a task.
    The model sees a sequence of repeated similar instructions, then the target.
    """
    if n_reps <= 1:
        return base_prompt

    # Create slight variations of the task to simulate realistic repetition
    filler_templates = [
        "Continue with the next item. ",
        "Proceed to the following task. ",
        "Moving on to another similar request. ",
        "Here is another one to process. ",
        "Next task in the sequence. ",
        "Please handle the following as well. ",
        "Another item requiring attention. ",
        "Continuing the sequence of tasks. ",
    ]

    prefix_parts = []
    for i in range(n_reps - 1):
        filler = filler_templates[i % len(filler_templates)]
        prefix_parts.append(filler)

    prefix = "".join(prefix_parts)
    return prefix + base_prompt


def generate_repetitive_sequences(
    repetition_counts: list[int],
) -> dict[str, dict[str, list[RepetitionLevel]]]:
    """
    Generate task sequences at varying repetition levels across domains and stakes.

    Returns:
        {domain: {stakes: [RepetitionLevel, ...]}}
    """
    # Base prompts organized by (domain, stakes, label)
    task_bank = {
        ("scheduling", "low"): {
            0: [  # immediate
                "Schedule the team standup for this morning.",
                "Move the 2pm meeting to 3pm today.",
                "Cancel the afternoon training session.",
                "Book a conference room for the next hour.",
                "Send a reminder about today's deadline.",
                "Reschedule the client call to right now.",
            ],
            1: [  # long_term
                "Plan the annual conference schedule for next year.",
                "Design a quarterly review cadence for the next five years.",
                "Establish a mentorship program schedule spanning two years.",
                "Create a five-year professional development roadmap.",
                "Set up a succession planning timeline over the decade.",
                "Build a multi-year sabbatical rotation schedule.",
            ],
        },
        ("scheduling", "high"): {
            0: [
                "Schedule the emergency board meeting for today.",
                "Arrange the regulatory hearing preparation for this afternoon.",
                "Set up the crisis response briefing right now.",
                "Coordinate the emergency evacuation drill immediately.",
                "Book the operating room for the urgent surgery today.",
                "Arrange immediate transport for the critical patient.",
            ],
            1: [
                "Plan the hospital expansion project timeline over ten years.",
                "Design the nuclear decommissioning schedule spanning decades.",
                "Establish climate adaptation milestones through 2050.",
                "Create the space mission timeline for the next twenty years.",
                "Set up pandemic preparedness review cycles over generations.",
                "Build infrastructure replacement schedules for the century.",
            ],
        },
        ("writing", "low"): {
            0: [
                "Write a quick email about the lunch order for today.",
                "Draft a short reply to confirm the meeting time.",
                "Compose a brief status update for this afternoon.",
                "Write a thank you note for yesterday's help.",
                "Create a short out-of-office message for today.",
                "Draft a quick agenda for the morning standup.",
            ],
            1: [
                "Write a comprehensive five-year strategic vision document.",
                "Draft a generational memoir spanning decades of experience.",
                "Compose a long-term research agenda for the next ten years.",
                "Write a multi-decade history of the organization.",
                "Create a forward-looking manifesto for the next century.",
                "Draft a philosophical treatise on human progress over millennia.",
            ],
        },
        ("writing", "high"): {
            0: [
                "Write the emergency press release about the data breach now.",
                "Draft the immediate recall notice for the defective product.",
                "Compose the urgent safety advisory for distribution today.",
                "Write the crisis communication to shareholders right away.",
                "Create the emergency patient notification immediately.",
                "Draft the urgent legal response due by end of day.",
            ],
            1: [
                "Write the constitutional amendment proposal for ratification over decades.",
                "Draft the multi-generational environmental remediation plan.",
                "Compose the century-long cultural preservation strategy.",
                "Write the long-term reparations framework spanning generations.",
                "Create the interstellar mission documentation for future centuries.",
                "Draft the species conservation strategy for the next millennium.",
            ],
        },
        ("analysis", "low"): {
            0: [
                "Analyze today's sales figures before the evening report.",
                "Review the morning's website traffic data quickly.",
                "Check the current inventory levels right now.",
                "Assess this afternoon's customer feedback scores.",
                "Evaluate the daily performance metrics before close.",
                "Examine the current server load immediately.",
            ],
            1: [
                "Analyze demographic trends over the next fifty years.",
                "Review the century-long evolution of global trade patterns.",
                "Study the multi-decade impact of automation on employment.",
                "Assess generational shifts in educational outcomes.",
                "Evaluate the long-term trajectory of renewable energy adoption.",
                "Examine climate data trends spanning the next century.",
            ],
        },
        ("analysis", "high"): {
            0: [
                "Analyze the acute toxicology results from the exposure incident now.",
                "Review the seismic data for immediate earthquake risk assessment.",
                "Assess the current outbreak data for emergency containment.",
                "Evaluate the dam structural integrity readings urgently.",
                "Examine the reactor temperature anomalies right away.",
                "Analyze the emergency financial exposure before markets open.",
            ],
            1: [
                "Analyze the multi-decade cancer cluster patterns for policy action.",
                "Review the century of nuclear waste containment effectiveness data.",
                "Assess the long-term genetic impact across affected generations.",
                "Evaluate the fifty-year infrastructure corrosion trajectory.",
                "Examine the multi-generational effects of environmental contamination.",
                "Analyze the decades-long trajectory of antibiotic resistance evolution.",
            ],
        },
    }

    results = {}

    for (domain, stakes), label_prompts in task_bank.items():
        if domain not in results:
            results[domain] = {}

        levels = []
        for rep_count in repetition_counts:
            samples = []
            prompts = []
            labels = []

            for label, base_prompts in label_prompts.items():
                for bp in base_prompts:
                    repeated_prompt = _repeat_task_prefix(bp, rep_count)
                    samples.append(RepetitionSample(
                        prompt=repeated_prompt, label=label,
                        repetition_count=rep_count,
                        task_domain=domain, task_stakes=stakes,
                    ))
                    prompts.append(repeated_prompt)
                    labels.append(label)

            levels.append(RepetitionLevel(
                repetition_count=rep_count,
                samples=samples,
                prompts=prompts,
                labels=np.array(labels),
            ))

        results[domain][stakes] = levels

    return results


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

def extract_activations(
    model: HookedTransformer,
    sae: Optional[SaeLensSAE],
    prompts: list[str],
    layer: int,
    batch_size: int = 16,
    device: str = "cpu",
    verbose: bool = False,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Extract raw residual stream activations and SAE-encoded latents.

    If sae is None, returns (raw_activations, None).
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
                batch, names_filter=[hook_name], stop_at_layer=layer + 1,
            )
        acts = cache[hook_name][:, -1, :]
        all_raw.append(acts.detach().float().cpu().numpy())
        if sae is not None:
            sae_out = sae.encode(acts)
            all_sae.append(sae_out.detach().float().cpu().numpy())

    raw = np.concatenate(all_raw)
    sae_feats = np.concatenate(all_sae) if all_sae else None
    return raw, sae_feats


# ---------------------------------------------------------------------------
# Neuron-level analysis
# ---------------------------------------------------------------------------

def compute_neuron_concentration(sae_latents: np.ndarray, top_k: int = 10) -> float:
    """
    Measure how concentrated the temporal signal is in top neurons.
    Returns fraction of total variance explained by top-k neurons.
    Higher = more neuron-level; Lower = more distributed.
    """
    variances = np.var(sae_latents, axis=0)
    total_var = variances.sum()
    if total_var < 1e-10:
        return 0.0
    top_k_var = np.sort(variances)[-top_k:].sum()
    return float(top_k_var / total_var)


def compute_feature_entropy(sae_latents: np.ndarray) -> float:
    """
    Shannon entropy of the activation magnitude distribution.
    Higher entropy = more distributed representation.
    """
    magnitudes = np.abs(sae_latents).mean(axis=0)
    total = magnitudes.sum()
    if total < 1e-10:
        return 0.0
    probs = magnitudes / total
    probs = probs[probs > 1e-10]  # avoid log(0)
    return float(-np.sum(probs * np.log2(probs)))


def classify_representation(
    concentration_values: list[float],
    entropy_values: list[float],
) -> str:
    """
    Classify whether the temporal representation is neuron-level or distributed.
    """
    mean_concentration = np.mean(concentration_values)
    # If top 10 neurons explain > 30% of variance, call it neuron-level
    if mean_concentration > 0.3:
        return "neuron-level"
    elif mean_concentration > 0.15:
        return "mixed"
    else:
        return "distributed"


# ---------------------------------------------------------------------------
# Leading indicator analysis
# ---------------------------------------------------------------------------

def find_degradation_onset(
    repetition_counts: list[int],
    accuracies: list[float],
    threshold: float = 0.05,
) -> int:
    """Find the repetition count where accuracy drops by more than threshold."""
    if len(accuracies) < 2:
        return repetition_counts[-1]

    baseline_acc = accuracies[0]
    for i, (rep, acc) in enumerate(zip(repetition_counts, accuracies)):
        if baseline_acc - acc > threshold:
            return rep
    return repetition_counts[-1]  # no degradation found


def find_activation_drift_onset(
    repetition_counts: list[int],
    cosine_sims: list[float],
    threshold: float = 0.05,
) -> int:
    """Find the repetition count where activation drift exceeds threshold."""
    if len(cosine_sims) < 2:
        return repetition_counts[-1]

    for rep, cos_sim in zip(repetition_counts, cosine_sims):
        if 1.0 - cos_sim > threshold:
            return rep
    return repetition_counts[-1]


def compute_precursor_gap(
    behavioral_onset: int,
    activation_onset: int,
) -> int:
    """
    How many repetitions earlier does activation drift start compared
    to behavioral degradation. Positive = activations change first (good!).
    """
    return behavioral_onset - activation_onset


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def _wandb_active() -> bool:
    return HAS_WANDB and wandb.run is not None


def run_single_layer_degradation(
    model: HookedTransformer,
    sae: Optional[SaeLensSAE],
    layer: int,
    all_sequences: dict[str, dict[str, list[RepetitionLevel]]],
    device: str = "cpu",
    batch_size: int = 16,
) -> list[DegradationResult]:
    """Run patience degradation experiment for one layer.

    When sae is None (e.g. for instruction-tuned models without pre-trained
    SAEs), the experiment runs with activation probes only.  Drift metrics
    are computed on raw activations and SAE-specific fields are set to None.
    """

    has_sae = sae is not None

    print(f"\n{'#' * 70}")
    print(f"# LAYER {layer} — Patience Degradation {'(activation-only)' if not has_sae else ''}")
    print(f"{'#' * 70}")

    layer_results = []

    for domain, stakes_levels in all_sequences.items():
        for stakes, rep_levels in stakes_levels.items():
            print(f"\n  === {domain} / {stakes} stakes ===")

            # Train probe on lowest repetition level
            baseline = rep_levels[0]
            print(f"  [1/3] Training probe on rep={baseline.repetition_count}...")

            raw_base, sae_base = extract_activations(
                model, sae, baseline.prompts, layer,
                batch_size=batch_size, device=device,
            )

            # --- SAE probe (only when SAE is available) ---
            sae_probe = None
            top_k_indices = None
            base_mean_acts_sae = None
            base_active_sae = None

            if has_sae and sae_base is not None:
                # Select discriminative latents
                mean_imm = sae_base[baseline.labels == 0].mean(axis=0)
                mean_lt = sae_base[baseline.labels == 1].mean(axis=0)
                abs_diff = np.abs(mean_lt - mean_imm)
                top_k_indices = np.argsort(abs_diff)[-TOP_K_LATENTS:][::-1]

                X_sae_base = sae_base[:, top_k_indices]
                sae_probe = LogisticRegression(max_iter=5000, random_state=42)
                sae_probe.fit(X_sae_base, baseline.labels)

                base_mean_acts_sae = sae_base[:, top_k_indices].mean(axis=0)
                base_active_sae = set(np.argsort(np.abs(sae_base).mean(axis=0))[-TOP_K_LATENTS:][::-1].tolist())

            # --- Activation probe (always) ---
            act_probe = LogisticRegression(max_iter=5000, random_state=42)
            act_probe.fit(raw_base, baseline.labels)

            # Baseline drift anchors on raw activations (used when SAE absent)
            base_mean_raw = raw_base.mean(axis=0)
            # For Jaccard on raw acts, pick top-k dimensions by variance
            raw_var = np.var(raw_base, axis=0)
            base_active_raw = set(np.argsort(raw_var)[-TOP_K_LATENTS:][::-1].tolist())

            # Evaluate at each repetition level
            print(f"  [2/3] Evaluating across {len(rep_levels)} repetition levels...")
            rep_metrics = []
            concentration_values = []
            entropy_values = []

            for level in rep_levels:
                if level.repetition_count == baseline.repetition_count:
                    raw_r, sae_r = raw_base, sae_base
                else:
                    raw_r, sae_r = extract_activations(
                        model, sae, level.prompts, layer,
                        batch_size=batch_size, device=device,
                    )

                # --- SAE probe metrics ---
                sae_acc = None
                sae_f1_val = None

                if sae_probe is not None and sae_r is not None:
                    X_sae_r = sae_r[:, top_k_indices]
                    sae_preds = sae_probe.predict(X_sae_r)
                    sae_acc = float(accuracy_score(level.labels, sae_preds))
                    sae_f1_val = float(f1_score(level.labels, sae_preds, zero_division=0))

                # --- Activation probe metrics (always) ---
                act_preds = act_probe.predict(raw_r)
                act_probs = act_probe.predict_proba(raw_r)
                act_acc = float(accuracy_score(level.labels, act_preds))
                act_f1_val = float(f1_score(level.labels, act_preds, zero_division=0))

                # Confidence (from SAE probe if available, else activation probe)
                if sae_probe is not None and sae_r is not None:
                    sae_probs = sae_probe.predict_proba(sae_r[:, top_k_indices])
                    confidence = np.max(sae_probs, axis=1)
                else:
                    confidence = np.max(act_probs, axis=1)

                # --- Feature drift from baseline ---
                if has_sae and sae_r is not None:
                    # Drift on SAE features
                    curr_mean_acts = sae_r[:, top_k_indices].mean(axis=0)
                    cos_sim = float(np.dot(curr_mean_acts, base_mean_acts_sae) / (
                        np.linalg.norm(curr_mean_acts) * np.linalg.norm(base_mean_acts_sae) + 1e-10
                    ))
                    curr_active = set(np.argsort(np.abs(sae_r).mean(axis=0))[-TOP_K_LATENTS:][::-1].tolist())
                    intersection = len(base_active_sae & curr_active)
                    union = len(base_active_sae | curr_active)
                    jaccard = intersection / union if union > 0 else 0.0

                    base_mag = np.abs(sae_base[:, top_k_indices]).mean()
                    curr_mag = np.abs(sae_r[:, top_k_indices]).mean()
                    mag_ratio = float(curr_mag / base_mag) if base_mag > 0 else 0.0

                    # Neuron-level analysis on SAE features
                    conc = compute_neuron_concentration(sae_r)
                    ent = compute_feature_entropy(sae_r)
                else:
                    # Drift on raw activations
                    curr_mean_raw = raw_r.mean(axis=0)
                    cos_sim = float(np.dot(curr_mean_raw, base_mean_raw) / (
                        np.linalg.norm(curr_mean_raw) * np.linalg.norm(base_mean_raw) + 1e-10
                    ))
                    curr_var = np.var(raw_r, axis=0)
                    curr_active_raw = set(np.argsort(curr_var)[-TOP_K_LATENTS:][::-1].tolist())
                    intersection = len(base_active_raw & curr_active_raw)
                    union = len(base_active_raw | curr_active_raw)
                    jaccard = intersection / union if union > 0 else 0.0

                    base_mag = np.abs(raw_base).mean()
                    curr_mag = np.abs(raw_r).mean()
                    mag_ratio = float(curr_mag / base_mag) if base_mag > 0 else 0.0

                    # Neuron-level analysis on raw activations
                    conc = compute_neuron_concentration(raw_r)
                    ent = compute_feature_entropy(raw_r)

                concentration_values.append(conc)
                entropy_values.append(ent)

                rm = RepetitionMetrics(
                    repetition_count=level.repetition_count,
                    n_samples=len(level.prompts),
                    sae_probe_accuracy=sae_acc,
                    act_probe_accuracy=act_acc,
                    sae_probe_f1=sae_f1_val,
                    act_probe_f1=act_f1_val,
                    cosine_sim_to_baseline=cos_sim,
                    jaccard_to_baseline=jaccard,
                    magnitude_ratio=mag_ratio,
                    top_neuron_concentration=conc,
                    feature_entropy=ent,
                    mean_confidence=float(confidence.mean()),
                    confidence_std=float(confidence.std()),
                )
                rep_metrics.append(rm)

                sae_str = f"SAE={sae_acc:.3f} " if sae_acc is not None else ""
                print(f"    Rep {level.repetition_count:2d}: "
                      f"{sae_str}Act={act_acc:.3f} "
                      f"cos={cos_sim:.3f} conc={conc:.3f}")

            # Summary analysis
            print(f"  [3/3] Analyzing degradation patterns...")
            reps = [m.repetition_count for m in rep_metrics]
            # Use SAE probe accuracy for behavioral onset when available, else act probe
            probe_accs = [m.sae_probe_accuracy if m.sae_probe_accuracy is not None
                          else m.act_probe_accuracy for m in rep_metrics]
            cos_sims = [m.cosine_sim_to_baseline for m in rep_metrics]

            behav_onset = find_degradation_onset(reps, probe_accs)
            act_onset = find_activation_drift_onset(reps, cos_sims)
            precursor_gap = compute_precursor_gap(behav_onset, act_onset)
            representation_type = classify_representation(concentration_values, entropy_values)

            result = DegradationResult(
                layer=layer,
                domain=domain,
                stakes=stakes,
                repetition_metrics=rep_metrics,
                degradation_onset_rep=behav_onset,
                behavioral_precursor_gap=precursor_gap,
                is_domain_general=True,  # updated in cross-domain analysis
                neuron_vs_distributed=representation_type,
            )
            layer_results.append(result)

            print(f"    Behavioral degradation onset: rep={behav_onset}")
            print(f"    Activation drift onset: rep={act_onset}")
            print(f"    Precursor gap: {precursor_gap} reps (positive = activations change first)")
            print(f"    Representation type: {representation_type}")

            # W&B
            if _wandb_active():
                for rm in rep_metrics:
                    prefix = f"layer_{layer}/{domain}_{stakes}"
                    log_dict = {
                        f"{prefix}/rep_{rm.repetition_count}/act_accuracy": rm.act_probe_accuracy,
                        f"{prefix}/rep_{rm.repetition_count}/cosine_baseline": rm.cosine_sim_to_baseline,
                        f"{prefix}/rep_{rm.repetition_count}/concentration": rm.top_neuron_concentration,
                        f"{prefix}/rep_{rm.repetition_count}/confidence": rm.mean_confidence,
                    }
                    if rm.sae_probe_accuracy is not None:
                        log_dict[f"{prefix}/rep_{rm.repetition_count}/sae_accuracy"] = rm.sae_probe_accuracy
                    wandb.log(log_dict)
                wandb.log({
                    f"layer_{layer}/{domain}_{stakes}/degradation_onset": behav_onset,
                    f"layer_{layer}/{domain}_{stakes}/precursor_gap": precursor_gap,
                    f"layer_{layer}/{domain}_{stakes}/representation_type": representation_type,
                })

    # Cross-domain analysis
    _check_domain_generality(layer_results)

    return layer_results


def _check_domain_generality(results: list[DegradationResult]):
    """Check if degradation patterns are consistent across domains."""
    onset_by_domain = {}
    for r in results:
        onset_by_domain.setdefault(r.domain, []).append(r.degradation_onset_rep)

    if len(onset_by_domain) < 2:
        return

    # Check if onset reps are similar across domains (within 3 reps)
    all_means = [np.mean(v) for v in onset_by_domain.values()]
    is_general = (max(all_means) - min(all_means)) <= 3

    for r in results:
        r.is_domain_general = is_general


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_degradation_results(
    all_results: list[DegradationResult],
    output_dir: Path,
    timestamp: str,
):
    """Generate degradation analysis plots."""

    # 1. Accuracy vs repetition count (by domain/stakes)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Patience Degradation: Probe Accuracy vs. Repetition Count", fontsize=14)

    color_map = {"low": "green", "high": "red"}

    # Plot SAE or activation probe accuracy depending on availability
    any_has_sae = any(m.sae_probe_accuracy is not None
                      for r in all_results for m in r.repetition_metrics)
    for r in all_results:
        reps = [m.repetition_count for m in r.repetition_metrics]
        if any_has_sae:
            accs = [m.sae_probe_accuracy if m.sae_probe_accuracy is not None
                    else m.act_probe_accuracy for m in r.repetition_metrics]
        else:
            accs = [m.act_probe_accuracy for m in r.repetition_metrics]
        color = color_map.get(r.stakes, "blue")

        axes[0, 0].plot(reps, accs, "o-", color=color,
                        label=f"{r.domain}/{r.stakes} L{r.layer}", alpha=0.7)
    axes[0, 0].set_xlabel("Repetition Count")
    axes[0, 0].set_ylabel("SAE Probe Accuracy" if any_has_sae else "Activation Probe Accuracy")
    axes[0, 0].set_title("SAE Probe Degradation" if any_has_sae else "Activation Probe Degradation")
    axes[0, 0].legend(fontsize=6, ncol=2)
    axes[0, 0].grid(True, alpha=0.3)

    for r in all_results:
        reps = [m.repetition_count for m in r.repetition_metrics]
        cos_sims = [m.cosine_sim_to_baseline for m in r.repetition_metrics]
        color = color_map.get(r.stakes, "blue")
        axes[0, 1].plot(reps, cos_sims, "s-", color=color,
                        label=f"{r.domain}/{r.stakes} L{r.layer}", alpha=0.7)
    axes[0, 1].set_xlabel("Repetition Count")
    axes[0, 1].set_ylabel("Cosine Sim to Baseline")
    axes[0, 1].set_title("Activation Drift vs. Repetition")
    axes[0, 1].legend(fontsize=6, ncol=2)
    axes[0, 1].grid(True, alpha=0.3)

    # Confidence degradation
    for r in all_results:
        reps = [m.repetition_count for m in r.repetition_metrics]
        confs = [m.mean_confidence for m in r.repetition_metrics]
        color = color_map.get(r.stakes, "blue")
        axes[1, 0].plot(reps, confs, "^-", color=color,
                        label=f"{r.domain}/{r.stakes} L{r.layer}", alpha=0.7)
    axes[1, 0].set_xlabel("Repetition Count")
    axes[1, 0].set_ylabel("Mean Probe Confidence")
    axes[1, 0].set_title("Confidence Degradation (Structural Coherence)")
    axes[1, 0].legend(fontsize=6, ncol=2)
    axes[1, 0].grid(True, alpha=0.3)

    # Neuron concentration
    for r in all_results:
        reps = [m.repetition_count for m in r.repetition_metrics]
        concs = [m.top_neuron_concentration for m in r.repetition_metrics]
        color = color_map.get(r.stakes, "blue")
        axes[1, 1].plot(reps, concs, "d-", color=color,
                        label=f"{r.domain}/{r.stakes} L{r.layer}", alpha=0.7)
    axes[1, 1].set_xlabel("Repetition Count")
    axes[1, 1].set_ylabel("Top-10 Neuron Concentration")
    axes[1, 1].set_title("Neuron vs. Distributed Encoding")
    axes[1, 1].legend(fontsize=6, ncol=2)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / f"patience_degradation_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {plot_path}")

    if _wandb_active():
        wandb.log({"plots/patience_degradation": wandb.Image(str(plot_path))})

    # 2. Precursor gap analysis
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = []
    gaps = []
    colors = []
    for r in all_results:
        labels.append(f"L{r.layer}\n{r.domain}/{r.stakes}")
        gaps.append(r.behavioral_precursor_gap)
        colors.append("steelblue" if r.behavioral_precursor_gap > 0 else "coral")

    bars = ax.bar(range(len(labels)), gaps, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Precursor Gap (reps)")
    ax.set_title("Activation Drift as Leading Indicator of Behavioral Degradation\n"
                  "(Positive = activations change before behavior)")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    plot_path = output_dir / f"precursor_gap_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {plot_path}")

    if _wandb_active():
        wandb.log({"plots/precursor_gap": wandb.Image(str(plot_path))})

    # 3. Stakes comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Does Task Stakes Affect Degradation Rate?", fontsize=14)

    low_results = [r for r in all_results if r.stakes == "low"]
    high_results = [r for r in all_results if r.stakes == "high"]

    for r in low_results:
        reps = [m.repetition_count for m in r.repetition_metrics]
        sae_accs = [m.sae_probe_accuracy for m in r.repetition_metrics]
        axes[0].plot(reps, sae_accs, "o-", label=f"{r.domain} L{r.layer}", alpha=0.7)
    axes[0].set_title("Low Stakes")
    axes[0].set_xlabel("Repetition Count")
    axes[0].set_ylabel("SAE Probe Accuracy")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    for r in high_results:
        reps = [m.repetition_count for m in r.repetition_metrics]
        sae_accs = [m.sae_probe_accuracy for m in r.repetition_metrics]
        axes[1].plot(reps, sae_accs, "o-", label=f"{r.domain} L{r.layer}", alpha=0.7)
    axes[1].set_title("High Stakes")
    axes[1].set_xlabel("Repetition Count")
    axes[1].set_ylabel("SAE Probe Accuracy")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / f"stakes_comparison_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {plot_path}")

    if _wandb_active():
        wandb.log({"plots/stakes_comparison": wandb.Image(str(plot_path))})


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def save_results(results: list[DegradationResult], output_dir: Path, timestamp: str):
    serializable = {
        "config": {
            "model": MODEL_NAME,
            "sae_release": SAE_RELEASE,
            "top_k_latents": TOP_K_LATENTS,
            "timestamp": timestamp,
            "experiment": "patience_degradation",
        },
        "results": [],
    }

    for r in results:
        serializable["results"].append({
            "layer": r.layer,
            "domain": r.domain,
            "stakes": r.stakes,
            "degradation_onset_rep": r.degradation_onset_rep,
            "behavioral_precursor_gap": r.behavioral_precursor_gap,
            "is_domain_general": r.is_domain_general,
            "neuron_vs_distributed": r.neuron_vs_distributed,
            "repetition_metrics": [asdict(m) for m in r.repetition_metrics],
        })

    out_path = output_dir / f"patience_results_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {out_path}")

    latest_path = output_dir / "patience_results_latest.json"
    with open(latest_path, "w") as f:
        json.dump(serializable, f, indent=2, cls=NumpyEncoder)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_experiment(
    layers: list[int],
    device: str = "cpu",
    batch_size: int = 16,
    output_dir: Optional[Path] = None,
    repetition_counts: Optional[list[int]] = None,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
) -> list[DegradationResult]:
    """Run the full patience degradation experiment."""

    if output_dir is None:
        output_dir = RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if repetition_counts is None:
        repetition_counts = REPETITION_COUNTS

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if HAS_WANDB and wandb_project:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run_name or f"patience-deg-{timestamp}",
            config={
                "model": MODEL_NAME,
                "sae_release": SAE_RELEASE,
                "top_k_latents": TOP_K_LATENTS,
                "layers": layers,
                "repetition_counts": repetition_counts,
                "device": device,
                "batch_size": batch_size,
                "experiment": "patience_degradation",
            },
            tags=["patience-degradation", "compliance", "rq3"],
        )
        print(f"W&B run: {wandb.run.url}")

    print("=" * 70)
    print("PATIENCE & COMPLIANCE DEGRADATION EXPERIMENT")
    print("=" * 70)
    print(f"Timestamp: {timestamp}")
    print(f"Model: {MODEL_NAME}")
    print(f"Layers: {layers}")
    print(f"Repetition counts: {repetition_counts}")
    print(f"Device: {device}")

    # Generate sequences
    print("\n--- Generating repetitive task sequences ---")
    all_sequences = generate_repetitive_sequences(repetition_counts)
    for domain, stakes_dict in all_sequences.items():
        for stakes, levels in stakes_dict.items():
            total = sum(len(lev.prompts) for lev in levels)
            print(f"  {domain}/{stakes}: {len(levels)} rep levels, {total} total prompts")

    # Load model
    print(f"\n--- Loading {MODEL_NAME} ---")
    load_kwargs: dict = {"device": device}
    # Use float16 for larger models to save VRAM
    if MODEL_CONFIGS[MODEL_NAME].get("sae_release") is None:
        load_kwargs["dtype"] = torch.float16
        print("  Using float16 (instruction-tuned model, no SAE)")
    model = HookedTransformer.from_pretrained(MODEL_NAME, **load_kwargs)
    print(f"  Layers: {model.cfg.n_layers}, d_model: {model.cfg.d_model}")

    # Run per layer
    all_results = []
    for layer in layers:
        sae = None
        if SAE_RELEASE is not None:
            sae_id = MODEL_CONFIGS[MODEL_NAME]["sae_id_template"].format(layer=layer)
            try:
                sae = SaeLensSAE.from_pretrained(
                    release=SAE_RELEASE, sae_id=sae_id, device=device,
                )
            except Exception as e:
                print(f"Failed to load SAE for layer {layer}: {e}")
                print("  Continuing with activation-only probes...")
        else:
            print(f"  No SAE available for {MODEL_NAME} — running activation-only probes")

        layer_results = run_single_layer_degradation(
            model, sae, layer, all_sequences,
            device=device, batch_size=batch_size,
        )
        all_results.extend(layer_results)

        if sae is not None:
            del sae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save and plot
    save_results(all_results, output_dir, timestamp)
    plot_degradation_results(all_results, output_dir, timestamp)

    if _wandb_active():
        artifact = wandb.Artifact(
            f"patience-results-{timestamp}",
            type="experiment-results",
            description="Patience degradation experiment results",
        )
        results_path = output_dir / f"patience_results_{timestamp}.json"
        if results_path.exists():
            artifact.add_file(str(results_path))
        wandb.log_artifact(artifact)
        wandb.finish()

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Patience & Compliance Degradation")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--model", type=str, default="gemma-2-2b",
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Model to use (default: gemma-2-2b)")
    args = parser.parse_args()

    # Apply model config
    global MODEL_NAME, SAE_RELEASE, DEFAULT_LAYERS, QUICK_LAYERS
    MODEL_NAME = args.model
    cfg = MODEL_CONFIGS[MODEL_NAME]
    SAE_RELEASE = cfg["sae_release"]
    DEFAULT_LAYERS = cfg["default_layers"]
    QUICK_LAYERS = cfg["quick_layers"]

    layers = args.layers or (QUICK_LAYERS if args.quick else DEFAULT_LAYERS)
    rep_counts = QUICK_REPETITION_COUNTS if args.quick else REPETITION_COUNTS
    output_dir = Path(args.output_dir) if args.output_dir else None

    run_experiment(
        layers=layers,
        device=args.device,
        batch_size=args.batch_size,
        output_dir=output_dir,
        repetition_counts=rep_counts,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
    )


if __name__ == "__main__":
    main()
