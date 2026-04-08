#!/usr/bin/env python3
"""
Sequential Activation Tracking — RQ1 Sub-question

Research question: Do activation signatures shift predictably as models
progress through extended task sequences?

This experiment tracks how SAE features and probe accuracy evolve across
multi-step task sequences. Instead of testing probes on different data
distributions (as in sae_feature_stability.py), we test them at different
*positions* within a sequential task to measure temporal drift.

Methodology:
  - Construct multi-step task sequences (e.g., multi-step reasoning chains,
    sequential instructions, progressive narratives)
  - Extract activations at each step within a sequence
  - Train probes on early-step activations, evaluate at later steps
  - Measure: probe accuracy vs. step position, feature overlap vs. step
    distance, activation trajectory smoothness, drift predictability

Key metrics:
  - Probe accuracy as a function of sequence position
  - Feature overlap (Jaccard, cosine) between step N and step 0
  - Activation trajectory curvature (is drift monotonic/predictable?)
  - Drift velocity (rate of change in activation space per step)

Based on Gemma-2-2B + Gemma Scope SAEs.

Usage:
    python scripts/experiments/sequential_activation_tracking.py --quick
    python scripts/experiments/sequential_activation_tracking.py --device cuda
    python scripts/experiments/sequential_activation_tracking.py --device cuda --wandb-project seq-tracking

Author: Adrian Sadik
Date: 2026-03-11
"""

import argparse
import json
import random
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
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
}

MODEL_NAME = "gemma-2-2b"
SAE_RELEASE = MODEL_CONFIGS[MODEL_NAME]["sae_release"]
DEFAULT_LAYERS = MODEL_CONFIGS[MODEL_NAME]["default_layers"]
QUICK_LAYERS = MODEL_CONFIGS[MODEL_NAME]["quick_layers"]

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "sequential_tracking"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SequenceStep:
    """A single step within a multi-step sequence."""
    step_index: int
    prompt: str
    label: int  # 0=immediate, 1=long_term
    sequence_id: str
    total_steps: int


@dataclass
class TaskSequence:
    """A full multi-step task sequence."""
    sequence_id: str
    name: str
    description: str
    steps: list  # list[SequenceStep]
    label: int  # overall temporal label


@dataclass
class StepMetrics:
    """Metrics at a single sequence position."""
    step_index: int
    n_samples: int
    sae_probe_accuracy: float
    act_probe_accuracy: float
    sae_probe_f1: float
    act_probe_f1: float
    # Feature drift from step 0
    cosine_sim_to_start: float
    jaccard_to_start: float
    activation_magnitude_ratio: float
    # Feature drift from previous step
    cosine_sim_to_prev: float
    drift_velocity: float  # L2 distance from previous step


@dataclass
class SequenceResult:
    """Results for one layer across all sequence positions."""
    layer: int
    sequence_type: str
    step_metrics: list  # list[StepMetrics]
    drift_is_monotonic: bool
    drift_predictability: float  # R^2 of linear fit to drift curve
    total_drift: float  # cosine distance from step 0 to last step


# ---------------------------------------------------------------------------
# Multi-step task sequence generation
# ---------------------------------------------------------------------------

def generate_planning_sequences(n_per_class: int = 10) -> list[TaskSequence]:
    """
    Multi-step planning sequences where each step builds on the previous.
    Track whether temporal representations shift as the model processes
    progressively longer-horizon plans.
    """
    sequences = []

    # Immediate planning sequences (short-horizon, step-by-step)
    immediate_templates = [
        [
            "I need to prepare dinner tonight.",
            "First, check what ingredients are in the fridge.",
            "Then go to the store for anything missing.",
            "Start cooking the main dish.",
            "Set the table while it simmers.",
            "Serve the meal before it gets cold.",
            "Clean up the kitchen right after eating.",
            "Put leftovers in the fridge tonight.",
        ],
        [
            "I have a meeting in one hour.",
            "Review the agenda for today's meeting.",
            "Print the quarterly report now.",
            "Grab a coffee before heading over.",
            "Set up the conference room projector.",
            "Take notes during the discussion.",
            "Send the summary email right after.",
            "Update the task board before end of day.",
        ],
        [
            "The client wants the fix deployed today.",
            "Reproduce the bug in the staging environment.",
            "Write the patch for the null pointer issue.",
            "Run the unit test suite now.",
            "Get code review from Sarah this morning.",
            "Deploy to staging for QA testing.",
            "Monitor error rates for the next hour.",
            "Push to production before end of business.",
        ],
        [
            "Pack for tomorrow morning's flight.",
            "Check the weather at the destination now.",
            "Lay out clothes for the trip tonight.",
            "Charge all devices before bed.",
            "Set the alarm for 5 AM tomorrow.",
            "Call a ride to the airport early.",
            "Print boarding pass and hotel confirmation.",
            "Double check passport is in the bag.",
        ],
        [
            "The presentation is in thirty minutes.",
            "Do a final review of the slides now.",
            "Test the screen sharing connection.",
            "Prepare answers for likely questions.",
            "Get water and clear throat.",
            "Start with the executive summary.",
            "Walk through the demo carefully.",
            "Close with the ask and next steps.",
        ],
    ]

    # Long-term planning sequences (extended horizon, strategic)
    longterm_templates = [
        [
            "I want to transition into machine learning research.",
            "Spend this semester building mathematical foundations in linear algebra and probability.",
            "Over the next six months, work through key ML textbooks and implement algorithms from scratch.",
            "Apply to summer research internships at leading AI labs.",
            "By next year, identify a specific subfield to specialize in.",
            "Publish a workshop paper based on initial research findings.",
            "Build relationships with potential PhD advisors over the coming years.",
            "Apply for doctoral programs three years from now with a strong research portfolio.",
        ],
        [
            "The company needs to expand into the European market.",
            "Spend the first quarter conducting market research across EU countries.",
            "Over six months, build partnerships with local distributors.",
            "Establish a regional office by the end of next year.",
            "Hire a local leadership team over the following year.",
            "Adapt the product for European regulations across two years.",
            "Target profitability in the region within five years.",
            "Build brand recognition to rival local competitors within a decade.",
        ],
        [
            "We need to address climate change in our operations.",
            "Commission a comprehensive carbon audit over the next quarter.",
            "Develop a five-year sustainability roadmap this year.",
            "Transition to renewable energy sources over the next three years.",
            "Redesign supply chains for lower emissions by 2030.",
            "Invest in carbon capture research for long-term offsets.",
            "Achieve net-zero operations within the decade.",
            "Become an industry leader in sustainable practices by 2040.",
        ],
        [
            "I want to build generational wealth for my family.",
            "This year, eliminate all high-interest debt and build an emergency fund.",
            "Over the next two years, maximize retirement account contributions.",
            "Start investing in diversified index funds over five years.",
            "Purchase rental property within the next seven years.",
            "Set up education funds for children over the decade.",
            "Create a family trust within fifteen years.",
            "Build a legacy portfolio that compounds over multiple generations.",
        ],
        [
            "The city needs better public transportation infrastructure.",
            "Conduct a ridership study over the next six months.",
            "Develop the master transit plan over the coming year.",
            "Secure federal funding applications over two years.",
            "Begin construction of the first light rail line in three years.",
            "Expand bus rapid transit routes over five years.",
            "Complete the downtown transit hub within seven years.",
            "Achieve full network integration connecting all neighborhoods by 2040.",
        ],
    ]

    for i, steps_text in enumerate(immediate_templates[:n_per_class]):
        steps = [
            SequenceStep(
                step_index=j, prompt=text, label=0,
                sequence_id=f"imm_plan_{i}", total_steps=len(steps_text),
            )
            for j, text in enumerate(steps_text)
        ]
        sequences.append(TaskSequence(
            sequence_id=f"imm_plan_{i}", name=f"Immediate Planning {i}",
            description="Short-horizon step-by-step execution plan",
            steps=steps, label=0,
        ))

    for i, steps_text in enumerate(longterm_templates[:n_per_class]):
        steps = [
            SequenceStep(
                step_index=j, prompt=text, label=1,
                sequence_id=f"lt_plan_{i}", total_steps=len(steps_text),
            )
            for j, text in enumerate(steps_text)
        ]
        sequences.append(TaskSequence(
            sequence_id=f"lt_plan_{i}", name=f"Long-term Planning {i}",
            description="Extended-horizon strategic planning sequence",
            steps=steps, label=1,
        ))

    return sequences


def generate_reasoning_chains(n_per_class: int = 10) -> list[TaskSequence]:
    """
    Multi-step reasoning chains where the temporal horizon is revealed
    progressively. Tests whether activations shift as temporal context
    accumulates.
    """
    sequences = []

    # Immediate-horizon reasoning chains
    immediate_chains = [
        [
            "A customer is experiencing a system outage.",
            "The outage started fifteen minutes ago.",
            "Critical business operations are currently halted.",
            "The team needs to restore service immediately.",
            "There is a known workaround that takes ten minutes.",
            "Apply the emergency patch to the production servers now.",
            "Verify all services are back online within the hour.",
            "Write the incident report before end of shift.",
        ],
        [
            "The patient presents with acute chest pain.",
            "Symptoms started approximately two hours ago.",
            "Initial ECG shows ST-segment elevation.",
            "This requires immediate intervention.",
            "Administer aspirin and nitroglycerin now.",
            "Prepare the cardiac catheterization lab urgently.",
            "The procedure should begin within ninety minutes.",
            "Monitor in the ICU for the next twenty-four hours.",
        ],
        [
            "A severe storm warning has been issued.",
            "The storm will reach our area within three hours.",
            "All outdoor events need to be cancelled today.",
            "Secure all loose equipment on the grounds now.",
            "Activate the emergency communication system immediately.",
            "Open the shelters for the afternoon.",
            "Position emergency response teams within the hour.",
            "Begin damage assessment once the storm passes tonight.",
        ],
    ]

    # Long-term reasoning chains
    longterm_chains = [
        [
            "Global demographics are shifting dramatically.",
            "Population aging will accelerate over the next two decades.",
            "Healthcare systems need fundamental restructuring.",
            "We should begin training geriatric specialists at scale.",
            "Investment in preventive medicine will pay off over generations.",
            "Retirement systems need redesigning for longer lifespans.",
            "Cities should be reimagined for aging populations by 2050.",
            "These changes will define society for the rest of the century.",
        ],
        [
            "Artificial intelligence is transforming every industry.",
            "Workforce displacement will increase over the next decade.",
            "Education systems must adapt over the coming years.",
            "Retraining programs should begin scaling immediately for long-term impact.",
            "New economic models will emerge over the next generation.",
            "Policy frameworks need years of development and testing.",
            "The full economic transformation will unfold over decades.",
            "Future generations will inhabit a fundamentally different labor market.",
        ],
        [
            "Antibiotic resistance is a growing global threat.",
            "Current resistance trends project catastrophic outcomes by 2050.",
            "New antibiotic development pipelines take a decade to mature.",
            "Phage therapy research needs sustained multi-year investment.",
            "Hospital stewardship programs must be implemented over the next five years.",
            "Agricultural antibiotic use needs phasing out over a generation.",
            "Global surveillance systems require decades of cooperation to build.",
            "The full solution will require a century of coordinated effort.",
        ],
    ]

    for i, chain in enumerate(immediate_chains[:n_per_class]):
        steps = [
            SequenceStep(
                step_index=j, prompt=text, label=0,
                sequence_id=f"imm_reason_{i}", total_steps=len(chain),
            )
            for j, text in enumerate(chain)
        ]
        sequences.append(TaskSequence(
            sequence_id=f"imm_reason_{i}", name=f"Immediate Reasoning {i}",
            description="Urgent reasoning chain with short time horizon",
            steps=steps, label=0,
        ))

    for i, chain in enumerate(longterm_chains[:n_per_class]):
        steps = [
            SequenceStep(
                step_index=j, prompt=text, label=1,
                sequence_id=f"lt_reason_{i}", total_steps=len(chain),
            )
            for j, text in enumerate(chain)
        ]
        sequences.append(TaskSequence(
            sequence_id=f"lt_reason_{i}", name=f"Long-term Reasoning {i}",
            description="Extended reasoning chain with long time horizon",
            steps=steps, label=1,
        ))

    return sequences


def generate_cumulative_context_sequences(n_per_class: int = 8) -> list[TaskSequence]:
    """
    Sequences where each step is the CUMULATIVE context (step 0 + step 1 + ...),
    testing how activations evolve as the model processes increasingly long
    temporal narratives.
    """
    sequences = []

    immediate_narratives = [
        [
            "The server is down.",
            "It crashed five minutes ago during peak traffic.",
            "Users are reporting errors right now and the team is scrambling.",
            "We identified the root cause: a memory leak introduced in today's deployment.",
            "The rollback is in progress and should complete within ten minutes.",
            "Services are partially restored. Full recovery expected within the hour.",
            "All systems are back online. Writing the post-mortem report tonight.",
        ],
        [
            "There's a fire alarm in the building.",
            "Everyone needs to evacuate immediately using the nearest exit.",
            "The fire department has been called and should arrive within minutes.",
            "All employees are assembling at the designated meeting point right now.",
            "The fire marshal is assessing the situation and will update us shortly.",
            "The alarm was caused by a kitchen incident. Building cleared for re-entry.",
            "Document the incident and update safety protocols before end of day.",
        ],
    ]

    longterm_narratives = [
        [
            "The ocean ecosystem is changing.",
            "Coral bleaching events have doubled in frequency over the past decade.",
            "Marine biodiversity loss is accelerating and will reshape ocean food webs over generations.",
            "Conservation efforts today will take decades to show measurable recovery.",
            "Marine protected areas established now need fifty years to reach full ecological potential.",
            "The choices we make this decade will determine ocean health for the next century.",
            "Our grandchildren will inherit either a thriving or collapsed marine ecosystem.",
        ],
        [
            "A new language is emerging in online communities.",
            "Linguistic evolution is happening faster than ever in digital spaces.",
            "Over the next decade, these language patterns will reshape formal communication norms.",
            "Educational curricula will need multi-year overhauls to accommodate shifting literacy.",
            "Within a generation, written language conventions may be fundamentally different.",
            "Translation systems will need continuous adaptation over decades to keep pace.",
            "The long-term trajectory of human language is being reshaped for centuries to come.",
        ],
    ]

    for i, steps_text in enumerate(immediate_narratives[:n_per_class]):
        # Build cumulative versions
        cumulative_steps = []
        for j in range(len(steps_text)):
            cumulative_text = " ".join(steps_text[: j + 1])
            cumulative_steps.append(
                SequenceStep(
                    step_index=j, prompt=cumulative_text, label=0,
                    sequence_id=f"imm_cumul_{i}", total_steps=len(steps_text),
                )
            )
        sequences.append(TaskSequence(
            sequence_id=f"imm_cumul_{i}", name=f"Immediate Cumulative {i}",
            description="Cumulative context building for short-horizon narrative",
            steps=cumulative_steps, label=0,
        ))

    for i, steps_text in enumerate(longterm_narratives[:n_per_class]):
        cumulative_steps = []
        for j in range(len(steps_text)):
            cumulative_text = " ".join(steps_text[: j + 1])
            cumulative_steps.append(
                SequenceStep(
                    step_index=j, prompt=cumulative_text, label=1,
                    sequence_id=f"lt_cumul_{i}", total_steps=len(steps_text),
                )
            )
        sequences.append(TaskSequence(
            sequence_id=f"lt_cumul_{i}", name=f"Long-term Cumulative {i}",
            description="Cumulative context building for long-horizon narrative",
            steps=cumulative_steps, label=1,
        ))

    return sequences


def load_all_sequences() -> dict[str, list[TaskSequence]]:
    """Load all sequence types for the experiment."""
    return {
        "planning": generate_planning_sequences(),
        "reasoning": generate_reasoning_chains(),
        "cumulative": generate_cumulative_context_sequences(),
    }


# ---------------------------------------------------------------------------
# Activation extraction (reused pattern from sae_feature_stability.py)
# ---------------------------------------------------------------------------

def extract_activations(
    model: HookedTransformer,
    sae: SaeLensSAE,
    prompts: list[str],
    layer: int,
    batch_size: int = 16,
    device: str = "cpu",
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract raw residual stream activations and SAE-encoded latents."""
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
        sae_out = sae.encode(acts)
        all_raw.append(acts.detach().float().cpu().numpy())
        all_sae.append(sae_out.detach().float().cpu().numpy())

    return np.concatenate(all_raw), np.concatenate(all_sae)


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_step_drift(
    step_activations: dict[int, np.ndarray],
    top_k_indices: np.ndarray,
    k: int = TOP_K_LATENTS,
) -> list[dict]:
    """
    Compute drift metrics at each step relative to step 0 and to previous step.

    Args:
        step_activations: {step_index: sae_latents array (n_samples, d_sae)}
        top_k_indices: discriminative latent indices from training
        k: number of top latents

    Returns:
        List of drift metric dicts per step.
    """
    sorted_steps = sorted(step_activations.keys())
    if len(sorted_steps) < 2:
        return []

    start_acts = step_activations[sorted_steps[0]]
    start_mean = start_acts[:, top_k_indices].mean(axis=0)
    start_active = set(np.argsort(np.abs(start_acts).mean(axis=0))[-k:][::-1].tolist())

    results = []
    prev_mean = start_mean.copy()

    for step_idx in sorted_steps:
        curr_acts = step_activations[step_idx]
        curr_mean = curr_acts[:, top_k_indices].mean(axis=0)
        curr_active = set(np.argsort(np.abs(curr_acts).mean(axis=0))[-k:][::-1].tolist())

        # Cosine similarity to start
        cos_to_start = float(np.dot(curr_mean, start_mean) / (
            np.linalg.norm(curr_mean) * np.linalg.norm(start_mean) + 1e-10
        ))

        # Jaccard to start
        intersection = len(start_active & curr_active)
        union = len(start_active | curr_active)
        jaccard_to_start = intersection / union if union > 0 else 0.0

        # Magnitude ratio vs start
        start_mag = np.abs(start_acts[:, top_k_indices]).mean()
        curr_mag = np.abs(curr_acts[:, top_k_indices]).mean()
        mag_ratio = float(curr_mag / start_mag) if start_mag > 0 else 0.0

        # Cosine to previous step
        cos_to_prev = float(np.dot(curr_mean, prev_mean) / (
            np.linalg.norm(curr_mean) * np.linalg.norm(prev_mean) + 1e-10
        ))

        # Drift velocity (L2 distance from previous step)
        drift_vel = float(np.linalg.norm(curr_mean - prev_mean))

        results.append({
            "step_index": step_idx,
            "cosine_sim_to_start": cos_to_start,
            "jaccard_to_start": jaccard_to_start,
            "activation_magnitude_ratio": mag_ratio,
            "cosine_sim_to_prev": cos_to_prev,
            "drift_velocity": drift_vel,
        })

        prev_mean = curr_mean.copy()

    return results


def assess_drift_predictability(step_metrics: list[dict]) -> tuple[bool, float, float]:
    """
    Assess whether activation drift follows a predictable pattern.

    Returns:
        is_monotonic: whether cosine_sim_to_start decreases monotonically
        linear_r2: R^2 of linear fit (higher = more predictable)
        total_drift: 1 - cosine_sim at final step
    """
    if len(step_metrics) < 3:
        return True, 1.0, 0.0

    cos_vals = [m["cosine_sim_to_start"] for m in step_metrics]
    steps = np.arange(len(cos_vals))

    # Check monotonicity (allow small violations)
    diffs = np.diff(cos_vals)
    is_monotonic = bool(np.sum(diffs > 0.01) <= 1)  # allow 1 violation

    # Linear fit R^2
    if np.std(cos_vals) < 1e-8:
        linear_r2 = 1.0
    else:
        coeffs = np.polyfit(steps, cos_vals, 1)
        predicted = np.polyval(coeffs, steps)
        ss_res = np.sum((np.array(cos_vals) - predicted) ** 2)
        ss_tot = np.sum((np.array(cos_vals) - np.mean(cos_vals)) ** 2)
        linear_r2 = float(1 - ss_res / (ss_tot + 1e-10))

    total_drift = 1.0 - cos_vals[-1]

    return is_monotonic, max(0.0, linear_r2), total_drift


# ---------------------------------------------------------------------------
# Main experiment pipeline
# ---------------------------------------------------------------------------

def _wandb_active() -> bool:
    return HAS_WANDB and wandb.run is not None


def run_single_layer_sequential(
    model: HookedTransformer,
    sae: SaeLensSAE,
    layer: int,
    sequences: dict[str, list[TaskSequence]],
    device: str = "cpu",
    batch_size: int = 16,
) -> list[SequenceResult]:
    """Run sequential tracking experiment for one layer."""

    print(f"\n{'#' * 70}")
    print(f"# LAYER {layer} — Sequential Activation Tracking")
    print(f"{'#' * 70}")

    layer_results = []

    for seq_type, seqs in sequences.items():
        print(f"\n  === Sequence type: {seq_type} ({len(seqs)} sequences) ===")

        # Determine max steps across all sequences of this type
        max_steps = max(len(s.steps) for s in seqs)
        print(f"  Max steps per sequence: {max_steps}")

        # Step 1: Collect all step-0 prompts to train the probe
        step0_prompts = [s.steps[0].prompt for s in seqs]
        step0_labels = np.array([s.label for s in seqs])

        print("  [1/3] Training probe on step-0 activations...")
        raw_0, sae_0 = extract_activations(
            model, sae, step0_prompts, layer,
            batch_size=batch_size, device=device,
        )

        # Select discriminative latents from step 0
        mean_imm = sae_0[step0_labels == 0].mean(axis=0)
        mean_lt = sae_0[step0_labels == 1].mean(axis=0)
        abs_diff = np.abs(mean_lt - mean_imm)
        top_k_indices = np.argsort(abs_diff)[-TOP_K_LATENTS:][::-1]

        # Train probes on step 0
        X_sae_0 = sae_0[:, top_k_indices]
        sae_probe = LogisticRegression(max_iter=5000, random_state=42)
        sae_probe.fit(X_sae_0, step0_labels)

        act_probe = LogisticRegression(max_iter=5000, random_state=42)
        act_probe.fit(raw_0, step0_labels)

        # Step 2: Extract activations at each step position
        print("  [2/3] Tracking activations across steps...")
        step_sae_activations = {0: sae_0}
        all_step_metrics = []

        for step_idx in range(max_steps):
            # Collect prompts for this step across all sequences
            step_prompts = []
            step_labels = []
            for s in seqs:
                if step_idx < len(s.steps):
                    step_prompts.append(s.steps[step_idx].prompt)
                    step_labels.append(s.label)
            step_labels = np.array(step_labels)

            if len(step_prompts) < 4:
                continue

            if step_idx > 0:
                raw_s, sae_s = extract_activations(
                    model, sae, step_prompts, layer,
                    batch_size=batch_size, device=device,
                )
                step_sae_activations[step_idx] = sae_s
            else:
                raw_s, sae_s = raw_0, sae_0

            # Evaluate probes at this step
            X_sae_s = sae_s[:, top_k_indices]
            sae_acc = accuracy_score(step_labels, sae_probe.predict(X_sae_s))
            act_acc = accuracy_score(step_labels, act_probe.predict(raw_s))
            sae_f1 = f1_score(step_labels, sae_probe.predict(X_sae_s), zero_division=0)
            act_f1 = f1_score(step_labels, act_probe.predict(raw_s), zero_division=0)

            all_step_metrics.append({
                "step_index": step_idx,
                "n_samples": len(step_prompts),
                "sae_probe_accuracy": float(sae_acc),
                "act_probe_accuracy": float(act_acc),
                "sae_probe_f1": float(sae_f1),
                "act_probe_f1": float(act_f1),
            })

            print(f"    Step {step_idx}: SAE acc={sae_acc:.3f}, Act acc={act_acc:.3f} ({len(step_prompts)} samples)")

        # Step 3: Compute drift metrics
        print("  [3/3] Computing drift metrics...")
        drift_metrics = compute_step_drift(step_sae_activations, top_k_indices)
        is_monotonic, linear_r2, total_drift = assess_drift_predictability(drift_metrics)

        # Merge drift metrics into step metrics
        step_results = []
        for sm in all_step_metrics:
            step_idx = sm["step_index"]
            dm = next((d for d in drift_metrics if d["step_index"] == step_idx), {})
            step_results.append(StepMetrics(
                step_index=step_idx,
                n_samples=sm["n_samples"],
                sae_probe_accuracy=sm["sae_probe_accuracy"],
                act_probe_accuracy=sm["act_probe_accuracy"],
                sae_probe_f1=sm["sae_probe_f1"],
                act_probe_f1=sm["act_probe_f1"],
                cosine_sim_to_start=dm.get("cosine_sim_to_start", 1.0),
                jaccard_to_start=dm.get("jaccard_to_start", 1.0),
                activation_magnitude_ratio=dm.get("activation_magnitude_ratio", 1.0),
                cosine_sim_to_prev=dm.get("cosine_sim_to_prev", 1.0),
                drift_velocity=dm.get("drift_velocity", 0.0),
            ))

        result = SequenceResult(
            layer=layer,
            sequence_type=seq_type,
            step_metrics=step_results,
            drift_is_monotonic=is_monotonic,
            drift_predictability=linear_r2,
            total_drift=total_drift,
        )
        layer_results.append(result)

        print(f"\n  Summary ({seq_type}):")
        print(f"    Monotonic drift: {is_monotonic}")
        print(f"    Drift predictability (R²): {linear_r2:.4f}")
        print(f"    Total drift: {total_drift:.4f}")

        # W&B logging
        if _wandb_active():
            for sm in step_results:
                wandb.log({
                    f"layer_{layer}/{seq_type}/step_{sm.step_index}/sae_accuracy": sm.sae_probe_accuracy,
                    f"layer_{layer}/{seq_type}/step_{sm.step_index}/act_accuracy": sm.act_probe_accuracy,
                    f"layer_{layer}/{seq_type}/step_{sm.step_index}/cosine_to_start": sm.cosine_sim_to_start,
                    f"layer_{layer}/{seq_type}/step_{sm.step_index}/drift_velocity": sm.drift_velocity,
                })
            wandb.log({
                f"layer_{layer}/{seq_type}/drift_predictability": linear_r2,
                f"layer_{layer}/{seq_type}/total_drift": total_drift,
                f"layer_{layer}/{seq_type}/is_monotonic": int(is_monotonic),
            })

    return layer_results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_sequential_results(
    all_results: list[SequenceResult],
    output_dir: Path,
    timestamp: str,
):
    """Generate plots for sequential activation tracking."""

    # Group results by sequence type
    by_type = {}
    for r in all_results:
        by_type.setdefault(r.sequence_type, []).append(r)

    for seq_type, results in by_type.items():
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Sequential Activation Tracking — {seq_type}", fontsize=14)

        for r in results:
            steps = [m.step_index for m in r.step_metrics]

            # Probe accuracy vs step
            axes[0, 0].plot(steps, [m.sae_probe_accuracy for m in r.step_metrics],
                           "o-", label=f"SAE L{r.layer}")
            axes[0, 0].plot(steps, [m.act_probe_accuracy for m in r.step_metrics],
                           "s--", label=f"Act L{r.layer}", alpha=0.7)
        axes[0, 0].set_xlabel("Sequence Step")
        axes[0, 0].set_ylabel("Probe Accuracy")
        axes[0, 0].set_title("Probe Accuracy vs. Sequence Position")
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].set_ylim(0, 1.05)
        axes[0, 0].grid(True, alpha=0.3)

        for r in results:
            steps = [m.step_index for m in r.step_metrics]
            # Cosine similarity to start
            axes[0, 1].plot(steps, [m.cosine_sim_to_start for m in r.step_metrics],
                           "o-", label=f"L{r.layer}")
        axes[0, 1].set_xlabel("Sequence Step")
        axes[0, 1].set_ylabel("Cosine Similarity to Step 0")
        axes[0, 1].set_title("Feature Drift from Initial State")
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)

        for r in results:
            steps = [m.step_index for m in r.step_metrics]
            # Drift velocity
            axes[1, 0].plot(steps, [m.drift_velocity for m in r.step_metrics],
                           "o-", label=f"L{r.layer}")
        axes[1, 0].set_xlabel("Sequence Step")
        axes[1, 0].set_ylabel("Drift Velocity (L2)")
        axes[1, 0].set_title("Activation Drift Rate per Step")
        axes[1, 0].legend(fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)

        for r in results:
            steps = [m.step_index for m in r.step_metrics]
            # Jaccard to start
            axes[1, 1].plot(steps, [m.jaccard_to_start for m in r.step_metrics],
                           "o-", label=f"L{r.layer}")
        axes[1, 1].set_xlabel("Sequence Step")
        axes[1, 1].set_ylabel("Jaccard Similarity to Step 0")
        axes[1, 1].set_title("Active Feature Set Stability")
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = output_dir / f"sequential_{seq_type}_{timestamp}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plot saved: {plot_path}")

        if _wandb_active():
            wandb.log({f"plots/sequential_{seq_type}": wandb.Image(str(plot_path))})

    # Predictability summary bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = []
    r2_vals = []
    total_drifts = []
    for r in all_results:
        labels.append(f"L{r.layer} {r.sequence_type}")
        r2_vals.append(r.drift_predictability)
        total_drifts.append(r.total_drift)

    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, r2_vals, width, label="Drift Predictability (R²)", color="steelblue")
    ax.bar(x + width / 2, total_drifts, width, label="Total Drift", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Value")
    ax.set_title("Drift Predictability and Magnitude Across Layers & Sequence Types")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    plot_path = output_dir / f"drift_predictability_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {plot_path}")

    if _wandb_active():
        wandb.log({"plots/drift_predictability": wandb.Image(str(plot_path))})


# ---------------------------------------------------------------------------
# Results serialization
# ---------------------------------------------------------------------------

def save_results(
    results: list[SequenceResult],
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
            "experiment": "sequential_activation_tracking",
        },
        "results": [],
    }

    for r in results:
        serializable["results"].append({
            "layer": r.layer,
            "sequence_type": r.sequence_type,
            "drift_is_monotonic": r.drift_is_monotonic,
            "drift_predictability": r.drift_predictability,
            "total_drift": r.total_drift,
            "step_metrics": [asdict(m) for m in r.step_metrics],
        })

    out_path = output_dir / f"sequential_results_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {out_path}")

    latest_path = output_dir / "sequential_results_latest.json"
    with open(latest_path, "w") as f:
        json.dump(serializable, f, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_experiment(
    layers: list[int],
    device: str = "cpu",
    batch_size: int = 16,
    output_dir: Optional[Path] = None,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
) -> list[SequenceResult]:
    """Run the full sequential activation tracking experiment."""

    if output_dir is None:
        output_dir = RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialize W&B
    if HAS_WANDB and wandb_project:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run_name or f"seq-tracking-{timestamp}",
            config={
                "model": MODEL_NAME,
                "sae_release": SAE_RELEASE,
                "top_k_latents": TOP_K_LATENTS,
                "layers": layers,
                "device": device,
                "batch_size": batch_size,
                "experiment": "sequential_activation_tracking",
            },
            tags=["sequential-tracking", "activation-drift", "rq1"],
        )
        print(f"W&B run: {wandb.run.url}")

    print("=" * 70)
    print("SEQUENTIAL ACTIVATION TRACKING EXPERIMENT")
    print("=" * 70)
    print(f"Timestamp: {timestamp}")
    print(f"Model: {MODEL_NAME}")
    print(f"Layers: {layers}")
    print(f"Device: {device}")

    # Load sequences
    print("\n--- Loading task sequences ---")
    sequences = load_all_sequences()
    for seq_type, seqs in sequences.items():
        total_steps = sum(len(s.steps) for s in seqs)
        print(f"  {seq_type}: {len(seqs)} sequences, {total_steps} total steps")

    # Load model
    print(f"\n--- Loading {MODEL_NAME} ---")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
    print(f"  Layers: {model.cfg.n_layers}, d_model: {model.cfg.d_model}")

    # Run per layer
    all_results = []
    for layer in layers:
        sae_id = MODEL_CONFIGS[MODEL_NAME]["sae_id_template"].format(layer=layer)
        try:
            sae = SaeLensSAE.from_pretrained(
                release=SAE_RELEASE, sae_id=sae_id, device=device,
            )
        except Exception as e:
            print(f"Failed to load SAE for layer {layer}: {e}")
            continue

        layer_results = run_single_layer_sequential(
            model, sae, layer, sequences,
            device=device, batch_size=batch_size,
        )
        all_results.extend(layer_results)

        del sae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save and plot
    save_results(all_results, output_dir, timestamp)
    plot_sequential_results(all_results, output_dir, timestamp)

    # Upload to W&B
    if _wandb_active():
        artifact = wandb.Artifact(
            f"sequential-results-{timestamp}",
            type="experiment-results",
            description="Sequential activation tracking results",
        )
        results_path = output_dir / f"sequential_results_{timestamp}.json"
        if results_path.exists():
            artifact.add_file(str(results_path))
        wandb.log_artifact(artifact)
        wandb.finish()

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Sequential Activation Tracking")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--quick", action="store_true", help="Single layer, quick check")
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
    output_dir = Path(args.output_dir) if args.output_dir else None

    run_experiment(
        layers=layers,
        device=args.device,
        batch_size=args.batch_size,
        output_dir=output_dir,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
    )


if __name__ == "__main__":
    main()
