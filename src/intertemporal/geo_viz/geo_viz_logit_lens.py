"""Logit Lens Analysis for geometric visualization.

Implements logit lens with proper LayerNorm correction to analyze
how model representations evolve across layers.

Key formula (CORRECT VERSION):
    normed = model.ln_final(resid_post[L])
    logits = normed @ W_U
    logit_diff = logits[:, token_a] - logits[:, token_b]
"""

import gc
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .geo_viz_config import ACTIVATION_DTYPE

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LogitLensResult:
    """Results from logit lens analysis per layer.

    Stores logit differences and cosine similarities for each sample at each layer.
    """

    n_samples: int
    n_layers: int
    # Per-layer logit differences: shape [n_layers, n_samples]
    logit_diffs: np.ndarray
    # Per-layer cosine similarity with logit direction: shape [n_layers, n_samples]
    cosine_sims: np.ndarray
    # Layer indices
    layers: list[int]
    # Token info
    token_a: int
    token_b: int
    token_a_str: str
    token_b_str: str

    def to_dict(self) -> dict:
        """Serialize to dict (without large arrays)."""
        return {
            "n_samples": self.n_samples,
            "n_layers": self.n_layers,
            "layers": self.layers,
            "token_a": self.token_a,
            "token_b": self.token_b,
            "token_a_str": self.token_a_str,
            "token_b_str": self.token_b_str,
            "mean_logit_diff_by_layer": [
                float(self.logit_diffs[i].mean()) for i in range(self.n_layers)
            ],
            "std_logit_diff_by_layer": [
                float(self.logit_diffs[i].std()) for i in range(self.n_layers)
            ],
            "mean_cosine_sim_by_layer": [
                float(self.cosine_sims[i].mean()) for i in range(self.n_layers)
            ],
        }

    def save(self, path: Path):
        """Save to disk."""
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "logit_diffs.npy", self.logit_diffs.astype(ACTIVATION_DTYPE))
        np.save(path / "cosine_sims.npy", self.cosine_sims.astype(ACTIVATION_DTYPE))
        with open(path / "metadata.json", "w") as f:
            json.dump(self.to_dict(), f, separators=(",", ":"))

    @classmethod
    def load(cls, path: Path) -> "LogitLensResult":
        """Load from disk."""
        with open(path / "metadata.json") as f:
            metadata = json.load(f)
        return cls(
            n_samples=metadata["n_samples"],
            n_layers=metadata["n_layers"],
            logit_diffs=np.load(path / "logit_diffs.npy"),
            cosine_sims=np.load(path / "cosine_sims.npy"),
            layers=metadata["layers"],
            token_a=metadata["token_a"],
            token_b=metadata["token_b"],
            token_a_str=metadata["token_a_str"],
            token_b_str=metadata["token_b_str"],
        )


def compute_logit_lens(
    runner,
    input_ids: torch.Tensor,
    response_position: int,
    token_a: int,
    token_b: int,
    layers: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute logit lens with LayerNorm correction for a single input.

    Args:
        runner: ModelRunner with TransformerLens backend
        input_ids: Input token IDs [1, seq_len]
        response_position: Position to extract residual stream from
        token_a: First token ID for logit difference
        token_b: Second token ID for logit difference
        layers: List of layers to analyze (default: all layers)

    Returns:
        Tuple of (logit_diffs, cosine_sims) for each layer
        - logit_diffs: [n_layers] array of logit(a) - logit(b)
        - cosine_sims: [n_layers] array of cosine similarity with logit direction
    """
    model = runner._model
    n_layers = runner.n_layers

    if layers is None:
        layers = list(range(n_layers))

    # Get W_U and compute logit direction
    W_U = runner.W_U  # [d_model, vocab_size]
    logit_direction = W_U[:, token_a] - W_U[:, token_b]  # [d_model]
    logit_direction_norm = logit_direction / (torch.norm(logit_direction) + 1e-10)

    logit_diffs = np.zeros(len(layers), dtype=ACTIVATION_DTYPE)
    cosine_sims = np.zeros(len(layers), dtype=ACTIVATION_DTYPE)

    # Build hook names for all layers' resid_post
    hook_names = [f"blocks.{layer}.hook_resid_post" for layer in layers]
    names_filter = lambda name: name in hook_names

    with torch.no_grad():
        # Run with cache to get all residual streams
        _, cache = runner._backend.run_with_cache(input_ids, names_filter)

        for i, layer in enumerate(layers):
            hook_name = f"blocks.{layer}.hook_resid_post"
            resid_post = cache[hook_name]  # [1, seq_len, d_model]

            # Extract at response position
            resid = resid_post[0, response_position, :]  # [d_model]

            # CRITICAL: Apply final LayerNorm before computing logits
            # This is the key insight - without LayerNorm, logit lens gives wrong results
            normed = model.ln_final(resid)  # [d_model]

            # Compute logits via projection onto W_U
            logits = normed @ W_U  # [vocab_size]

            # Logit difference
            logit_diff = logits[token_a] - logits[token_b]
            logit_diffs[i] = float(logit_diff.cpu())

            # Cosine similarity between normalized residual and logit direction
            normed_normalized = normed / (torch.norm(normed) + 1e-10)
            cos_sim = torch.dot(normed_normalized, logit_direction_norm)
            cosine_sims[i] = float(cos_sim.cpu())

    return logit_diffs, cosine_sims


def run_logit_lens_analysis(
    runner,
    samples: list,
    choices: list,
    config,
    token_a_str: str = "a",
    token_b_str: str = "b",
) -> LogitLensResult | None:
    """Run logit lens analysis on all samples.

    Args:
        runner: ModelRunner with TransformerLens backend
        samples: List of PromptSample objects
        choices: List of ChoiceInfo objects
        config: GeoVizConfig
        token_a_str: String for token A (default "a")
        token_b_str: String for token B (default "b")

    Returns:
        LogitLensResult or None if analysis cannot be run
    """
    # Check backend compatibility
    if not hasattr(runner, "_model") or runner._model is None:
        logger.warning("Logit lens requires TransformerLens backend with model access")
        return None

    if not hasattr(runner._model, "ln_final"):
        logger.warning("Model does not have ln_final, cannot run logit lens")
        return None

    # Get token IDs for a and b
    tokenizer = runner._tokenizer
    token_a_ids = tokenizer.encode(token_a_str, add_special_tokens=False)
    token_b_ids = tokenizer.encode(token_b_str, add_special_tokens=False)

    if not token_a_ids or not token_b_ids:
        logger.warning(f"Could not encode tokens: {token_a_str}, {token_b_str}")
        return None

    token_a = token_a_ids[0]
    token_b = token_b_ids[0]

    n_layers = runner.n_layers
    layers = list(range(n_layers))
    n_samples = len(samples)

    logger.info(f"Running logit lens analysis on {n_samples} samples...")
    logger.info(f"  Token A: '{token_a_str}' (id={token_a})")
    logger.info(f"  Token B: '{token_b_str}' (id={token_b})")

    # Pre-allocate result arrays
    all_logit_diffs = np.zeros((n_layers, n_samples), dtype=ACTIVATION_DTYPE)
    all_cosine_sims = np.zeros((n_layers, n_samples), dtype=ACTIVATION_DTYPE)

    from ..formatting.prompt_formats import find_prompt_format_config
    from ..preference import PreferenceQuerier, PreferenceQueryConfig

    query_config = PreferenceQueryConfig(skip_generation=True)
    querier = PreferenceQuerier(query_config)

    for sample_idx, sample in enumerate(samples):
        if sample_idx % 50 == 0:
            logger.info(f"  Processing sample {sample_idx}/{n_samples}")

        try:
            # Get the formatted prompt
            prompt_format = find_prompt_format_config(sample.formatting_id)
            choice_prefix = prompt_format.get_response_prefix_before_choice()

            # Run the model to get response position
            pref = querier.query_sample(
                sample, runner, choice_prefix, activation_names=[]
            )

            if pref.chosen_traj is None:
                continue

            # Get token IDs including response
            full_tokens = pref.chosen_traj.token_ids
            prompt_len = pref.prompt_token_count

            # Response position is first token of response
            response_position = prompt_len

            # Create input tensor
            input_ids = torch.tensor([full_tokens], device=runner.device)

            # Run logit lens
            logit_diffs, cosine_sims = compute_logit_lens(
                runner,
                input_ids,
                response_position,
                token_a,
                token_b,
                layers,
            )

            all_logit_diffs[:, sample_idx] = logit_diffs
            all_cosine_sims[:, sample_idx] = cosine_sims

            # Clean up
            pref.internals = None
            del pref

        except Exception as e:
            logger.warning(f"  Skipping sample {sample_idx}: {e}")
            continue

        # Periodic GC
        if sample_idx % 100 == 0:
            gc.collect()

    logger.info(f"Logit lens analysis complete")

    return LogitLensResult(
        n_samples=n_samples,
        n_layers=n_layers,
        logit_diffs=all_logit_diffs,
        cosine_sims=all_cosine_sims,
        layers=layers,
        token_a=token_a,
        token_b=token_b,
        token_a_str=token_a_str,
        token_b_str=token_b_str,
    )


def run_logit_lens_from_cache(
    runner,
    data: "ActivationData",
    config,
    token_a_str: str = "a",
    token_b_str: str = "b",
) -> LogitLensResult | None:
    """Run logit lens analysis using cached activations.

    This version uses pre-extracted resid_post activations from disk,
    which is more memory efficient for large datasets.

    Args:
        runner: ModelRunner with TransformerLens backend
        data: ActivationData with cached activations
        config: GeoVizConfig
        token_a_str: String for token A (default "a")
        token_b_str: String for token B (default "b")

    Returns:
        LogitLensResult or None if analysis cannot be run
    """
    # Check backend compatibility
    if not hasattr(runner, "_model") or runner._model is None:
        logger.warning("Logit lens requires TransformerLens backend with model access")
        return None

    if not hasattr(runner._model, "ln_final"):
        logger.warning("Model does not have ln_final, cannot run logit lens")
        return None

    # Get token IDs for a and b
    tokenizer = runner._tokenizer
    token_a_ids = tokenizer.encode(token_a_str, add_special_tokens=False)
    token_b_ids = tokenizer.encode(token_b_str, add_special_tokens=False)

    if not token_a_ids or not token_b_ids:
        logger.warning(f"Could not encode tokens: {token_a_str}, {token_b_str}")
        return None

    token_a = token_a_ids[0]
    token_b = token_b_ids[0]

    model = runner._model
    W_U = runner.W_U  # [d_model, vocab_size]
    logit_direction = W_U[:, token_a] - W_U[:, token_b]
    logit_direction_norm = logit_direction / (torch.norm(logit_direction) + 1e-10)

    # Find available layers from target keys
    # Target keys look like "L21_resid_post_response_choice"
    target_keys = data.get_target_keys()
    resid_post_keys = [k for k in target_keys if "resid_post" in k and "_response" in k]

    if not resid_post_keys:
        logger.warning("No resid_post activations found at response position")
        return None

    # Extract layer numbers
    layers = sorted(set(
        int(k.split("_")[0][1:])
        for k in resid_post_keys
    ))

    n_layers = len(layers)
    n_samples = len(data.samples)

    logger.info(f"Running logit lens from cache on {n_samples} samples, {n_layers} layers")
    logger.info(f"  Token A: '{token_a_str}' (id={token_a})")
    logger.info(f"  Token B: '{token_b_str}' (id={token_b})")

    all_logit_diffs = np.zeros((n_layers, n_samples), dtype=ACTIVATION_DTYPE)
    all_cosine_sims = np.zeros((n_layers, n_samples), dtype=ACTIVATION_DTYPE)

    for layer_idx, layer in enumerate(layers):
        # Find the target key for this layer
        target_key = f"L{layer}_resid_post_response"
        if target_key not in target_keys:
            # Try alternate naming
            target_key = f"L{layer}_resid_post_dest"
            if target_key not in target_keys:
                logger.warning(f"  No resid_post for layer {layer}")
                continue

        if layer_idx % 5 == 0:
            logger.info(f"  Processing layer {layer}/{layers[-1]}")

        # Load activations for this layer
        try:
            resid_posts = data.load_target(target_key)  # [n_samples, d_model]
        except Exception as e:
            logger.warning(f"  Failed to load {target_key}: {e}")
            continue

        # Process all samples for this layer
        with torch.no_grad():
            for sample_idx in range(min(resid_posts.shape[0], n_samples)):
                resid = torch.tensor(
                    resid_posts[sample_idx],
                    device=runner.device,
                    dtype=runner.dtype,
                )

                # CRITICAL: Apply final LayerNorm
                normed = model.ln_final(resid)

                # Compute logits
                logits = normed @ W_U

                # Logit difference
                logit_diff = logits[token_a] - logits[token_b]
                all_logit_diffs[layer_idx, sample_idx] = float(logit_diff.cpu())

                # Cosine similarity
                normed_normalized = normed / (torch.norm(normed) + 1e-10)
                cos_sim = torch.dot(normed_normalized, logit_direction_norm)
                all_cosine_sims[layer_idx, sample_idx] = float(cos_sim.cpu())

        # Unload to free memory
        data.unload_target(target_key)

        if layer_idx % 10 == 0:
            gc.collect()

    logger.info(f"Logit lens analysis complete")

    return LogitLensResult(
        n_samples=n_samples,
        n_layers=n_layers,
        logit_diffs=all_logit_diffs,
        cosine_sims=all_cosine_sims,
        layers=layers,
        token_a=token_a,
        token_b=token_b,
        token_a_str=token_a_str,
        token_b_str=token_b_str,
    )
