"""Top-K Sparse Autoencoder: model, training, and metrics."""

from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ...common.device_utils import get_device
from .sae_activations import Sentence


# ── Constants ───────────────────────────────────────────────────────────────

DECODER_NORM_EPS = 1e-5
DEFAULT_LR_BASE = 2e-4
DEFAULT_LR_SCALE_FACTOR = 2**14


# ── SAE Model ──────────────────────────────────────────────────────────────


@dataclass
class SAESpec:
    """Lightweight SAE specification before d_in is known.

    Attributes:
        layer: Transformer layer index
        component: Hook type (resid_pre, resid_post, mlp_out, attn_out)
        position_name: Named position (source, dest, secondary_source)
        num_latents: Number of SAE latent features
        k: Top-k sparsity constraint
    """

    layer: int
    component: str
    position_name: str
    num_latents: int
    k: int

    def get_name(self) -> str:
        """Canonical name for this SAE config, used for file/dir naming."""
        return f"L{self.layer}_{self.component}_P{self.position_name}_k{self.num_latents}_t{self.k}"

    def get_target_key(self) -> str:
        """Get key for the (layer, component, position) tuple."""
        return f"L{self.layer}_{self.component}_P{self.position_name}"


class SAE(nn.Module):
    """Top-K Sparse Autoencoder for clustering activations.

    Architecture:
        Encoder: Linear(d_in -> num_latents) with bias
        Sparsity: Keep only top-k activations
        Decoder: Weighted sum of k decoder columns + bias (unit-norm columns)

    Attributes:
        d_in: Input dimension (model hidden size)
        num_latents: Number of latent features
        k: Top-k sparsity constraint
        layer: Transformer layer index (set after init)
        component: Hook type (resid_pre, resid_post, mlp_out, attn_out)
        position_name: Named position (source, dest, secondary_source)
    """

    def __init__(self, d_in: int, num_latents: int, k: int = 3):
        super().__init__()
        self.d_in = d_in
        self.num_latents = num_latents
        self.k = k

        # These are set after initialization
        self.layer: int = -1
        self.component: str = ""
        self.position_name: str = ""

        self.encoder = nn.Linear(d_in, num_latents, bias=True)
        self.encoder.bias.data.zero_()

        self.W_dec = nn.Parameter(self.encoder.weight.data.clone())
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        self.register_buffer("activation_mean", torch.zeros(d_in))
        self.set_decoder_norm_to_unit_norm()

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        norm = torch.norm(self.W_dec.data, dim=1, keepdim=True)
        self.W_dec.data /= norm + DECODER_NORM_EPS

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        forward = self.encoder(x - self.b_dec)
        top_acts, top_indices = forward.topk(self.k, dim=-1)
        return top_acts, top_indices

    def decode(self, top_acts: torch.Tensor, top_indices: torch.Tensor) -> torch.Tensor:
        selected_weights = self.W_dec[top_indices]
        weighted = top_acts.unsqueeze(-1) * selected_weights
        return weighted.sum(dim=1) + self.b_dec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        top_acts, top_indices = self.encode(x)
        return self.decode(top_acts, top_indices)

    def get_cluster_assignments(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.encoder(x - self.b_dec).argmax(dim=1)

    def get_name(self) -> str:
        """Canonical name for this SAE config, used for file/dir naming."""
        if self.component and self.position_name:
            return f"L{self.layer}_{self.component}_P{self.position_name}_k{self.num_latents}_t{self.k}"
        return f"L{self.layer}_k{self.num_latents}_t{self.k}"

    def get_target_key(self) -> str:
        """Get key for the (layer, component, position) tuple."""
        return f"L{self.layer}_{self.component}_P{self.position_name}"

    def get_all_activations(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.encoder(x - self.b_dec)

    def save_checkpoint(self, path: Path, metadata: dict = None):
        checkpoint = {
            "d_in": self.d_in,
            "num_latents": self.num_latents,
            "k": self.k,
            "layer": self.layer,
            "component": self.component,
            "position_name": self.position_name,
            "encoder_weight": self.encoder.weight.data.cpu(),
            "encoder_bias": self.encoder.bias.data.cpu(),
            "W_dec": self.W_dec.data.cpu(),
            "b_dec": self.b_dec.data.cpu(),
        }
        if metadata:
            checkpoint.update(metadata)
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path: Path, device: str = "cpu") -> "SAE":
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        sae = cls(checkpoint["d_in"], checkpoint["num_latents"], checkpoint["k"])
        sae.encoder.weight.data = checkpoint["encoder_weight"]
        sae.encoder.bias.data = checkpoint["encoder_bias"]
        sae.W_dec.data = checkpoint["W_dec"]
        sae.b_dec.data = checkpoint["b_dec"]
        sae.layer = checkpoint.get("layer", -1)
        sae.component = checkpoint.get("component", "")
        sae.position_name = checkpoint.get("position_name", "")
        return sae.to(device)


# ── Training Metrics ────────────────────────────────────────────────────────


@dataclass
class TrainingMetrics:
    epoch: int = 0
    mse_loss: float = 0.0
    l0_sparsity: float = 0.0
    dead_latent_ratio: float = 0.0
    active_latent_count: int = 0
    best_loss: float = float("inf")

    def to_dict(self) -> dict:
        return asdict(self)


def compute_training_metrics(
    sae: SAE,
    X_tensor: torch.Tensor,
    latent_activation_counts: np.ndarray,
) -> TrainingMetrics:
    with torch.no_grad():
        top_acts, top_indices = sae.encode(X_tensor)
        l0_sparsity = (top_acts > 0).float().sum(dim=-1).mean().item()
        predicted = sae.decode(top_acts, top_indices)
        mse_loss = ((predicted - X_tensor) ** 2).mean().item()
        dead_latent_ratio = (latent_activation_counts == 0).mean()
        active_latent_count = (latent_activation_counts > 0).sum()

    return TrainingMetrics(
        mse_loss=mse_loss,
        l0_sparsity=l0_sparsity,
        dead_latent_ratio=float(dead_latent_ratio),
        active_latent_count=int(active_latent_count),
    )


# ── Training ────────────────────────────────────────────────────────────────


def train_single_sae(
    sae: SAE,
    X: np.ndarray,
    lr: float = 0.001,
    device: str = "mps",
    max_epochs: int = 500,
    patience: int = 20,
    log_interval: int = 50,
    tb_writer=None,
    tb_tag: str = "",
    tb_global_step: int = 0,
) -> tuple[SAE, float, list[dict]]:
    """Train a single SAE with early stopping. Returns (sae, best_loss, history).

    If resume_from is set, loads the checkpoint and continues training
    (fresh patience counter, same weights).
    """
    X_tensor = torch.from_numpy(X).float().to(device)
    n_samples, d_in = X.shape
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    X_centered_sq_sum = ((X_tensor - X_tensor.mean(dim=0, keepdim=True)) ** 2).sum()

    best_loss = float("inf")
    patience_counter = 0
    best_state = None
    training_history = []
    latent_activation_counts = np.zeros(sae.num_latents, dtype=np.int64)

    for epoch in range(max_epochs):
        sae.train()

        top_acts, top_indices = sae.encode(X_tensor)
        predicted = sae.decode(top_acts, top_indices)
        loss = ((predicted - X_tensor) ** 2).sum() / X_centered_sq_sum

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sae.set_decoder_norm_to_unit_norm()

        avg_loss = loss.item()

        indices_np = top_indices.detach().cpu().numpy().flatten()
        np.add.at(latent_activation_counts, indices_np, 1)

        step = tb_global_step + epoch
        if tb_writer is not None:
            tb_writer.add_scalar(f"{tb_tag}/loss", avg_loss, step)

        if epoch % log_interval == 0 or epoch == max_epochs - 1:
            metrics = compute_training_metrics(sae, X_tensor, latent_activation_counts)
            metrics.epoch = epoch
            metrics.best_loss = best_loss
            training_history.append(metrics.to_dict())

            if tb_writer is not None:
                tb_writer.add_scalar(f"{tb_tag}/l0_sparsity", metrics.l0_sparsity, step)
                tb_writer.add_scalar(
                    f"{tb_tag}/dead_latent_ratio", metrics.dead_latent_ratio, step
                )
                tb_writer.add_scalar(
                    f"{tb_tag}/active_latent_count", metrics.active_latent_count, step
                )

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in sae.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state:
        sae.load_state_dict(best_state)

    final_metrics = compute_training_metrics(sae, X_tensor, latent_activation_counts)
    final_metrics.epoch = epoch
    final_metrics.best_loss = best_loss
    training_history.append(final_metrics.to_dict())

    return sae, best_loss, training_history


# ── Pipeline helpers ─────────────────────────────────────────────────────────


def initialize_sae_models(state) -> list[SAESpec]:
    """Create SAE specs for all (layer, component, position, cluster_size, topk) configurations."""
    specs = []
    components = getattr(state.config, "components", ["resid_post"])
    position_names = getattr(state.config, "position_names", ["dest"])

    for layer in state.config.layers:
        for component in components:
            for position_name in position_names:
                for n_clusters in state.config.num_time_horizons_bins:
                    for topk in state.config.topk_values:
                        if n_clusters <= topk:
                            continue
                        specs.append(
                            SAESpec(
                                layer=layer,
                                component=component,
                                position_name=position_name,
                                num_latents=n_clusters,
                                k=topk,
                            )
                        )
    print(f"  Initialized {len(specs)} SAE configurations")
    return specs


def load_sae_models(state, sae_dir: str) -> list[SAE]:
    """Load all SAE checkpoints from sae_dir."""
    sae_path = Path(sae_dir)
    device = get_device()
    models = []
    components = getattr(state.config, "components", ["resid_post"])
    position_names = getattr(state.config, "position_names", ["dest"])

    for layer in state.config.layers:
        for component in components:
            for position_name in position_names:
                for n_clusters in state.config.num_time_horizons_bins:
                    for topk in state.config.topk_values:
                        if n_clusters <= topk:
                            continue
                        spec = SAESpec(
                            layer=layer,
                            component=component,
                            position_name=position_name,
                            num_latents=n_clusters,
                            k=topk,
                        )
                        path = sae_path / f"{spec.get_name()}.pt"
                        if path.exists():
                            sae = SAE.load_checkpoint(path, device)
                            models.append(sae)
                        else:
                            # No checkpoint yet, create a spec for fresh training
                            print(
                                f"\n\nload_sae_models cannot find {path}. Creating fresh. \n\n"
                            )
                            models.append(spec)
    print(f"  Loaded {len(models)} SAE models/specs")
    return models


def save_sae_model(sae_dir: str, sae: SAE) -> None:
    """Save an SAE checkpoint to sae_dir."""
    sae_path = Path(sae_dir)
    sae_path.mkdir(parents=True, exist_ok=True)
    name = sae.get_name()
    path = sae_path / f"{name}.pt"
    sae.save_checkpoint(path)
    print(f"    Saved {name} -> {path}")


def train_sae(
    x_norm: np.ndarray,
    sae: SAE | SAESpec,
    batch_size: int,
    max_epochs: int,
    patience: int,
    tb_writer=None,
    tb_prefix: str = "",
    tb_global_step: int = 0,
) -> tuple[SAE, dict]:
    """Train a single SAE on position-specific activation data.

    Accepts either an existing SAE model (for continued training) or an
    SAESpec (for fresh training).

    Args:
        x_norm: Activation matrix (n_samples, d_in) for training.
        tb_global_step: Global step offset for TensorBoard logging so that
            online-SGD iterations produce a continuous loss curve.

    Returns (trained_sae, training_results_dict).
    """
    device = get_device()
    layer = sae.layer
    component = getattr(sae, "component", "")
    position_name = getattr(sae, "position_name", "")
    n_clusters = sae.num_latents
    topk = sae.k
    name = sae.get_name()

    # initialize sae if needed
    if isinstance(sae, SAESpec):
        d_in = x_norm.shape[1]
        new_sae = SAE(d_in, n_clusters, k=topk).to(device)
        new_sae.layer = layer
        new_sae.component = component
        new_sae.position_name = position_name
        sae = new_sae

    lr = DEFAULT_LR_BASE / (n_clusters / DEFAULT_LR_SCALE_FACTOR) ** 0.5

    print(f"    Training {name}...", end=" ", flush=True)
    trained, loss, history = train_single_sae(
        sae,
        x_norm,
        lr=lr,
        device=device,
        max_epochs=max_epochs,
        patience=patience,
        tb_writer=tb_writer,
        tb_tag=f"{tb_prefix}{name}",
        tb_global_step=tb_global_step,
    )
    trained.layer = layer
    trained.component = component
    trained.position_name = position_name

    final_metric = history[-1] if history else {}
    dead_pct = final_metric.get("dead_latent_ratio", 0) * 100
    print(
        f"loss={loss:.4f}, L0={final_metric.get('l0_sparsity', 0):.2f}, dead={dead_pct:.1f}%"
    )

    results = {
        "name": name,
        "layer": layer,
        "component": component,
        "position_name": position_name,
        "n_clusters": n_clusters,
        "topk": topk,
        "loss": loss,
        "history": history,
    }
    return trained, results


# ── Feature extraction ─────────────────────────────────────────────────────


def get_sae_features_for_sentences(
    sae_model: SAE,
    sentences: list[dict],
    filter_sentence=None,
) -> tuple[torch.Tensor, list[dict]]:
    """Extract SAE feature activations for a list of sentence dicts.

    Assumes activations are already centered by section means.
    Returns (features_tensor, filtered_sentences) — only sentences that
    matched the filter and had activations for the SAE's layer.
    """
    layer_key = f"layer_{sae_model.layer}"

    vectors = []
    kept = []
    for s in sentences:
        if filter_sentence is not None and not filter_sentence(Sentence.from_dict(s)):
            continue
        act = s["activations"].get(layer_key)
        if act is None:
            continue
        vectors.append(act)
        kept.append(s)

    if not vectors:
        raise ValueError(f"No activations found for layer {sae_model.layer}")

    X_norm = np.stack(vectors, axis=0)

    device = get_device()
    X_tensor = torch.from_numpy(X_norm).float().to(device)
    sae_model = sae_model.to(device)
    with torch.no_grad():
        features = sae_model.get_all_activations(X_tensor)

    return features, kept
