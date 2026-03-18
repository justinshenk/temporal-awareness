"""Pyvene-based inference backend using IntervenableModel."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch

try:
    import pyvene as pv
    from pyvene import (
        IntervenableConfig,
        IntervenableModel,
        RepresentationConfig,
    )

    PYVENE_AVAILABLE = True
except ImportError:
    PYVENE_AVAILABLE = False

from .huggingface import HuggingFaceBackend
from ..interventions import Intervention


class PyveneBackend(HuggingFaceBackend):
    """Backend using pyvene's IntervenableModel for interventions.

    Inherits from HuggingFaceBackend for all non-intervention functionality,
    and overrides intervention methods to use pyvene's IntervenableModel.
    """

    def __init__(self, runner: Any, tokenizer: Any):
        if not PYVENE_AVAILABLE:
            raise ImportError(
                "pyvene is required for PyveneBackend. "
                "Install with: pip install pyvene"
            )
        super().__init__(runner, tokenizer)

    def _get_pyvene_component(self, layer_idx: int, component: str) -> str:
        """Convert component name to pyvene component path."""
        base = self._layers_attr

        if component in ("resid_post", "resid_mid"):
            return f"{base}[{layer_idx}].output"
        elif component == "resid_pre":
            return f"{base}[{layer_idx}].input"
        elif component == "mlp_out":
            return f"{base}[{layer_idx}].mlp.output"
        elif component == "attn_out":
            if self._layers_attr == "transformer.h":
                return f"{base}[{layer_idx}].attn.output"
            elif self._layers_attr == "gpt_neox.layers":
                return f"{base}[{layer_idx}].attention.output"
            else:
                return f"{base}[{layer_idx}].self_attn.output"
        else:
            raise ValueError(f"Unknown component: {component}")

    def _create_intervenable_model(
        self, interventions: Sequence[Intervention], seq_len: int
    ) -> tuple[IntervenableModel, list]:
        """Create an IntervenableModel configured for the given interventions.

        Returns:
            Tuple of (IntervenableModel, unit_locations for calling it)
        """
        configs = []
        all_positions = []

        for intervention in interventions:
            component_path = self._get_pyvene_component(
                intervention.layer, intervention.component
            )

            # Prepare the values tensor
            values = torch.tensor(
                intervention.scaled_values,
                dtype=self.runner.dtype,
                device=self.runner.device,
            )

            # Determine positions
            target = intervention.target
            if target.is_all_positions:
                positions = list(range(seq_len))
            else:
                positions = list(target.positions)

            # Get hidden size for proper broadcasting
            hidden_size = self.runner._model.config.hidden_size

            # Ensure values shape matches positions: [batch, num_pos, hidden]
            # Pyvene expects source_representation with batch dimension for multi-position
            if values.ndim == 0 or (values.ndim == 1 and values.shape[0] == 1):
                # Scalar value - expand to full hidden size, then to all positions
                scalar_val = values.item() if values.ndim == 0 else values[0].item()
                values = torch.full(
                    (len(positions), hidden_size),
                    scalar_val,
                    dtype=self.runner.dtype,
                    device=self.runner.device,
                )
            elif values.ndim == 1:
                # Single vector [hidden] - repeat for each position
                if values.shape[0] != hidden_size:
                    # If not matching hidden size, expand to hidden size
                    values = values.expand(hidden_size).contiguous()
                values = values.unsqueeze(0).expand(len(positions), -1).contiguous()
            elif values.ndim == 2:
                # Adjust to match positions
                if values.shape[0] > len(positions):
                    values = values[: len(positions)]
                elif values.shape[0] < len(positions):
                    pad_size = len(positions) - values.shape[0]
                    padding = values[-1:].expand(pad_size, -1)
                    values = torch.cat([values, padding], dim=0)

            # Add batch dimension: [num_pos, hidden] -> [1, num_pos, hidden]
            values = values.unsqueeze(0)

            # Create config with unit="pos" for position-specific targeting
            # Pass source_representation directly in RepresentationConfig for add/set
            source_repr = None
            if intervention.mode == "add":
                intervention_type = pv.AdditionIntervention
                source_repr = values
            elif intervention.mode == "set":
                intervention_type = pv.VanillaIntervention
                source_repr = values
            elif intervention.mode == "mul":
                # Pyvene doesn't have built-in multiply, use custom
                intervention_type = self._make_multiply_intervention(values)
            elif intervention.mode == "interpolate":
                target_values = torch.tensor(
                    intervention.target_values,
                    dtype=self.runner.dtype,
                    device=self.runner.device,
                )
                if target_values.ndim == 1:
                    target_values = (
                        target_values.unsqueeze(0)
                        .expand(len(positions), -1)
                        .contiguous()
                    )
                elif target_values.shape[0] != len(positions):
                    if target_values.shape[0] > len(positions):
                        target_values = target_values[: len(positions)]
                    else:
                        pad_size = len(positions) - target_values.shape[0]
                        padding = target_values[-1:].expand(pad_size, -1)
                        target_values = torch.cat([target_values, padding], dim=0)
                # Add batch dimension: [num_pos, hidden] -> [1, num_pos, hidden]
                target_values = target_values.unsqueeze(0)
                intervention_type = self._make_interpolate_intervention(
                    target_values, intervention.alpha
                )
            else:
                raise ValueError(f"Unknown intervention mode: {intervention.mode}")

            config = RepresentationConfig(
                layer=intervention.layer,
                component=component_path,
                unit="pos",
                intervention_type=intervention_type,
                source_representation=source_repr,
            )
            configs.append(config)
            all_positions.append(positions)

        intervenable_config = IntervenableConfig(representations=configs)
        intervenable = IntervenableModel(
            intervenable_config, model=self.runner._model
        )

        return intervenable, all_positions

    def _make_multiply_intervention(self, multiplier: torch.Tensor):
        """Create a custom multiply intervention class.

        Args:
            multiplier: Tensor with shape [1, num_positions, hidden] (batch dimension included)
        """
        mult_tensor = multiplier  # Capture in closure

        # Inherit from LocalistRepresentationIntervention to prevent activation flattening
        class MultiplyIntervention(
            pv.SourcelessIntervention, pv.LocalistRepresentationIntervention
        ):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.register_buffer("multiplier", mult_tensor)

            def forward(self, base, source=None, subspaces=None):
                # base: [batch, num_positions, hidden] when using unit="pos"
                # multiplier: [1, num_positions, hidden] - already has batch dimension
                return base * self.multiplier

        return MultiplyIntervention

    def _make_interpolate_intervention(
        self, target_values: torch.Tensor, alpha: float
    ):
        """Create a custom interpolation intervention class.

        Args:
            target_values: Tensor with shape [1, num_positions, hidden] (batch dimension included)
            alpha: Interpolation factor
        """
        target_tensor = target_values
        alpha_val = alpha

        # Inherit from LocalistRepresentationIntervention to prevent activation flattening
        class InterpolateIntervention(
            pv.SourcelessIntervention, pv.LocalistRepresentationIntervention
        ):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.register_buffer("target_values", target_tensor)
                self.alpha = alpha_val

            def forward(self, base, source=None, subspaces=None):
                # base: [batch, num_positions, hidden]
                # target_values: [1, num_positions, hidden] - already has batch dimension
                return base + self.alpha * (self.target_values - base)

        return InterpolateIntervention

    def _build_unit_locations(self, positions_list: list[list[int]]) -> dict:
        """Build unit_locations dict for pyvene.

        Pyvene expects unit_locations in format:
        - For single intervention with single position: {"base": position} or {"base": [position]}
        - For multiple interventions: {"base": [[[positions_0]], [[positions_1]], ...]}
          where each element is triple-nested: [[[positions_for_batch]]]
        """
        # For multiple interventions, use the triple-nested format
        # [[[positions_for_intervention_0]], [[positions_for_intervention_1]], ...]
        return {"base": [[[p for p in pos]] for pos in positions_list]}

    def run_with_intervention(
        self,
        input_ids: torch.Tensor,
        interventions: Sequence[Intervention],
    ) -> torch.Tensor:
        """Run forward with interventions using pyvene's IntervenableModel."""
        if not interventions:
            return self.forward(input_ids)

        seq_len = input_ids.shape[1]
        intervenable, positions_list = self._create_intervenable_model(
            interventions, seq_len
        )

        unit_locations = self._build_unit_locations(positions_list)

        with torch.no_grad():
            _, outputs = intervenable(
                {"input_ids": input_ids},
                unit_locations=unit_locations,
            )

        return outputs.logits

    def run_with_intervention_and_cache(
        self,
        input_ids: torch.Tensor,
        interventions: Sequence[Intervention],
        names_filter: Optional[callable],
    ) -> tuple[torch.Tensor, dict]:
        """Run forward with interventions AND capture activations.

        Note: Pyvene's IntervenableModel doesn't trigger standard PyTorch hooks,
        so we run the model twice - once for caching (direct call) and once for
        intervention (through pyvene). This is less efficient but necessary.
        """
        cache = {}
        hooks = []

        # Set up cache hooks
        hooks_to_capture = []
        for i in range(self._n_layers):
            for component in ["resid_pre", "resid_post", "attn_out", "mlp_out"]:
                name = f"blocks.{i}.hook_{component}"
                if names_filter is None or names_filter(name):
                    hooks_to_capture.append((i, component, name))

        def make_cache_hook(name):
            def hook(mod, input, output):
                if isinstance(output, tuple):
                    cache[name] = output[0].detach().clone()
                else:
                    cache[name] = output.detach().clone()

            return hook

        for layer_idx, component, name in hooks_to_capture:
            module = self._get_component_module(layer_idx, component)
            hook = module.register_forward_hook(make_cache_hook(name))
            hooks.append(hook)

        try:
            # Run model directly to capture activations (pyvene doesn't trigger hooks)
            with torch.no_grad():
                outputs = self.runner._model(input_ids)

            # Now run with intervention if needed
            if interventions:
                logits = self.run_with_intervention(input_ids, interventions)
            else:
                logits = outputs.logits
        finally:
            for hook in hooks:
                hook.remove()

        return logits, cache

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.0,
        intervention: Optional[Intervention] = None,
        past_kv_cache: Any = None,
    ) -> str:
        """Generate text, optionally with intervention using pyvene.

        Note: past_kv_cache is accepted for API compatibility but not used.
        """
        input_ids = self.encode(prompt)

        if intervention is not None:
            generated = input_ids.clone()
            eos_id = self._tokenizer.eos_token_id

            for _ in range(max_new_tokens):
                seq_len = generated.shape[1]
                intervenable, positions_list = self._create_intervenable_model(
                    [intervention], seq_len
                )
                unit_locations = self._build_unit_locations(positions_list)

                with torch.no_grad():
                    _, outputs = intervenable(
                        {"input_ids": generated},
                        unit_locations=unit_locations,
                    )
                    logits = outputs.logits

                if temperature > 0:
                    probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

                generated = torch.cat([generated, next_token], dim=1)

                if eos_id is not None and next_token.item() == eos_id:
                    break

            return self.decode(generated[0])
        else:
            # Use parent's generate for non-intervention case
            return super().generate(prompt, max_new_tokens, temperature, None, past_kv_cache)
