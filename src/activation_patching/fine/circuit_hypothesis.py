"""Circuit hypothesis testing: test specific circuits via ablation and noise injection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ...common.base_schema import BaseSchema
from ...common.contrastive_pair import ContrastivePair
from ...inference.interventions import (
    zero_ablation_intervention,
    mean_ablation_intervention,
    gaussian_noise_intervention,
    compute_mean_activations,
)

if TYPE_CHECKING:
    from ...binary_choice import BinaryChoiceRunner


@dataclass
class CircuitHypothesis(BaseSchema):
    """Specifies a circuit to test via ablation.

    A circuit hypothesis identifies specific layers and positions that are
    hypothesized to be important for a model's behavior on a task.
    """

    layers: list[int]
    positions: list[int] | None = None  # None = all positions
    component: str = "resid_post"  # resid_post, attn_out, mlp_out


@dataclass
class CircuitTestResult(BaseSchema):
    """Result of testing a circuit hypothesis.

    Tests circuit necessity by measuring behavior change under different
    interventions (ablation, noise injection).
    """

    hypothesis: CircuitHypothesis

    # Baseline (clean run)
    baseline_logit_diff: float

    # Ablation results
    zero_ablation_logit_diff: float | None = None
    mean_ablation_logit_diff: float | None = None
    gaussian_noise_logit_diff: float | None = None

    # Recovery metrics: (ablated_diff / baseline_diff) * 100
    # 100% = behavior fully preserved (circuit not necessary)
    # 0% = behavior destroyed (circuit necessary)
    zero_ablation_recovery: float | None = None
    mean_ablation_recovery: float | None = None
    gaussian_noise_recovery: float | None = None

    def __post_init__(self):
        """Compute recovery percentages from logit diffs."""
        if self.baseline_logit_diff != 0:
            if self.zero_ablation_logit_diff is not None:
                self.zero_ablation_recovery = (
                    self.zero_ablation_logit_diff / self.baseline_logit_diff
                ) * 100
            if self.mean_ablation_logit_diff is not None:
                self.mean_ablation_recovery = (
                    self.mean_ablation_logit_diff / self.baseline_logit_diff
                ) * 100
            if self.gaussian_noise_logit_diff is not None:
                self.gaussian_noise_recovery = (
                    self.gaussian_noise_logit_diff / self.baseline_logit_diff
                ) * 100


def test_circuit_hypothesis(
    runner: "BinaryChoiceRunner",
    pair: ContrastivePair,
    hypothesis: CircuitHypothesis,
    test_zero: bool = True,
    test_mean: bool = True,
    test_noise: bool = True,
    noise_sigma: float = 1.0,
    mean_prompts: list[str] | None = None,
) -> CircuitTestResult:
    """Test a circuit hypothesis using ablation and noise injection.

    This function measures how much model behavior (measured by logit_diff)
    is preserved when the hypothesized circuit is ablated or corrupted.

    Args:
        runner: Model runner
        pair: Contrastive pair to test on
        hypothesis: Circuit specification (layers, positions, component)
        test_zero: Whether to test zero ablation
        test_mean: Whether to test mean ablation
        test_noise: Whether to test Gaussian noise injection
        noise_sigma: Standard deviation for Gaussian noise
        mean_prompts: Prompts to use for computing mean activations
                      (defaults to [clean_prompt] if None)

    Returns:
        CircuitTestResult with recovery metrics for each test type
    """
    d_model = runner._backend.get_d_model()

    # 1. Get baseline (clean run)
    baseline = runner.choose(pair.clean_prompt, pair.choice_prefix, pair.clean_labels)
    baseline_logit_diff = baseline.logit_diff

    # Initialize result
    zero_diff = None
    mean_diff = None
    noise_diff = None

    # 2. Test zero ablation
    if test_zero:
        interventions = [
            zero_ablation_intervention(
                layer=layer,
                d_model=d_model,
                positions=hypothesis.positions,
                component=hypothesis.component,
            )
            for layer in hypothesis.layers
        ]
        zero_result = runner.choose(
            pair.clean_prompt,
            pair.choice_prefix,
            pair.clean_labels,
            intervention=interventions,
        )
        zero_diff = zero_result.logit_diff

    # 3. Test mean ablation
    if test_mean:
        # Compute mean activations for each layer
        prompts = mean_prompts if mean_prompts else [pair.clean_prompt]
        interventions = []
        for layer in hypothesis.layers:
            mean_acts = compute_mean_activations(
                runner, layer, prompts, hypothesis.component
            )
            interventions.append(
                mean_ablation_intervention(
                    layer=layer,
                    mean_activations=mean_acts,
                    positions=hypothesis.positions,
                    component=hypothesis.component,
                )
            )
        mean_result = runner.choose(
            pair.clean_prompt,
            pair.choice_prefix,
            pair.clean_labels,
            intervention=interventions,
        )
        mean_diff = mean_result.logit_diff

    # 4. Test Gaussian noise
    if test_noise:
        interventions = [
            gaussian_noise_intervention(
                layer=layer,
                d_model=d_model,
                sigma=noise_sigma,
                positions=hypothesis.positions,
                component=hypothesis.component,
            )
            for layer in hypothesis.layers
        ]
        noise_result = runner.choose(
            pair.clean_prompt,
            pair.choice_prefix,
            pair.clean_labels,
            intervention=interventions,
        )
        noise_diff = noise_result.logit_diff

    return CircuitTestResult(
        hypothesis=hypothesis,
        baseline_logit_diff=baseline_logit_diff,
        zero_ablation_logit_diff=zero_diff,
        mean_ablation_logit_diff=mean_diff,
        gaussian_noise_logit_diff=noise_diff,
    )
