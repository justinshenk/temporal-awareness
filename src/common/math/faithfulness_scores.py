"""Faithfulness score computation functions.

Implements the 2x2 matrix of faithfulness scores:

|                | IN-Circuit Patch     | OUT-Circuit Patch      |
|----------------|----------------------|------------------------|
| DENOISING      | Sufficiency          | Completeness           |
| (run corrupt,  | (recovery -> 1)      | (1 - recovery -> 1)    |
| patch clean)   |                      |                        |
|----------------|----------------------|------------------------|
| NOISING        | Necessity            | Independence           |
| (run clean,    | (disruption -> 1)    | (1 - disruption -> 1)  |
| patch corrupt) |                      |                        |
"""


def compute_recovery(y_intervened: float, y_clean: float, y_corrupted: float) -> float:
    """Raw recovery toward clean: R = (y_intervened - y_corrupted) / (y_clean - y_corrupted).

    Used for denoising experiments where we want output to move toward clean.
    """
    delta = y_clean - y_corrupted
    if abs(delta) < 1e-10:
        return 0.0
    return (y_intervened - y_corrupted) / delta


def compute_disruption(
    y_intervened: float, y_clean: float, y_corrupted: float
) -> float:
    """Raw disruption toward corrupt: D = (y_clean - y_intervened) / (y_clean - y_corrupted).

    Used for noising experiments where we expect output to move away from clean.
    Note: D = 1 - R (disruption and recovery are complements).
    """
    delta = y_clean - y_corrupted
    if abs(delta) < 1e-10:
        return 0.0
    return (y_clean - y_intervened) / delta


def compute_sufficiency_score(
    y_intervened: float, y_clean: float, y_corrupted: float
) -> float:
    """Sufficiency = Denoise In-Circuit -> recovery.

    Run corrupted input, patch IN-circuit with clean activations.
    High score (->1) means: circuit alone can recover the behavior.
    """
    return compute_recovery(y_intervened, y_clean, y_corrupted)


def compute_completeness_score(
    y_intervened: float, y_clean: float, y_corrupted: float
) -> float:
    """Completeness = Denoise Out-Circuit -> 1 - recovery = disruption.

    Run corrupted input, patch OUT-circuit with clean activations.
    High score (->1) means: out-circuit doesn't help recover (circuit is complete).

    We WANT low recovery from out-circuit, so we report disruption.
    """
    return compute_disruption(y_intervened, y_clean, y_corrupted)


def compute_necessity_score(
    y_intervened: float, y_clean: float, y_corrupted: float
) -> float:
    """Necessity = Noise In-Circuit -> disruption.

    Run clean input, patch IN-circuit with corrupted activations.
    High score (->1) means: corrupting circuit breaks behavior (it's necessary).
    """
    return compute_disruption(y_intervened, y_clean, y_corrupted)


def compute_independence_score(
    y_intervened: float, y_clean: float, y_corrupted: float
) -> float:
    """Independence = Noise Out-Circuit -> 1 - disruption = recovery.

    Run clean input, patch OUT-circuit with corrupted activations.
    High score (->1) means: corrupting out-circuit doesn't break behavior (circuit is independent).

    We WANT low disruption from out-circuit noise, so we report recovery.
    """
    return compute_recovery(y_intervened, y_clean, y_corrupted)


# Convenience functions for when you already have recovery computed
def sufficiency_from_recovery(recovery: float) -> float:
    """Sufficiency = recovery (for denoising IN-circuit)."""
    return recovery


def completeness_from_recovery(recovery: float) -> float:
    """Completeness = 1 - recovery (for denoising OUT-circuit)."""
    return 1.0 - recovery


def necessity_from_recovery(recovery: float) -> float:
    """Necessity = 1 - recovery = disruption (for noising IN-circuit)."""
    return 1.0 - recovery


def independence_from_recovery(recovery: float) -> float:
    """Independence = recovery (for noising OUT-circuit)."""
    return recovery
