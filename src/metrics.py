"""Define correctness, calibration, and error-detection metrics."""

from typing import Sequence


def normalize_answer(text: str) -> str:
    """Normalize a short answer before exact-match and F1 scoring."""
    raise NotImplementedError


def exact_match(prediction: str, references: Sequence[str]) -> float:
    """Return whether the normalized prediction matches any reference."""
    raise NotImplementedError


def token_f1(prediction: str, references: Sequence[str]) -> float:
    """Return the maximum token-level F1 over reference answers."""
    raise NotImplementedError


def expected_calibration_error(
    confidences: Sequence[float],
    correctness: Sequence[int],
    n_bins: int,
) -> float:
    """Compute equal-width expected calibration error."""
    raise NotImplementedError


def brier_score(confidences: Sequence[float], correctness: Sequence[int]) -> float:
    """Compute the Brier score for binary answer correctness."""
    raise NotImplementedError


def error_detection_auroc(
    uncertainty_scores: Sequence[float],
    correctness: Sequence[int],
) -> float:
    """Compute AUROC when detecting incorrect model answers."""
    raise NotImplementedError


def bootstrap_interval(
    values: Sequence[float],
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    """Estimate a percentile bootstrap confidence interval."""
    raise NotImplementedError

