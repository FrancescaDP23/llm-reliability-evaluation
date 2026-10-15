"""Compute confidence measures from saved model generations."""

from pathlib import Path
from typing import Any


def sequence_confidence(token_log_probs: list[float]) -> float:
    """Compute length-normalized confidence for a generated sequence."""
    raise NotImplementedError


def mean_token_entropy(token_entropies: list[float]) -> float:
    """Compute mean predictive entropy across generated tokens."""
    raise NotImplementedError


def response_consistency(sampled_answers: list[str]) -> float:
    """Measure agreement among repeated sampled answers."""
    raise NotImplementedError


def compute_confidence_records(
    generations_path: Path,
) -> list[dict[str, Any]]:
    """Compute confidence features for all saved generations."""
    raise NotImplementedError


def save_confidence(records: list[dict[str, Any]], output_path: Path) -> None:
    """Save per-question confidence features as CSV."""
    raise NotImplementedError


def main() -> None:
    """Run the confidence scoring stage."""
    raise NotImplementedError


if __name__ == "__main__":
    main()

