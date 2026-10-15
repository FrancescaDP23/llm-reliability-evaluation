"""Evaluate saved predictions and confidence scores."""

from pathlib import Path
from typing import Any


def evaluate_predictions(
    generations_path: Path,
    confidence_path: Path,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score answers and compute aggregate reliability metrics."""
    raise NotImplementedError


def save_evaluation(
    records: list[dict[str, Any]],
    metrics: dict[str, Any],
    records_path: Path,
    metrics_path: Path,
) -> None:
    """Save per-question scores and aggregate metrics to disk."""
    raise NotImplementedError


def main() -> None:
    """Run evaluation using previously saved intermediate outputs."""
    raise NotImplementedError


if __name__ == "__main__":
    main()

