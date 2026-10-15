"""Load and sample TriviaQA questions for the experiment."""

from pathlib import Path
from typing import Any


def load_config(config_path: Path) -> dict[str, Any]:
    """Load the experiment configuration from YAML."""
    raise NotImplementedError


def load_questions(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Load and reproducibly sample questions from the configured dataset."""
    raise NotImplementedError


def save_questions(questions: list[dict[str, Any]], output_path: Path) -> None:
    """Save sampled questions as JSON Lines."""
    raise NotImplementedError


def main() -> None:
    """Run the dataset loading stage."""
    raise NotImplementedError


if __name__ == "__main__":
    main()

