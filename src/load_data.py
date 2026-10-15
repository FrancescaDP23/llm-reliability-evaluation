"""Load and reproducibly sample TriviaQA questions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from datasets import Dataset, load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment.yaml"


def load_config(config_path: Path) -> dict[str, Any]:
    """Load and minimally validate the experiment configuration."""
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    required_sections = {"model", "dataset", "experiment", "outputs"}
    missing = required_sections.difference(config or {})
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")
    return config


def resolve_project_path(path_value: str) -> Path:
    """Resolve a configured path relative to the repository root."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _gold_aliases(answer: dict[str, Any]) -> list[str]:
    """Collect non-empty reference aliases while preserving their order."""
    candidates = [answer.get("value", ""), *answer.get("aliases", [])]
    return list(dict.fromkeys(str(item).strip() for item in candidates if str(item).strip()))


def load_questions(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Load and reproducibly sample questions from TriviaQA."""
    dataset_config = config["dataset"]
    experiment_config = config["experiment"]
    n_questions = int(experiment_config["n_questions"])
    seed = int(experiment_config["seed"])

    dataset = load_dataset(
        dataset_config["name"],
        dataset_config["config"],
        split=dataset_config["split"],
    )
    if not isinstance(dataset, Dataset):
        raise TypeError("Expected load_dataset to return a single Dataset split.")
    if n_questions <= 0:
        raise ValueError("n_questions must be positive.")
    if n_questions > len(dataset):
        raise ValueError(
            f"Requested {n_questions} questions, but the split contains {len(dataset)}."
        )

    # Shuffle before selecting so the same seed always produces the same subset.
    sampled = dataset.shuffle(seed=seed).select(range(n_questions))
    return [
        {
            "id": str(row["question_id"]),
            "question": str(row["question"]).strip(),
            "gold_aliases": _gold_aliases(row["answer"]),
        }
        for row in sampled
    ]


def save_questions(questions: list[dict[str, Any]], output_path: Path) -> None:
    """Save sampled questions as UTF-8 JSON Lines."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for question in questions:
            stream.write(json.dumps(question, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    """Parse the configuration path from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the YAML experiment configuration.",
    )
    return parser.parse_args()


def main() -> None:
    """Load, sample, and persist the configured TriviaQA subset."""
    args = parse_args()
    config = load_config(args.config)
    questions = load_questions(config)
    output_path = resolve_project_path(config["outputs"]["questions"])
    save_questions(questions, output_path)
    print(f"Saved {len(questions)} questions to {output_path}")


if __name__ == "__main__":
    main()
