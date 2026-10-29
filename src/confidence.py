"""Compute and calibrate confidence measures from saved generations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn

from load_data import DEFAULT_CONFIG_PATH, load_config, resolve_project_path
from metrics import exact_match


class TemperatureScaler(nn.Module):
    """Learn one positive temperature for binary confidence logits."""

    def __init__(self) -> None:
        super().__init__()
        # Optimizing log(T) guarantees T remains strictly positive.
        self.log_temperature = nn.Parameter(torch.zeros(()))

    @property
    def temperature(self) -> torch.Tensor:
        """Return the positive scalar temperature."""
        return self.log_temperature.exp()

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by the learned temperature."""
        return logits / self.temperature


def mean_token_probability(token_log_probs: Sequence[float]) -> float:
    """Return the arithmetic mean probability of generated tokens."""
    if not token_log_probs:
        raise ValueError("At least one token log-probability is required.")
    return sum(math.exp(value) for value in token_log_probs) / len(token_log_probs)


def sequence_confidence(token_log_probs: Sequence[float]) -> float:
    """Return the geometric mean token probability for a sequence."""
    if not token_log_probs:
        raise ValueError("At least one token log-probability is required.")
    return math.exp(sum(token_log_probs) / len(token_log_probs))


def mean_token_entropy(token_entropies: Sequence[float]) -> float:
    """Compute mean predictive entropy across generated tokens."""
    if not token_entropies:
        raise ValueError("At least one token entropy is required.")
    return sum(token_entropies) / len(token_entropies)


def _probabilities_to_logits(probabilities: torch.Tensor) -> torch.Tensor:
    """Convert probabilities to finite binary logits."""
    epsilon = torch.finfo(probabilities.dtype).eps
    clipped = probabilities.clamp(epsilon, 1.0 - epsilon)
    return torch.logit(clipped)


def fit_temperature(
    confidences: Sequence[float],
    labels: Sequence[int],
    max_iter: int = 100,
) -> TemperatureScaler:
    """Fit a temperature by minimizing binary cross-entropy in PyTorch."""
    if len(confidences) != len(labels) or not confidences:
        raise ValueError("Confidences and labels must have the same non-zero length.")

    probabilities = torch.tensor(confidences, dtype=torch.float64)
    targets = torch.tensor(labels, dtype=torch.float64)
    logits = _probabilities_to_logits(probabilities)
    scaler = TemperatureScaler().double()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.LBFGS(
        scaler.parameters(),
        lr=0.1,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = criterion(scaler(logits), targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return scaler


def apply_temperature(
    confidences: Sequence[float], scaler: TemperatureScaler
) -> list[float]:
    """Apply a fitted scaler and return calibrated probabilities."""
    probabilities = torch.tensor(confidences, dtype=torch.float64)
    logits = _probabilities_to_logits(probabilities)
    with torch.no_grad():
        return torch.sigmoid(scaler(logits)).tolist()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read generation records from a JSON Lines file."""
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def compute_confidence_records(generations_path: Path) -> list[dict[str, Any]]:
    """Compute raw confidence features for all saved generations."""
    output: list[dict[str, Any]] = []
    for record in _read_jsonl(generations_path):
        tokens = record.get("tokens", [])
        if not tokens:
            raise ValueError(f"Generation {record.get('id')} has no scored tokens.")
        log_probs = [float(token["log_probability"]) for token in tokens]
        entropies = [float(token["entropy"]) for token in tokens]
        output.append(
            {
                "id": str(record["id"]),
                "question": str(record["question"]),
                "answer": str(record["answer"]),
                "is_correct": int(
                    exact_match(str(record["answer"]), record["gold_aliases"])
                ),
                "n_tokens": len(tokens),
                "mean_token_probability": mean_token_probability(log_probs),
                "sequence_confidence": sequence_confidence(log_probs),
                "mean_token_entropy": mean_token_entropy(entropies),
            }
        )
    return output


def calibrate_records(
    records: list[dict[str, Any]], split_ratio: float, seed: int, max_iter: int
) -> tuple[list[dict[str, Any]], float]:
    """Fit on a deterministic calibration split and annotate all records."""
    if len(records) < 2:
        raise ValueError("Temperature scaling requires at least two records.")
    if not 0.0 < split_ratio < 1.0:
        raise ValueError("calibration_split_ratio must be between zero and one.")

    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    split_size = min(max(round(len(records) * split_ratio), 1), len(records) - 1)
    calibration_indices = set(indices[:split_size])
    scaler = fit_temperature(
        [records[index]["sequence_confidence"] for index in calibration_indices],
        [records[index]["is_correct"] for index in calibration_indices],
        max_iter=max_iter,
    )
    calibrated = apply_temperature(
        [record["sequence_confidence"] for record in records], scaler
    )

    for index, (record, confidence) in enumerate(zip(records, calibrated, strict=True)):
        record["temperature_scaled_confidence"] = confidence
        record["split"] = "calibration" if index in calibration_indices else "evaluation"
    return records, float(scaler.temperature.detach())


def save_confidence(records: list[dict[str, Any]], output_path: Path) -> None:
    """Save per-question confidence features as CSV."""
    if not records:
        raise ValueError("No confidence records to save.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def parse_args() -> argparse.Namespace:
    """Parse the experiment configuration path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def main() -> None:
    """Compute raw and temperature-scaled confidence scores."""
    args = parse_args()
    config = load_config(args.config)
    generations_path = resolve_project_path(config["outputs"]["generations"])
    output_path = resolve_project_path(config["outputs"]["confidence"])
    records = compute_confidence_records(generations_path)
    records, temperature = calibrate_records(
        records,
        float(config["experiment"]["calibration_split_ratio"]),
        int(config["experiment"]["seed"]),
        int(config["experiment"]["temperature_max_iter"]),
    )
    save_confidence(records, output_path)
    print(f"Learned temperature: {temperature:.6f}")
    print(f"Saved {len(records)} confidence records to {output_path}")


if __name__ == "__main__":
    main()
