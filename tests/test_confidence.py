"""Tests for confidence computation and temperature scaling."""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from confidence import (  # noqa: E402
    TemperatureScaler,
    apply_temperature,
    fit_temperature,
    mean_token_probability,
    sequence_confidence,
)


def test_token_probability_summaries() -> None:
    """Compute arithmetic and geometric token-probability means."""
    log_probs = [math.log(0.8), math.log(0.2)]
    assert math.isclose(mean_token_probability(log_probs), 0.5)
    assert math.isclose(sequence_confidence(log_probs), 0.4)


def test_temperature_scaler_is_positive_and_changes_confidence() -> None:
    """Fit a finite positive temperature and apply it to probabilities."""
    raw = [0.95, 0.85, 0.70, 0.30, 0.15, 0.05]
    labels = [1, 0, 1, 0, 1, 0]
    scaler = fit_temperature(raw, labels)
    calibrated = apply_temperature(raw, scaler)

    assert isinstance(scaler, TemperatureScaler)
    assert torch.isfinite(scaler.temperature)
    assert float(scaler.temperature) > 0.0
    assert calibrated != raw
    assert all(0.0 <= value <= 1.0 for value in calibrated)
