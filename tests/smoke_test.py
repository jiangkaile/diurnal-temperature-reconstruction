#!/usr/bin/env python3
"""Small deterministic smoke test; no external data are required."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from reconstruction_modes import baseline, harmonic_shape, one_anchor, scaled, two_anchor


def main():
    hours = np.arange(24)
    shape = harmonic_shape(hours, 4.0, 0.7, 1.1, -0.2)
    truth = 18.5 + 1.25 * shape
    assert np.allclose(baseline(shape, 18.5), 18.5 + shape)
    rec_one = one_anchor(shape, truth[12], 12)
    assert np.isclose(rec_one[12], truth[12], atol=1e-12)
    rec, alpha, beta = two_anchor(shape, truth[0], truth[12], 0, 12)
    assert np.allclose(rec, truth, atol=1e-12)
    assert np.isclose(alpha, 18.5) and np.isclose(beta, 1.25)
    assert np.isfinite(scaled(shape, 18.5, truth.min(), truth.max())).all()
    print("Smoke test passed: four reconstruction modes are executable.")


if __name__ == "__main__":
    main()
