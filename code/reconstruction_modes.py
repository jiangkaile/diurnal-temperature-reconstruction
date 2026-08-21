#!/usr/bin/env python3
"""Portable implementations of the four reconstruction modes used in the study."""

from __future__ import annotations

import numpy as np


def harmonic_shape(hours, a1, phi1, a2, phi2):
    """Return the zero-mean 24 h + 12 h harmonic shape."""
    hours = np.asarray(hours, dtype=float)
    omega = 2.0 * np.pi / 24.0
    return a1 * np.sin(omega * hours - phi1) + a2 * np.sin(2.0 * omega * hours - phi2)


def baseline(shape, daily_mean):
    return np.asarray(shape, dtype=float) + float(daily_mean)


def scaled(shape, daily_mean, daily_min, daily_max, epsilon=1e-4):
    shape = np.asarray(shape, dtype=float)
    shape_range = np.nanmax(shape) - np.nanmin(shape)
    beta = (float(daily_max) - float(daily_min)) / shape_range if shape_range > epsilon else 1.0
    return float(daily_mean) + beta * shape


def one_anchor(shape, observed_anchor, anchor_hour=12):
    shape = np.asarray(shape, dtype=float)
    alpha = float(observed_anchor) - shape[int(anchor_hour)]
    return alpha + shape


def two_anchor(shape, observed_h1, observed_h2, h1=0, h2=12,
               delta=0.5, beta_bounds=(0.1, 5.0)):
    shape = np.asarray(shape, dtype=float)
    ds = shape[int(h2)] - shape[int(h1)]
    if abs(ds) <= delta:
        beta = 1.0
        alpha = ((float(observed_h1) - shape[int(h1)]) +
                 (float(observed_h2) - shape[int(h2)])) / 2.0
    else:
        beta = (float(observed_h2) - float(observed_h1)) / ds
        beta = float(np.clip(beta, beta_bounds[0], beta_bounds[1]))
        alpha = float(observed_h2) - beta * shape[int(h2)]
    return alpha + beta * shape, alpha, beta
