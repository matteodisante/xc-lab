"""Synthetic trajectories for the tests: fractional Brownian motion of known H on the 10-s grid."""

from functools import cache

import numpy as np
import pandas as pd


@cache
def _factor(n, h):
    k = np.arange(n)
    gamma = 0.5 * (np.abs(k + 1) ** (2 * h) - 2 * np.abs(k) ** (2 * h) + np.abs(k - 1) ** (2 * h))
    return np.linalg.cholesky(gamma[np.abs(k[:, None] - k[None, :])] + 1e-12 * np.eye(n))


def fbm_fixes(n, h, rng, step=10.0, scale=10.0):
    """The fixes of a flight whose E and N are independent fBm of exponent h, every `step` s, one segment."""
    e, north = (np.cumsum(_factor(n, h) @ rng.standard_normal(n)) * scale for _ in range(2))
    return pd.DataFrame({"segment_id": 0, "t": np.arange(n) * step, "E": e, "N": north, "z": 1000.0})
