import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import pauses as pa  # noqa: E402


def lomax_sample(rng, n, mu, tau0, x0):
    return np.round((tau0 + x0) * rng.random(n) ** (-1 / mu) - tau0)


def test_lomax_fit_recovers_known_parameters_and_beats_the_exponential():
    rng = np.random.default_rng(0)
    x = lomax_sample(rng, 40_000, 4.0, 900.0, 4.5)
    x0 = x.min() - 0.5
    u, c = pa.histogram(x)
    mu, tau0 = pa.fit_lomax(u, c, x0)
    assert abs(mu - 4.0) < 0.25 and abs(tau0 - 900) < 70
    aic_l = 4 - 2 * pa.lomax_loglik(u, c, mu, tau0, x0)
    aic_e = 2 - 2 * pa.exp_loglik(u, c, pa.fit_exp(u, c, x0), x0)
    assert aic_l < aic_e - 100


def test_exponential_data_prefer_the_exponential_and_theta_is_the_mean_excess():
    rng = np.random.default_rng(1)
    x = np.round(10.5 + rng.exponential(120, 20_000))
    x0 = x.min() - 0.5
    u, c = pa.histogram(x)
    assert abs(pa.fit_exp(u, c, x0) - 120) < 4
    mu, tau0 = pa.fit_lomax(u, c, x0)
    assert 4 - 2 * pa.lomax_loglik(u, c, mu, tau0, x0) > 2 - 2 * pa.exp_loglik(u, c, pa.fit_exp(u, c, x0), x0) - 0.5


def test_bootstrap_interval_holds_the_truth():
    rng = np.random.default_rng(2)
    x = lomax_sample(rng, 6000, 4.0, 900.0, 4.5)
    flight = rng.integers(0, 300, len(x))
    r = pa.fit_phase(x, flight, n_boot=60)
    assert r["mu_ci"][0] < 4.0 < r["mu_ci"][1] + 0.3 and r["lomax_better"] > 0.9 and r["n_flights"] <= 300


def test_blocks_join_search_and_climb_and_drop_censored_and_noncontiguous():
    rows = []
    t = 0
    for ph, n in [("transition", 10), ("search", 5), ("climb", 20), ("transition", 30), ("climb", 8), ("search", 4), ("transition", 6)]:
        rows.append(("f", 0, ph, t, t + n - 1, n, ph == "transition" and t == 0, ph == "transition" and n == 6))
        t += n
    r = pd.DataFrame(rows, columns=["flight_id", "segment_id", "phase", "t_start", "t_end", "n_fixes", "left_censored", "right_censored"])
    b = pa.blocks(r)
    assert b["kind"].tolist() == ["pause", "glide", "pause"] and b["duration_s"].tolist() == [25, 30, 12]
    r.loc[2, "t_start"] += 3  # a hole before the climb: the search and the climb are two pauses
    assert pa.blocks(r)["duration_s"].tolist() == [5, 20, 30, 12]
