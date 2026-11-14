import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import pauses as pa  # noqa: E402


def lomax_sample(rng, n, mu, tau0, x0):
    return np.round((tau0 + x0) * rng.random(n) ** (-1 / mu) - tau0)


def sample(name, params, n, seed):
    return pa.Law(name, params, 4.5).sample(np.random.default_rng(seed), n).astype(float)


def test_every_law_is_recovered_and_the_right_one_wins_by_aic():
    for name, params in (("lomax", (4.0, 900.0)), ("exponential", (150.0,)), ("weibull", (0.7, 120.0)), ("gamma", (2.5, 60.0))):
        x = sample(name, params, 60_000, 1)
        x0 = x.min() - 0.5
        u, c = pa.histogram(x)
        fits = pa.fit_all(u, c, x0)
        best = min(fits, key=lambda k: fits[k][1])
        assert best == name or (name == "exponential" and fits[name][1] - fits[best][1] < 3), (name, best)
        assert np.allclose(fits[best][0], params, rtol=0.15), (name, fits[best][0])


def test_law_survival_mean_and_quantiles_agree():
    for law in (pa.Law("lomax", (4.0, 700.0), 4.5), pa.Law("weibull", (0.7, 120.0), 4.5), pa.Law("gamma", (2.5, 60.0), 4.5), pa.Law("exponential", (90.0,), 4.5)):
        u = np.array([1.0, 0.5, 0.1, 0.01])
        assert np.allclose(law.surv(law.isf(u)), u, atol=1e-9), law.name
        x = law.isf(np.random.default_rng(0).random(300_000))
        assert abs(x.mean() - law.mean()) / law.mean() < 0.03, law.name


def test_bootstrap_interval_holds_the_truth_and_names_the_best_law():
    rng = np.random.default_rng(2)
    x = sample("weibull", (0.7, 120.0), 6000, 3)
    flight = rng.integers(0, 300, len(x))
    r = pa.fit_phase(x, flight, n_boot=40)
    assert r["best"] == "weibull" and r["params_ci"][0][0] < 0.7 < r["params_ci"][0][1] + 0.05 and r["wins"]["weibull"] > 0.8


def test_blocks_join_pause_runs_and_drop_censored_and_noncontiguous():
    rows = []
    t = 0
    for ph, n in [("transition", 10), ("search", 5), ("climb", 20), ("transition", 30), ("climb", 8), ("search", 4), ("transition", 6)]:
        rows.append(("f", 0, ph, t, t + n - 1, n, ph == "transition" and t == 0, ph == "transition" and n == 6))
        t += n
    r = pd.DataFrame(rows, columns=["flight_id", "segment_id", "phase", "t_start", "t_end", "n_fixes", "left_censored", "right_censored"])
    b = pa.blocks(r)  # a pause is a climb; the search runs are in neither phase
    assert b["kind"].tolist() == ["pause", "glide", "pause"] and b["duration_s"].tolist() == [20, 30, 8]
    old = pa.blocks(r, ("search", "climb"))  # search and climb joined
    assert old["kind"].tolist() == ["pause", "glide", "pause"] and old["duration_s"].tolist() == [25, 30, 12]
    r.loc[2, "t_start"] += 3  # a hole before the climb: the search and the climb are two pauses
    assert pa.blocks(r, ("search", "climb"))["duration_s"].tolist() == [5, 20, 30, 12]
