import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import decompose as dc  # noqa: E402


def test_decomposition_terms_add_up_to_one_hundred_percent():
    rng = np.random.default_rng(0)
    hg, hm, hr = (rng.uniform(a, b, 50) for a, b in ((0.92, 0.95), (0.85, 0.9), (0.86, 0.91)))
    d = dc.decomposition(np.r_[0.93, hg], np.r_[0.88, hm], np.r_[0.89, hr])
    assert abs(sum(d[k][0] for k in ("geometry", "intermittency", "other")) - 100) < 1e-9
    assert abs(d["total"][0] - 0.11) < 1e-12


def test_rule_starts_where_alpha_leaves_the_ballistic_value_and_recovers_the_slope():
    tau = np.geomspace(1, 1e4, 70)
    msd = np.where(tau < 100, tau**2, 1e4 * (tau / 100) ** 1.8)  # ballistic to 100 s, then H = 0.9
    lnm = np.log(msd)[None, :]
    h, start = dc.rule_fit(lnm, tau, floor=5.0, tau_max=5000.0)
    assert abs(h[0] - 0.9) < 0.01 and 90 < start[0] < 130
    h2, start2 = dc.rule_fit(lnm, tau, floor=300.0, tau_max=5000.0)  # the floor holds the start back
    assert start2[0] >= 300


def test_bootstrap_counts_are_whole_groups_and_row_zero_is_the_data():
    c = dc.draw_counts(20, 30, np.random.default_rng(1))
    assert (c[0] == 1).all() and (c[1:].sum(1) == 20).all()
    rows = np.tile(np.arange(1.0, 6.0), (20, 1))
    lnm = dc.log_means(rows, np.arange(20), c)
    assert np.allclose(lnm, np.log(np.arange(1.0, 6.0)))
