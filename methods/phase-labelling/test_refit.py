import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import label  # noqa: E402
import phases  # noqa: E402
import refit  # noqa: E402

HIS = refit.HIS


def simulate(model, n, seed):
    rng = np.random.default_rng(seed)
    state = np.empty(n, dtype=int)
    state[0] = rng.choice(3, p=model["initial"])
    for k in range(1, n):
        state[k] = rng.choice(3, p=model["transition"][state[k - 1]])
    sym = np.array([rng.choice(8, p=model["emission"][s]) for s in state], dtype=np.int8)
    return sym, state


def track(seed=0, n=900):
    """A glide, a left-hand thermal and a search, at 1 Hz, east/north/up in metres."""
    rng = np.random.default_rng(seed)
    t = np.arange(float(n))
    e, u = 10 * t, -1.0 * t
    nn = np.zeros(n)
    th = (t >= 300) & (t < 600)
    s = t[th] - 300
    e[th] = 3000 + 30 * np.sin(2 * np.pi / 26 * s)
    nn[th] = 30 - 30 * np.cos(2 * np.pi / 26 * s)
    u[th] = -300 + 2.0 * s
    e[t >= 600] = e[th][-1] + 8 * (t[t >= 600] - 600)
    return t, np.vstack([e, nn, u]) + rng.normal(0, 0.5, (3, n))


def test_compiled_viterbi_is_the_reference_one():
    for seed in range(5):
        sym, _ = simulate(HIS, 3000, seed)
        assert (refit.viterbi_nb(sym, *HIS.values()) == phases.viterbi_symbols(sym, *HIS.values())).all()


def test_log_likelihood_matches_brute_force():
    sym, _ = simulate(HIS, 6, 1)
    total = 0.0
    for path in product(range(3), repeat=len(sym)):
        p = HIS["initial"][path[0]] * HIS["emission"][path[0], sym[0]]
        for k in range(1, len(sym)):
            p *= HIS["transition"][path[k - 1], path[k]] * HIS["emission"][path[k], sym[k]]
        total += p
    ll = refit.log_likelihood(sym, np.array([0, 6]), *HIS.values())
    assert ll == pytest.approx(np.log(total))
    assert refit.em_stats(sym, np.array([0, 6]), *HIS.values())[0] == pytest.approx(np.log(total))


def test_expected_counts_add_up():
    a, _ = simulate(HIS, 400, 2)
    b, _ = simulate(HIS, 300, 3)
    sym, off = np.r_[a, b], np.array([0, 400, 700])
    _, cb, ca, c0 = refit.em_stats(sym, off, *HIS.values())
    assert cb.sum() == pytest.approx(700) and ca.sum() == pytest.approx(698) and c0.sum() == pytest.approx(2)


def test_baum_welch_raises_the_likelihood_and_recovers_the_chain():
    truth = dict(emission=np.array([[.02, .02, .02, .9, .01, .01, .01, .01], [.4, .2, .3, .05, .02, .01, .01, .01],
                                    [.01, .01, .01, .01, .7, .05, .2, .01]]),
                 transition=np.array([[.97, .01, .02], [.02, .96, .02], [.01, .02, .97]]), initial=np.array([.3, .4, .3]))
    parts = [simulate(truth, 5000, s)[0] for s in range(6)]
    sym, off = np.concatenate(parts), np.arange(0, 30001, 5000)
    start = refit.random_start(np.random.default_rng(0))
    model, trace, _ = refit.baum_welch(sym, off, start, max_iter=150, rel_tol=1e-7)
    assert (np.diff(trace) > -1e-6).all()
    model = refit.permute(model, refit.identify(model["emission"]))
    assert np.abs(model["transition"] - truth["transition"]).max() < 0.02
    assert np.abs(model["emission"] - truth["emission"]).max() < 0.06


def test_states_are_identified_from_the_emissions():
    assert refit.identify(HIS["emission"]) == [0, 1, 2]
    p = [2, 0, 1]
    shuffled = refit.permute(HIS, p)
    again = refit.permute(shuffled, refit.identify(shuffled["emission"]))
    assert np.allclose(again["emission"], HIS["emission"]) and np.allclose(again["transition"], HIS["transition"])
    assert refit.meaning_ok(HIS["emission"])
    assert not refit.meaning_ok(HIS["emission"][[1, 0, 2]])


def test_runs_break_at_state_and_piece():
    state = np.array([0, 0, 0, 1, 1, -1, -1, 1, 1, 1, 1, 1])
    piece = np.array([0, 0, 0, 0, 0, -1, -1, 9, 9, 9, 9, 9])
    first, last, st, left, right = label.runs_of(state, piece)
    assert first.tolist() == [0, 3, 7] and last.tolist() == [2, 4, 11] and st.tolist() == [0, 1, 1]
    assert left.tolist() == [True, False, True] and right.tolist() == [False, True, True]


def test_runs_do_not_join_across_pieces_of_one_state():
    state = np.array([2, 2, 2, 2])
    first, last, _, left, right = label.runs_of(state, np.array([0, 0, 5, 5]))
    assert first.tolist() == [0, 2] and last.tolist() == [1, 3] and left.tolist() == [True, True] and right.tolist() == [True, True]


def test_label_segment_at_1hz_is_the_reference():
    t, xyz = track()
    state, piece = refit.label_segment(t, xyz, 1, phases.ALPHA["para"], HIS)
    ref, _ = phases.label(t, *xyz, phases.ALPHA["para"])
    assert (state == ref).all() and (piece == 0).all()


def test_label_segment_thinned_labels_real_fixes_only_and_agrees():
    t, xyz = track()
    ref = refit.label_segment(t, xyz, 1, phases.ALPHA["para"], HIS)[0]
    keep = (t % 5) == 0
    state, _ = refit.label_segment(t[keep], xyz[:, keep], 5, phases.ALPHA["para"], HIS)
    assert len(state) == keep.sum() and (state >= 0).all()
    assert (state == ref[keep]).mean() > 0.9


def test_pieces_shorter_than_the_window_stay_unlabelled():
    t = np.r_[np.arange(0., 100), np.arange(200., 220)]
    xyz = np.vstack([10 * t, np.zeros_like(t), -t])
    state, piece = refit.label_segment(t, xyz, 1, phases.ALPHA["para"], HIS)
    assert (state[:100] >= 0).all() and (state[100:] == -1).all() and (piece[100:] == -1).all()


def test_sample_quotas_are_equal_and_capped():
    assert refit.allocate([10, 100, 100], 60).tolist() == [10, 25, 25]
    assert refit.allocate([5, 5], 100).tolist() == [5, 5]


def test_sample_is_drawn_from_the_pool_without_repeats_and_spread_over_years():
    rng = np.random.default_rng(0)
    pool = pd.DataFrame(dict(flight_id=np.arange(3000), year=rng.choice([2015, 2020], 3000, p=[.2, .8]),
                             stratum=rng.choice(["a", "b", "c"], 3000)))
    s = refit.draw_sample(pool, 300, seed=1)
    assert len(s) == len(set(s)) == 300 and set(s) <= set(pool.flight_id)
    chosen = pool[pool.flight_id.isin(s)]
    assert chosen.stratum.value_counts().to_dict() == {"a": 100, "b": 100, "c": 100}
    assert abs((chosen.year == 2015).mean() - (pool.year == 2015).mean()) < 0.06


def test_cadence_summary_counts():
    conf = {f"c{a}{b}": 0 for a in range(3) for b in range(3)}
    row = lambda **kw: dict(model="m", k=2, flight_id=1, segment=0, compared=10, **(conf | kw))
    table = pd.DataFrame([row(c00=5, c11=3, c01=1, c22=1)])
    r = refit.cadence_summary(table)["m"][2]
    assert r["same_label"] == pytest.approx(0.9) and r["climb_kept"] == pytest.approx(5 / 6)
    assert r["climb_share_1hz"] == pytest.approx(0.6) and r["climb_share_thinned"] == pytest.approx(0.5)


@pytest.mark.skipif(not refit.DISCIPLINES["para"]["root"].exists(), reason="the data disk is not mounted")
def test_a_real_flight_decodes_like_the_reference():
    c = refit.candidates("para")
    f = int(c.flight_id.iloc[0])
    path = refit.DISCIPLINES["para"]["tracks"] / refit.catalogue("para").set_index("flight_id").local_path[f]
    flight, row = refit.read_flight(path, "para", refit.config())
    for _, t, xyz in refit.segments(flight):
        state, _ = refit.label_segment(t, xyz, 1, phases.ALPHA["para"], HIS)
        for grid, vals in refit.grids(t, xyz, 1):
            ref, _ = phases.label(grid, *vals, phases.ALPHA["para"])
            sel = (t >= grid[0]) & (t <= grid[-1])
            assert (state[sel] == ref[(t[sel] - grid[0]).astype(int)]).all()
