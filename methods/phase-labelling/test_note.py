import importlib.util
import math
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("phase_labelling_phases", HERE / "phases.py")
phases = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(phases)

REFERENCE = Path("/Users/matteodisante/Documents/university/LM/Tesi_LM/econophysics_paris/jérémie_work/"
                 "package_python_econophysix/flight_phase_inference_minimal.py.txt")
DISK = Path("/Volumes/SSD_DATA/paragliders/ffvl_cfd_igc")
OUT = HERE / "out"


def synthetic_track():
    """A sinking straight glide, a rising left-hand thermal, then a level search."""
    rng = np.random.default_rng(0)
    t = np.arange(0.0, 600.0)
    om, r = 2 * np.pi / 26, 30.0
    e, n, u = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)
    glide = t < 200
    e[glide], u[glide] = 10 * t[glide], -1.0 * t[glide]
    th = (t >= 200) & (t < 400)
    s = t[th] - 200
    e[th] = 2000 + r * np.sin(om * s)
    n[th] = r - r * np.cos(om * s)
    u[th] = -200 + 2.0 * s
    se = t >= 400
    s = t[se] - 400
    # Search: tight turns, 0.4 rad/s, alternately left and right every 10 s, at 8 m/s.
    heading = np.cumsum(np.where((s // 10) % 2 == 0, 0.4, -0.4))
    e[se] = e[th][-1] + np.cumsum(8 * np.cos(heading))
    n[se] = n[th][-1] + np.cumsum(8 * np.sin(heading))
    u[se] = 200 - 0.3 * s
    return t, e + rng.normal(0, 0.5, t.size), n + rng.normal(0, 0.5, t.size), u + rng.normal(0, 0.3, t.size)


# Section 2: the small example and the chain.

def test_small_example_viterbi_keeps_the_climb_through_one_sink():
    emission = np.array([[0.1, 0.9], [0.8, 0.2]])
    transition = np.array([[0.9, 0.1], [0.1, 0.9]])
    x = np.array([1, 1, 0, 1, 1, 0, 0, 0])
    path = phases.viterbi_symbols(x, emission, transition, np.array([0.5, 0.5]))
    assert path.tolist() == [0, 0, 0, 0, 0, 1, 1, 1]
    assert emission[:, x].argmax(axis=0).tolist() == [0, 0, 1, 0, 0, 1, 1, 1]
    assert (0.9 * 0.1 * 0.9) / (0.1 * 0.8 * 0.1) == pytest.approx(10.125)


def test_small_example_trellis_values():
    emission = np.array([[0.1, 0.9], [0.8, 0.2]])
    transition = np.array([[0.9, 0.1], [0.1, 0.9]])
    delta = np.array([0.5, 0.5]) * emission[:, 1]
    columns = [delta / delta.sum()]
    for o in [1, 0, 1, 1, 0, 0, 0]:
        delta = (columns[-1][:, None] * transition).max(axis=0) * emission[:, o]
        columns.append(delta / delta.sum())
    climb = [round(c[0], 3) for c in columns]
    assert climb == [0.818, 0.953, 0.529, 0.835, 0.958, 0.529, 0.123, 0.017]


def test_mean_durations_of_the_chain():
    durations = 1 / (1 - np.diag(phases.TRANSITION))
    assert np.round(durations).tolist() == [96, 100, 123]
    assert 2 * np.pi / 26 == pytest.approx(0.24, abs=0.005)


# Sections 3 and 4: the fitted model.

def test_emissions_name_the_states():
    assert np.allclose(phases.EMISSION.sum(axis=1), 1, atol=1e-6)
    assert phases.EMISSION[0].argmax() == 3                     # climb: (0,1,1)
    assert phases.EMISSION[2, 4:].sum() > 0.99                   # transition: straight symbols
    assert phases.EMISSION[1, [0, 1, 2]].sum() > 0.99            # search: turning symbols


def test_transition_matrix_carries_the_constraint_signature():
    a = phases.TRANSITION
    assert a[0, 2] / a[0, 1] == pytest.approx(2, rel=1e-5)
    assert a[1, 0] / a[1, 2] == pytest.approx(2, rel=1e-5)
    assert a[1, 1] == 0.99


def test_climb_symbols_needed_to_turn_the_viterbi_path():
    def climbs(background, n):
        x = np.array([background] * 200 + [3] * n + [background] * 200)
        path = phases.viterbi_symbols(x, phases.EMISSION, phases.TRANSITION, phases.INITIAL)
        return bool((path[200:200 + n] == 0).all())
    assert climbs(4, 1)                                         # inside a glide, one (0,1,1)
    assert not climbs(0, 1) and climbs(0, 2)                    # inside a search, two in a row


def test_height_above_the_tangent_plane():
    r = 6.371e6
    assert round(100e3 ** 2 / (2 * r)) == 785
    assert round(100e3 / r * 10.0, 2) == 0.16


@pytest.mark.skipif(not (OUT / "example.npz").exists(), reason="run cadence.py first")
def test_the_altitude_instead_of_the_tangent_height_changes_no_label_on_the_saved_flight():
    d = np.load(OUT / "example.npz")
    t, (e, n, u) = d["t"], d["xyz"]
    dist = np.hypot(e, n)
    assert 10e3 < dist.max() < 11e3
    z = u + dist ** 2 / (2 * 6.371e6)
    changed = 0
    for (g, v), (_, w) in zip(phases.regular_grid(t, np.vstack([e, n, u]), 4.0),
                              phases.regular_grid(t, np.vstack([e, n, z]), 4.0)):
        if len(g) >= phases.WINDOW:
            changed += int((phases.label(g, *v, phases.ALPHA["para"])[0]
                            != phases.label(g, *w, phases.ALPHA["para"])[0]).sum())
    assert changed == 0


def test_window_figure():
    spec = importlib.util.spec_from_file_location("phase_labelling_plots", HERE / "plots.py")
    plots = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(HERE))
    spec.loader.exec_module(plots)
    alpha = phases.ALPHA["para"]
    t, xyz = plots.synthetic_flight()
    obs, turn, vz = phases.features(t, *xyz, alpha)
    answers = {}
    for name, k in plots.WINDOWS.items():
        _, _, below, left, right, _ = plots.window_answers(turn, vz, k, alpha)
        answers[name] = (below, left, right)
        assert tuple(obs[k]) == {"glide": (1, 0, 0), "search": (0, 0, 0), "climb": (0, 1, 1)}[name]
    assert answers["glide"][0] == 30
    assert answers["search"][1] > 0 and answers["search"][2] > 0
    assert answers["climb"][1:] == (15, 0)


def test_parameter_count():
    assert 3 * 8 + 3 * 3 + 3 == 36
    assert 3 * 7 + 3 * 2 + 2 == 29


def test_features_and_labels_on_a_synthetic_track():
    t, e, n, u = synthetic_track()
    obs, _, _ = phases.features(t, e, n, u, phases.ALPHA["para"])
    symbol = obs @ [4, 2, 1]
    assert np.mean(symbol[50:150] == 4) > 0.9                   # glide: (1,0,0)
    assert np.mean(symbol[250:350] == 3) > 0.9                  # thermal: (0,1,1)
    final, _ = phases.label(t, e, n, u, phases.ALPHA["para"])
    assert np.mean(final[50:150] == 2) > 0.9
    assert np.mean(final[250:350] == 0) > 0.9
    assert np.mean(final[450:550] == 1) > 0.9


@pytest.mark.skipif(not REFERENCE.exists(), reason="Vilpellet's reference script is not on this machine")
def test_same_labels_as_the_reference_script():
    sys.modules.setdefault("numba", types.SimpleNamespace(njit=lambda *a, **k: a[0] if a and callable(a[0]) else (lambda f: f)))
    source = REFERENCE.read_text()
    ns = {}
    exec(source[: source.index("def get_regime")], ns)
    t, e, n, u = synthetic_track()
    lat0, lon0 = 45.0, 6.0
    lat = lat0 + n / 111_132.0
    lon = lon0 + e / (111_320.0 * math.cos(math.radians(lat0)))
    track = pd.DataFrame({"time": t, "Lat": lat, "Long": lon, "AltGNSS": 1500 + u})
    obs_ref = ns["factory_data_HMM"].get_data_hmm(raw_data=track, window_smooth_data=6,
                                                  alpha_straight_rad=phases.ALPHA["para"],
                                                  window_persitence=30, beta_persistence=0.9)
    path_ref = ns["hmm"].predict_states_viterbi(obs_ref)
    final_ref = ns["roll_majority"](pd.Series(path_ref), 90).to_numpy()
    xyz = phases.enu(lat, lon, 1500 + u)
    obs, _, _ = phases.features(t, *xyz, phases.ALPHA["para"])
    final, path = phases.label(t, *xyz, phases.ALPHA["para"])
    assert np.array_equal(obs, obs_ref)
    assert np.array_equal(path, path_ref)
    assert np.array_equal(final, final_ref)


# Section 1: logging intervals.

@pytest.mark.skipif(not DISK.exists(), reason="SSD_DATA is not mounted")
def test_logging_interval_table():
    expected = {"paragliders": (185_321, [69.6, 9.5, 5.1, 2.2, 9.7, 2.4]),
                "hang_gliders/delta_cfd_igc": (6_536, [30.4, 13.5, 9.6, 3.0, 26.6, 15.2])}
    for folder, (count, shares) in expected.items():
        root = DISK if folder == "paragliders" else DISK.parents[1] / folder
        v = pd.read_csv(root / "derived/preprocessing/vertical/flights.csv")
        ok = v[v.gnss_ok.astype(bool) & v.speed_ok.fillna(False).astype(bool)]
        assert len(ok) == count
        got = [round(100 * np.mean(ok.sampling_s == s), 1) for s in (1.0, 2.0, 3.0, 4.0, 5.0, 10.0)]
        assert got == shares


# Section 6: the measurement, against the files cadence.py, mechanism.py and plots.py wrote.

def aggregated():
    d = pd.read_csv(OUT / "cadence.csv")
    cols = [f"c{a}{b}" for a in range(3) for b in range(3)]
    per = d.groupby(["flight_id", "k", "cubic"])[cols].sum()
    m = per.to_numpy().reshape(-1, 3, 3)
    per = per.assign(agree=np.trace(m, axis1=1, axis2=2) / m.sum(axis=(1, 2)))
    return d, d.groupby(["k", "cubic"])[cols].sum(), per


@pytest.mark.skipif(not (OUT / "cadence.csv").exists(), reason="run cadence.py first")
def test_cadence_table():
    d, tot, per = aggregated()
    assert d.flight_id.nunique() == 300
    table = {  # (same label, climb kept, search kept, climb share, worst 10 %), in percent
        (2, False): (98.9, 99.0, 95.8, 33.3, 97.0), (2, True): (99.0, 99.0, 97.2, 33.3, 97.2),
        (3, False): (98.2, 98.2, 92.7, 33.2, 95.6), (3, True): (98.8, 98.8, 96.1, 33.3, 96.4),
        (4, False): (97.1, 96.4, 89.4, 32.8, 94.1), (4, True): (98.3, 98.3, 94.1, 33.3, 96.1),
        (5, False): (94.8, 90.7, 86.2, 31.1, 89.3), (5, True): (97.5, 96.9, 91.3, 33.0, 94.7),
        (10, False): (73.2, 33.1, 81.7, 12.6, 58.7), (10, True): (81.4, 57.0, 71.6, 20.8, 69.6),
    }
    fixes = {2: 1_683_979, 3: 1_125_818, 4: 844_257, 5: 676_249, 10: 338_359}
    for key, row in table.items():
        c = tot.loc[key].to_numpy().reshape(3, 3)
        worst = per.xs(key, level=("k", "cubic")).agree.quantile(0.1)
        got = (np.trace(c) / c.sum(), c[0, 0] / c[0].sum(), c[1, 1] / c[1].sum(), c[:, 0].sum() / c.sum(), worst)
        assert [round(100 * g, 1) for g in got] == list(row), key
        assert round(100 * c[0].sum() / c.sum(), 1) == 33.3
        assert c.sum() == fixes[key[0]]
    shares = tot.loc[(5, True)].to_numpy().reshape(3, 3)
    assert np.abs(shares.sum(axis=0) - shares.sum(axis=1)).max() / shares.sum() < 0.005
    # The flight of the figures, at 5 s: in the worst tenth with linear interpolation.
    example = per.xs(20356685, level="flight_id").agree
    assert round(100 * example.loc[(5, False)]) == 87 and round(100 * example.loc[(5, True)], 1) == 98.5
    assert example.loc[(5, False)] < per.xs((5, False), level=("k", "cubic")).agree.quantile(0.1)


@pytest.mark.skipif(not (OUT / "mechanism.yaml").exists(), reason="run mechanism.py first")
def test_mechanism_numbers():
    m = yaml.safe_load((OUT / "mechanism.yaml").read_text())
    assert round(100 * m["climb_fixes_011_to_010"]["linear"], 1) == 5.6
    assert round(100 * m["climb_fixes_011_to_010"]["cubic"], 1) == 1.2
    w = m["climb_windows_two_minority"]
    assert [round(100 * w[k], 1) for k in ("native", "linear", "cubic")] == [8.4, 12.9, 8.9]
    h5, h10 = m["climb_heading_change_rad"][5], m["climb_heading_change_rad"][10]
    assert round(h5["median"], 2) == 1.26 and round(math.degrees(h5["median"])) == 72
    assert round(100 * h5["above_2"]) == 15 and round(100 * h5["above_2.5"]) == 5
    assert round(h10["median"], 2) == 2.10 and 0.30 < h10["above_2.5"] < 0.34


@pytest.mark.skipif(not (OUT / "turn_shares.yaml").exists(), reason="run plots.py first")
def test_turn_shares_below_alpha():
    s = yaml.safe_load((OUT / "turn_shares.yaml").read_text())
    assert [round(100 * s[k]) for k in ("transition", "search", "climb")] == [98, 66, 37]


@pytest.mark.skipif(not (REFERENCE.exists() and DISK.exists() and (OUT / "cadence.csv").exists()),
                    reason="needs the reference script, the disk and out/cadence.csv")
def test_same_labels_as_the_reference_script_on_three_flights():
    from xc_dynamics.preprocessing.flight import preprocess
    sys.modules.setdefault("numba", types.SimpleNamespace(njit=lambda *a, **k: a[0] if a and callable(a[0]) else (lambda f: f)))
    source = REFERENCE.read_text()
    ns = {}
    exec(source[: source.index("def get_regime")], ns)
    cfg = yaml.safe_load((HERE.parents[2] / "xc-dynamics/configs/preprocessing.yaml").read_text())
    paths = pd.read_csv(DISK / "catalog/catalog.csv", usecols=["flight_id", "local_path"]).set_index("flight_id").local_path
    for fid in pd.read_csv(OUT / "cadence.csv").flight_id.unique()[:3]:
        fixes, _ = preprocess(Path("/Volumes/SSD_DATA/xc-viewer-data/paragliders") / paths[fid], cfg["altitude"],
                              cfg["speed"]["median_max_kmh"]["para"], cfg["speed"]["max_kmh"]["para"],
                              cfg["excursion"]["window_s"], cfg["vertical"]["max_ms"]["para"],
                              cfg["airborne"]["min_kmh"], cfg["airborne"]["min_s"])
        seg = fixes[fixes.segment == fixes.segment.value_counts().idxmax()]
        track = pd.DataFrame({"time": seg.t.to_numpy(float), "Lat": seg.lat.to_numpy(),
                              "Long": seg.lon.to_numpy(), "AltGNSS": seg.gnss_alt.to_numpy()})
        obs_ref = ns["factory_data_HMM"].get_data_hmm(raw_data=track, window_smooth_data=6,
                                                      alpha_straight_rad=phases.ALPHA["para"],
                                                      window_persitence=30, beta_persistence=0.9)
        path_ref = ns["hmm"].predict_states_viterbi(obs_ref)
        final_ref = ns["roll_majority"](pd.Series(path_ref), 90).to_numpy()
        xyz = phases.enu(track.Lat.to_numpy(), track.Long.to_numpy(), track.AltGNSS.to_numpy())
        final, path = phases.label(track.time.to_numpy(), *xyz, phases.ALPHA["para"])
        assert np.array_equal(path, path_ref) and np.array_equal(final, final_ref), fid
