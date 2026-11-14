import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import cleaned as cl  # noqa: E402


def make(tmp, monkeypatch, sizes, row_group=25):
    d = tmp / "x" / "derived"
    d.mkdir(parents=True)
    rows = [pd.DataFrame({"flight_id": str(100 + k), "segment_id": 0, "t": np.arange(n, dtype=float), "E": np.arange(n, dtype=float),
                          "N": 0.0, "z": 0.0}) for k, n in enumerate(sizes)]
    pd.concat(rows).to_parquet(d / "fixes.parquet", row_group_size=row_group)
    pd.DataFrame({"flight_id": [str(100 + k) for k in range(len(sizes))], "lat0": 45.0, "lon0": 6.0, "alt0": 1000.0, "drop_reason": None}).to_parquet(d / "flights_meta.parquet")
    monkeypatch.setattr(cl, "DISK", tmp)
    monkeypatch.setattr(cl, "FOLDERS", {"para": "x"})


def test_chunks_give_every_selected_flight_once_whole(tmp_path, monkeypatch):
    sizes = [30, 7, 60, 12, 41, 5, 90, 20, 33]  # flights cross row groups of 25 fixes, one covers three
    make(tmp_path, monkeypatch, sizes)
    ids = {str(100 + k) for k in (0, 2, 3, 6, 8)}
    n = cl.n_row_groups("para")
    for chunk in (1, 2, 3, n):
        got = [f for a in range(0, n, chunk) for f in cl.stream("para", ids, a, a + chunk)]
        assert sorted(f["flight"] for f in got) == sorted(ids), chunk
        assert all(len(f["t"]) == sizes[int(f["flight"]) - 100] and (np.diff(f["t"]) == 1).all() for f in got), chunk


def test_lambert_of_the_frame_origin_is_the_lambert_of_its_latlon(tmp_path):
    import flights as fl

    item = {"E": np.zeros(2), "N": np.zeros(2), "z": np.zeros(2), "frame": (45.0, 6.0, 0.0)}
    xy = cl.lambert(item)
    e, n = fl.lambert93(45.0, 6.0)
    assert np.hypot(xy[0, 0] - e, xy[0, 1] - n) < 1.0
