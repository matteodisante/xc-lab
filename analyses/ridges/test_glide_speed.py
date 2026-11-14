import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import glide_speed as gs  # noqa: E402


def test_glide_steps_count_only_steps_inside_transition_runs(tmp_path):
    t = np.arange(20.0)
    rows = []
    for fid, speed in (("101", 10.0), ("102", 4.0)):
        for seg in (0, 1):
            rows.append(pd.DataFrame({"flight_id": fid, "segment_id": seg, "t": t, "E": speed * t, "N": 0.0}))
    pd.concat(rows).to_parquet(tmp_path / "fixes.parquet", row_group_size=30)
    runs = pd.DataFrame({"flight_id": ["101", "101", "102"], "segment_id": [0, 0, 1], "phase": ["transition", "climb", "transition"],
                         "t_start": [0.0, 10.0, 5.0], "t_end": [9.0, 19.0, 14.0]})
    r = gs.glide_steps(tmp_path / "fixes.parquet", runs, {"101", "102"}).set_index("flight_id")
    assert r.loc["101", "time_s"] == 9 and np.isclose(r.loc["101", "dist_m"], 90)
    assert r.loc["102", "time_s"] == 9 and np.isclose(r.loc["102", "dist_m"], 36)
