"""The facts of the grouping that the report quotes, from the tables of one run: out/<run>/facts.json.

    uv run python report/facts.py [--run NAME]    # from analyses/groups/group-identification
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from tamsd import H_MAX, LABELS, OUT, SIZES, size_of  # noqa: E402

if __name__ == "__main__":
    flights = pd.read_parquet(LABELS / "flights.parquet")
    summary = pd.read_parquet(LABELS / "flight_summary.parquet").set_index("flight")
    membership = pd.read_parquet(LABELS / "membership.parquet")
    groups = pd.read_parquet(LABELS / "groups.parquet")
    episodes = pd.read_parquet(LABELS / "episodes.parquet")
    pairs = pd.read_parquet(LABELS / "pair_intervals.parquet")
    calibration = json.loads((LABELS / "calibration.json").read_text())
    eligible = flights[(flights["dt_native"] <= H_MAX + 1e-6) & (flights["n_grid"] > 0)]
    facts = {"run": LABELS.name, "tau_star_s": calibration["tau_star"], "dates": int(flights["date"].nunique()),
             "flights": flights.groupby("discipline").size().to_dict(), "pair intervals": len(pairs),
             "groups": len(groups), "episodes": len(episodes)}
    discipline = flights.set_index("flight")["discipline"]
    for d in ("para", "hang"):
        e = eligible.loc[eligible["discipline"] == d, "flight"]
        mine = membership[membership["flight"].isin(e)]
        after = mine[mine["state"] != "L"]
        hours = (after["t_end"] - after["t_start"]).groupby(after["state"]).sum() / 3600
        grouped = summary["n_entries"].reindex(e).gt(0)
        g = mine[mine["state"] == "G"].assign(size=lambda m: size_of(m["n"].to_numpy()))
        by_size = ((g["t_end"] - g["t_start"]).groupby(g["size"]).sum() / 3600).reindex(list(SIZES)).fillna(0)
        own = groups[groups["members"].map(lambda m: discipline.get(int(m[0])) == d)]
        facts[d] = {"eligible flights": int(len(e)), "flights ever in a group": int(grouped.sum()),
                    "share ever in a group": float(grouped.mean()),
                    "hours after departure": float(hours.sum()), "hours in a group": float(hours.get("G", 0)),
                    "share of the time in a group, all flights": float(hours.get("G", 0) / hours.sum()),
                    "share of the time in a group, group flights": float(
                        summary.loc[grouped[grouped].index, "phi_G"].mean()),
                    "group hours by size": by_size.round(1).to_dict(),
                    "groups by size": own["n"].map(lambda n: str(size_of(np.array([n]))[0])).value_counts()
                    .reindex(list(SIZES)).fillna(0).astype(int).to_dict(),
                    "largest group": int(own["n"].max())}
    duration = pairs["duration"] / 60
    facts["pair interval minutes (quantiles 10, 50, 90)"] = np.percentile(duration, [10, 50, 90]).round(1).tolist()
    facts["episode flights (quantiles 50, 90, 99, max)"] = [*np.percentile(episodes["n_flights"], [50, 90, 99]).tolist(),
                                                           int(episodes["n_flights"].max())]
    facts["episode hours (quantiles 50, 90)"] = (np.percentile(episodes["t_end"] - episodes["t_start"], [50, 90]) / 3600
                                                ).round(2).tolist()
    facts["|ell_med| seconds (quantiles 25, 50, 75, 90)"] = np.nanpercentile(pairs["ell_med"].abs(), [25, 50, 75, 90]).tolist()
    facts["pair intervals with takeoffs more than 10 km apart"] = float((pairs["d_dec"] >= 10_000).mean())
    (OUT / "facts.json").write_text(json.dumps(facts, indent=1))
    print(json.dumps(facts, indent=1))
