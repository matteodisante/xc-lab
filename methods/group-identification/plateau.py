"""The plateau check of the note on a sample of dates: for each setting of one parameter,
tau* from the null model, the fraction f_G of the flight time after departure spent in a
group (Eq. fg of the note), the number of episodes and of flights ever in a group.

    uv run python plateau.py [--sample N] [--jobs J]   # from this folder

Each setting is a run of run.py in trials/sample<N>/<name> on SSD_DISANTE (<name> from its
parameters); the table goes to plateau/sample<N>.json there.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

import run

# One parameter at a time around the starting values of the note.
SETTINGS = ["", "eps_h=250", "eps_h=1000", "eps_z=150", "eps_z=600", "delta=120", "delta=600",
            "theta_max=30", "theta_max=90", "gap=60", "gap=300"]


def f_g(membership):
    """Time in a group over the time after departure, summed over flights."""
    after = membership[membership["state"] != "L"]
    duration = after["t_end"] - after["t_start"]
    return float(duration[after["state"] == "G"].sum() / duration.sum())


def summarize(folder):
    calibration = json.loads((folder / "calibration.json").read_text())
    summary = pd.read_parquet(folder / "flight_summary.parquet")
    return {"run": folder.name, "tau_star": calibration["tau_star"],
            "f_G": f_g(pd.read_parquet(folder / "membership.parquet")),
            "episodes": len(pd.read_parquet(folder / "episodes.parquet")),
            "group_flights": int((summary["n_entries"] > 0).sum()), "flights": len(summary)}


def folder_of(setting, sample):
    p = dict(run.STARTING)
    for item in filter(None, setting.split(",")):
        key, value = item.split("=")
        p[key] = type(p[key])(float(value))
    p["r_min"] = 2 * p["delta"] + p["step"]
    return run.DATA / "trials" / f"sample{sample}" / run.run_name(p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=300)
    parser.add_argument("--jobs", type=int, default=5)
    args = parser.parse_args()
    table = run.DATA / "plateau" / f"sample{args.sample}.json"
    table.parent.mkdir(parents=True, exist_ok=True)
    rows = {}
    for setting in SETTINGS:
        folder = folder_of(setting, args.sample)
        if not (folder / "calibration.json").exists():
            subprocess.run([sys.executable, "run.py", "--sample", str(args.sample), "--jobs", str(args.jobs),
                            "--set", setting, "--out", str(folder)], check=True, cwd=Path(__file__).parent)
        rows[setting or "start"] = summarize(folder)
        table.write_text(json.dumps(rows, indent=1))
    print(pd.DataFrame(rows).T.to_string())
