"""Copy the tables of one run into the data folder of xc-viewer, where its Group flights tab
reads them by default: <data folder>/groups/, replacing the run exported before.

    uv run python export_viewer.py RUN [DATA_FOLDER]   # DATA_FOLDER: /Volumes/SSD_DATA/xc-viewer-data

RUN is a run folder or its name under runs/; PARAMETERS.md goes with it, so the folder says
which delta, q and tau* its groups come from.
"""

import shutil
import sys
from pathlib import Path

import run

FILES = ("flights.parquet", "pair_intervals.parquet", "configurations.parquet", "membership.parquet",
         "episodes.parquet", "flight_summary.parquet", "episode_distance.parquet", "calibration.json",
         "PARAMETERS.md")
VIEWER_DATA = Path("/Volumes/SSD_DATA/xc-viewer-data")


def export(source, data_folder):
    """Copy FILES of `source` to data_folder/groups through a temporary folder."""
    missing = [name for name in FILES if not (source / name).is_file()]
    if missing:
        raise SystemExit(f"{source} has no {', '.join(missing)}")
    target, temporary = data_folder / "groups", data_folder / ".groups-export"
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir()
    for name in FILES:
        shutil.copy2(source / name, temporary / name)
    shutil.rmtree(target, ignore_errors=True)
    temporary.rename(target)
    return target


if __name__ == "__main__":
    source = Path(sys.argv[1])
    source = source if source.is_dir() else run.DATA / "runs" / sys.argv[1]
    data_folder = Path(sys.argv[2]) if len(sys.argv) > 2 else VIEWER_DATA
    print(f"{source.name} -> {export(source, data_folder)}")
