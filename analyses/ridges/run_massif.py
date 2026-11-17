"""One command per massif: DEM -> H_geo on the 5 m grid (or 25 m), every stage in its own process, resumable.

    uv run --with numba python run_massif.py alps [--dem 5] [--tile 2500] [--no-dimension]
    uv run --with numba python run_massif.py alps --window alps1      # trial on a 35 x 35 km window of dem_resolution.py
    uv run --with numba python run_massif.py alps --window 940,1040,6420,6520   # or on a box in km (mask: every cell counts)

Stages, in order (each one a child process under `/usr/bin/time -l`, so its peak memory is its own):
  tiled-dem ... tiled-edges   run_tiled.py: DEM raster, mountain area, routing, crest sides (mask_, edges_, hand_)
  curves                      build.py --stage curves
  hgeo                        hgeo.py: joins, trees, paths, TAMSD, hgeo_windows.csv, paths_, joins_
  paths-all, dim-crests, dim-network   export_paths.py and run_dimension.py (inputs of the box dimension)
  clean                       run_tiled.py --stage clean: the big rasters of the tiled run (the products stay)
`driver_<region>.json` in the parameter folder marks the finished stages (seconds, peak memory); run the same command
again after a crash and it goes on (the tiled stages also resume tile by tile). `run.log` has every line of every stage
with a timestamp. The folder is CACHE/<rgealti5_..._flats-blm>/ (CACHE = RIDGES_CACHE, default below), the
window folder CACHE/trial_<window>/<...>/. Run one massif at a time: the Alps need about 70 GB of disk at the peak.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = "/Volumes/SSD_DATA/xc-lab/ridges"
MIN_FREE_GB = 90  # a full massif refuses to start below this
HERE = Path(__file__).parent
PEAK = re.compile(r"(\d+)\s+maximum resident set size")
STATS = re.compile(r"^\s+[\d.]+\s+(real\b|maximum|average|page|swaps|block|messages|signals|voluntary|involuntary|instructions|cycles|peak)")  # /usr/bin/time -l


def stamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def set_environment(dem, window):
    """RIDGES_FLATS, RIDGES_DEM and RIDGES_CACHE of every child; set before the modules read them."""
    os.environ["RIDGES_FLATS"] = "blm"
    os.environ["RIDGES_DEM"] = str(dem)
    os.environ.setdefault("RIDGES_CACHE", ROOT + (f"/trial_{window.replace(',', '-')}" if window else ""))


def stages(region, dem, tile, window, dimension):
    py = [sys.executable, "-u"]
    rt = py + [str(HERE / "run_tiled.py"), region, "--dem", str(dem), "--tile", str(tile)] + (["--window", window] if window else [])
    out = [(f"tiled-{s}", rt + ["--stage", s]) for s in ("dem", "mask", "fill", "flats", "dirs", "acc", "links", "edges")]
    out += [("curves", py + [str(HERE / "build.py"), region, "--stage", "curves"]),
            ("hgeo", py + [str(HERE / "hgeo.py"), region])]
    if dimension:
        out += [("paths-all", py + [str(HERE / "export_paths.py"), region]),
                ("dim-crests", py + [str(HERE / "run_dimension.py"), "crests", "--region", region]),
                ("dim-network", py + [str(HERE / "run_dimension.py"), "network", "--region", region])]
    return out + [("clean", rt + ["--stage", "clean"])]


class Log:
    def __init__(self, path):
        self.f = open(path, "a", buffering=1)

    def __call__(self, msg, stage=""):
        line = f"{stamp()} {'[' + stage + '] ' if stage else ''}{msg.rstrip()}"
        self.f.write(line + "\n")
        print(line, flush=True)


def run_stage(name, cmd, log):
    """Runs a child under /usr/bin/time -l, copying its output to the log; returns (seconds, peak GB)."""
    t, peak = time.time(), None
    p = subprocess.Popen(["/usr/bin/time", "-l"] + cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=HERE)
    for line in p.stdout:
        m = PEAK.search(line)
        if m:
            peak = int(m[1]) / 2**30  # bytes on macOS
        if line.strip() and not STATS.match(line):
            log(line, name)
    if p.wait() != 0:
        raise SystemExit(f"{name} failed with status {p.returncode}; see the log, then run the same command again")
    return round(time.time() - t, 1), None if peak is None else round(peak, 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("region", choices=["alps", "pyrenees"])
    ap.add_argument("--dem", type=int, choices=[25, 5], default=5)
    ap.add_argument("--tile", type=int, default=2500, help="tile side in cells (25 m: 400 is typical)")
    ap.add_argument("--window", help="trial window of dem_resolution.py (alps1, alps2, pyr1, pyr2) or a box x0,x1,y0,y1 in km")
    ap.add_argument("--no-dimension", action="store_true", help="skip the inputs of the box dimension")
    a = ap.parse_args()
    set_environment(a.dem, a.window)
    import build
    import geo

    d = build.folder(a.dem)
    root = geo.CACHE
    root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free / 1e9
    if not a.window and free < MIN_FREE_GB:
        raise SystemExit(f"only {free:.0f} GB free in {root}; a full massif needs about {MIN_FREE_GB}")
    build.write_parameters(d, a.dem)
    log = Log(d / "run.log")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=HERE).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", str(HERE)], capture_output=True, text=True, cwd=HERE).stdout.strip())
    log(f"{a.region} {a.dem} m tile {a.tile}{' window ' + a.window if a.window else ''}, folder {d}, "
        f"commit {commit}{' (uncommitted changes in analyses/ridges)' if dirty else ''}, {free:.0f} GB free")
    marks = d / f"driver_{a.region}.json"
    done = json.loads(marks.read_text()) if marks.exists() else {}
    for name, cmd in stages(a.region, a.dem, a.tile, a.window, not a.no_dimension):
        if name in done:
            log(f"skip, done in {done[name]['seconds']} s", name)
            continue
        log("start", name)
        seconds, peak = run_stage(name, cmd, log)
        done[name] = dict(seconds=seconds, peak_ram_gb=peak, finished=stamp())
        marks.write_text(json.dumps(done, indent=1))
        log(f"done in {seconds} s, peak RAM {peak} GB, free disk {shutil.disk_usage(root).free / 1e9:.0f} GB", name)
    log(f"finished: {sum(v['seconds'] for v in done.values()) / 3600:.2f} h in all, peak RAM "
        f"{max(v['peak_ram_gb'] or 0 for v in done.values()):.1f} GB")


if __name__ == "__main__":
    main()
