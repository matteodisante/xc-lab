"""Open or closed flights, as the pilot declared them and as the track shows them, for the flights of
group time and the others.

Declared: flight_type of the FFVL catalogue. Closed: triangle, triangle FAI, quadrilateral, out and
return; open: free distance and distance with 1, 2 or 3 points; hike and fly is left apart.
On the track: the distance between the first and the last fix of the flight (the takeoff and the
landing, in the east-north frame of the fixes), alone and over the path flown (flights_meta path_km).
The ends are read once from the fixes of the paragliders (cache <run>/analyses/flight_ends.parquet).

Flights: paragliders with a logger of at most tamsd.H_MAX s that depart (those of tamsd.py); "group
flights" have group time (n_entries > 0), the others never had a group.

    uv run python closure.py [--run NAME] [--recompute]    # writes out/<run>/closure.json
"""

import json
import sys

import numpy as np
import pandas as pd

import tamsd
from common.legacy import stream_flights
from openflights import OPEN_KM
from tamsd import FIXES, H_MAX, LABELS, OUT

ENDS = tamsd.SHARED / "flight_ends.parquet"
CLOSED = {"triangle", "triangle FAI", "Quadrilatère", "Aller-Retour", "Marche et Vol bouclé", "Marche et Vol FAI"}
OPEN = {"Dist libre", "Dist 1 pt", "Dist 2 pts", "Dist 3 pts"}


def read_ends():
    """First and last fix, by time, of every paraglider flight: flight_id, e0, n0, e1, n1, t0, t1."""
    rows = []
    for fixes in stream_flights(FIXES["para"], ["t", "E", "N"]):
        t, e, n = (fixes[c].to_numpy(float) for c in ("t", "E", "N"))
        first, last = int(np.argmin(t)), int(np.argmax(t))
        rows.append((str(fixes["flight_id"].iat[0]), e[first], n[first], e[last], n[last], t[first], t[last]))
    return pd.DataFrame(rows, columns=["flight_id", "e0", "n0", "e1", "n1", "t0", "t1"])


def flight_table():
    """The eligible paraglider flights with their declared kind, the takeoff-landing distance and the
    path flown."""
    flights = pd.read_parquet(LABELS / "flights.parquet").astype({"flight_id": str})
    summary = pd.read_parquet(LABELS / "flight_summary.parquet")[["flight", "n_entries"]]
    catalog = pd.read_csv(FIXES["para"].parent.parent / "catalog/catalog.csv", usecols=["flight_id", "flight_type"],
                          dtype={"flight_id": str})
    meta = pd.read_parquet(FIXES["para"].parent / "flights_meta.parquet", columns=["flight_id", "path_km"]).astype({"flight_id": str})
    ends = pd.read_parquet(ENDS)
    f = flights.merge(summary, on="flight").merge(catalog, on="flight_id", how="left").merge(meta, on="flight_id", how="left")
    f = f.merge(ends, on="flight_id", how="left")
    f = f[(f["discipline"] == "para") & (f["dt_native"] <= H_MAX + 1e-6) & (f["n_grid"] > 0)].copy()
    f["declared"] = np.where(f["flight_type"].isin(CLOSED), "closed", np.where(f["flight_type"].isin(OPEN), "open", "other"))
    f["ends_km"] = np.hypot(f["e1"] - f["e0"], f["n1"] - f["n0"]) / 1000
    f["ratio"] = f["ends_km"] / f["path_km"]
    f["group"] = f["n_entries"] > 0
    return f


RULES = {f"ends <= {k} km": ("ends_km", k) for k in (3, 4, 5, 6, 8, 10)} | {f"ends <= {k:.0%} of the path": ("ratio", k) for k in (0.05, 0.1, 0.15, 0.2, 0.3)}


def shares(kind, mask):
    """Percent of the flights of `mask` that are closed, open and other, by the labels `kind`."""
    return {k: round(100 * float((kind[mask] == k).mean()), 1) for k in ("closed", "open", "other")}


if __name__ == "__main__":
    if "--recompute" in sys.argv or not ENDS.exists():
        read_ends().to_parquet(ENDS, index=False)
    f = flight_table()
    known = f[f["declared"] != "other"]
    result = {"flights": len(f), "declared": {"group": shares(f["declared"], f["group"]), "not group": shares(f["declared"], ~f["group"])},
              "n": {"group": int(f["group"].sum()), "not group": int((~f["group"]).sum())}, "rules": {}}
    for name, (col, k) in RULES.items():
        closed = f[col] <= k
        label = pd.Series(np.where(closed, "closed", "open"), index=f.index)
        agree = (label[known.index] == known["declared"])
        result["rules"][name] = {
            "agreement of the declared closed (%)": round(100 * float(agree[known["declared"] == "closed"].mean()), 1),
            "agreement of the declared open (%)": round(100 * float(agree[known["declared"] == "open"].mean()), 1),
            "agreement (%)": round(100 * float(agree.mean()), 1),
            "group": shares(label, f["group"]), "not group": shares(label, ~f["group"])}
    rule = result["rules"][f"ends <= {OPEN_KM} km"]
    result["open"] = {"km": OPEN_KM, "flights": int((f["ends_km"] > OPEN_KM).sum()), "of": len(f),
                      "agreement (%)": rule["agreement (%)"], "group": 100 - rule["group"]["closed"],
                      "declared open and open on the track": int(((f["ends_km"] > OPEN_KM) & (f["declared"] == "open")).sum())}
    (OUT / "closure.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result["declared"]), result["n"])
    for name, r in result["rules"].items():
        print(f"{name:28s} agree {r['agreement (%)']:5.1f} (closed {r['agreement of the declared closed (%)']:5.1f}, open "
              f"{r['agreement of the declared open (%)']:5.1f})  group closed {r['group']['closed']:5.1f}  not group {r['not group']['closed']:5.1f}")
