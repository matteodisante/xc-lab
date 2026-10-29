"""The tables and the numbers of section 7 of the group identification note
(methods/group-identification/group_identification.tex), from the JSON of the analyses:
report/generated/structure_*.tex. Every number in those sections is a macro of
generated/structure_macros.tex or of generated/macros.tex.

    uv run python report/structure_tables.py    # from analyses/groups/group-identification, after describe.py
"""

import numpy as np

from tables import GEN, MAIN, load, num, write

OUTCOMES = ("join", "leave", "landing", "split", "merge")
macros = {}


def macro(name, value):
    macros[name] = value


def sig(x):
    """Two significant digits, trailing zeros kept."""
    if not x:
        return "0"
    x = round(x, 1 - int(np.floor(np.log10(abs(x)))))
    return f"{x:.{max(0, 1 - int(np.floor(np.log10(abs(x)))))}f}"


def describe_tex():
    r = load(MAIN, "describe")
    count, hours, form, end = r["size_count"], r["size_hours"], r["formations"], r["dissolutions"]
    rt = r["rates"]
    macro("StGroups", num(r["groups"], 0))
    macro("StEpisodes", num(r["episodes"], 0))
    macro("StWithHang", f"{100 * r['with_hang']:.1f}")
    macro("StPairsCount", f"{100 * count['2'] / sum(count.values()):.0f}")
    macro("StPairsTime", f"{100 * hours['2'] / sum(hours.values()):.0f}")
    macro("StLargest", str(max(map(int, count))))
    macro("StEpisodeSingle", f"{100 * r['episode_single']:.0f}")
    macro("StEpisodeSplit", f"{100 * r['episode_with_split']:.1f}")
    macro("StEpisodeMerge", f"{100 * r['episode_with_merge']:.1f}")
    macro("StMedianGroup", f"{r['median_group_min']:.0f}")
    macro("StMedianEpisode", f"{r['median_episode_min']:.0f}")
    macro("StEpisodeTwoHours", f"{100 * r['episode_2h']:.0f}")
    macro("StFormations", num(sum(form.values()), 0))
    macro("StFormDeparture", f"{100 * form['departure'] / sum(form.values()):.0f}")
    macro("StEnds", num(sum(end.values()), 0))
    macro("StEndFlight", f"{100 * end['leave'] / sum(end.values()):.0f}")
    macro("StEndLanding", f"{100 * end['landing'] / sum(end.values()):.0f}")
    for x in ("leave", "join", "landing", "merge"):
        macro(f"StRate{x.capitalize()}Two", sig(rt["rate"][x][0]))
    macro("StRateSplitFour", sig(rt["rate"]["split"][2]))
    macro("StRateSplitMax", sig(rt["rate"]["split"][-1]))
    macro("StSizeMax", str(rt["sizes"][-1]))
    by_size = r["median_group_min_by_size"]
    macro("StMedianGroupTwo", f"{by_size['2']:.0f}")
    macro("StMedianGroupMax", f"{by_size[str(rt['sizes'][-1])]:.0f}")
    rows = []
    for k, n in enumerate(rt["sizes"]):
        cells = ["--" if x == "split" and n < 4 else
                 sig(rt["rate"][x][k]) + r"\,{\tiny[" + sig(rt["band"][x][0][k]) + ", " + sig(rt["band"][x][1][k]) + "]}"
                 for x in OUTCOMES]
        label = f"{n}+" if k == len(rt["sizes"]) - 1 else str(n)
        rows.append(" & ".join([label, num(rt["hours"][k], 0), f"{by_size[str(n)]:.0f}", *cells]) + r" \\")
    write("structure_rates_table", "\n".join([
        r"\begin{tabular}{@{}r r r l l l l l@{}}", r"\toprule",
        r"pilots & hours & median (min) & join & leave & landing & split & merge \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}"]))


def pct(x, digits=0):
    return f"{100 * x:.{digits}f}"


def band_pct(b, digits=1):
    return f"[{pct(b[0], digits)}, {pct(b[1], digits)}]"


def chains_tex():
    r = load(MAIN, "chains")
    s, e = r["shares"], r["effect"]
    macro("StTriangleThree", pct(s["shape"]["complete"][0]))
    macro("StChainThree", pct(s["shape"]["tree"][0]))
    macro("StCompleteFour", pct(s["shape"]["complete"][1]))
    macro("StCompleteMax", pct(s["shape"]["complete"][-1], 1))
    macro("StDensityThree", f"{s['density'][0]:.2f}")
    macro("StDensityMax", f"{s['density'][-1]:.2f}")
    macro("StDiamFourMax", pct(s["diameter"]["4"][-1]))
    for key, name in (("n3 triangle", "Triangle"), ("n3 chain", "Chain"), ("n4+ complete", "Complete"), ("n4+ tree", "Tree")):
        v = e[key]
        macro(f"StHloc{name}", f"{v['H_loc_1000']:.3f} [{v['H_loc_1000_band'][0]:.3f}, {v['H_loc_1000_band'][1]:.3f}]")
        macro(f"StFlights{name}", num(v["flights"], 0))


def partners_tex():
    r = load(MAIN, "partners")
    o, n = r["observed"], r["null"]
    macro("StPairDays", num(r["pair_days"], 0))
    macro("StPilots", num(r["pilots"], 0))
    macro("StPairs", num(r["pairs"], 0))
    macro("StMaxDays", str(r["max_days"]))
    macro("StRepeatObs", pct(o["repeat_share"], 1))
    macro("StRepeatNull", pct(n["repeat_share"]["mean"], 1))
    macro("StRepeatNullBand", band_pct(n["repeat_share"]["band"]))
    macro("StRepeatPairsObs", num(o["repeat_pairs"], 0))
    macro("StRepeatPairsNull", num(n["repeat_pairs"]["mean"], 0))
    macro("StSameClubObs", pct(o["same_club_main"], 1))
    macro("StSameClubNull", pct(n["same_club"]["mean"], 1))
    macro("StSameClubNullBand", band_pct(n["same_club"]["band"]))


def fission_tex():
    r = load(MAIN, "fission")
    c = r["capita"]
    loss = np.array(r["rates"]["loss"]) / np.array(r["rates"]["n"])
    macro("StMultiTwo", pct(r["multi_share"][0]))
    macro("StMultiMax", pct(r["multi_share"][-1]))
    macro("StCapitaA", f"{c['a']:.3f}")
    macro("StCapitaABand", f"[{c['a_band'][0]:.3f}, {c['a_band'][1]:.3f}]")
    macro("StGamma", f"{c['gamma']:.2f}")
    macro("StGammaBand", f"[{c['gamma_band'][0]:.2f}, {c['gamma_band'][1]:.2f}]")
    macro("StLossTwo", f"{loss[0]:.2f}")
    macro("StLossMax", f"{loss[-1]:.2f}")
    macro("StX", f"{r['x']:.2f}")
    macro("StNTop", str(r["n"][-1]))
    macro("StLogSeriesFactor", sig(r["T"][-1] / r["log_series"][-1]))
    macro("StCapitaFactor", f"{r['T'][-1] / r['per_capita'][-1]:.0f}")
    macro("StAlpha", f"{r['bonabeau']['alpha']:.1f}")
    macro("StNc", f"{r['bonabeau']['n_c']:.0f}")
    macro("StChainThreePred", f"{r['measured'][1]:.3f}")
    macro("StChainThreeObs", f"{r['T'][1]:.3f}")


def cohesion_tex():
    r = load(MAIN, "cohesion")
    f, d, s = r["fit"], r["drift"], r["sep"]
    for key, name in (("real", "Real"), ("chance", "Chance"), ("chance_long", "ChanceLong")):
        macro(f"StCoh{name}", num(r["intervals"][key], 0))

    def minutes(k, band):
        return f"{1 / k / 60:.1f}", f"[{1 / band[1] / 60:.1f}, {1 / band[0] / 60:.1f}]"

    for key, name in (("real/all", "Real"), ("chance_long/all", "Chance")):
        macro(f"StOu{name}", minutes(f[key]["k"], f[key]["k_band"])[0])
        macro(f"StOu{name}Band", minutes(f[key]["k"], f[key]["k_band"])[1])
        macro(f"StSigma{name}", f"{f[key]['sigma']:.0f}")
        macro(f"StSigma{name}Band", "[{:.0f}, {:.0f}]".format(*f[key]["sigma_band"]))
        macro(f"StDrift{name}", minutes(d[key]["k"], d[key]["k_band"])[0])
        macro(f"StDrift{name}Band", minutes(d[key]["k"], d[key]["k_band"])[1])
    macro("StQReal", f"{f['real/all']['q']:.1f}")
    macro("StQRealBand", "[{:.1f}, {:.1f}]".format(*f["real/all"]["q_band"]))
    for key, name in (("chance_long/all", "ChanceLong"), ("chance/all", "Chance")):
        macro(f"StQ{name}", f"{f[key]['q']:.0f}")
        macro(f"StQ{name}Band", "[{:.0f}, {:.0f}]".format(*f[key]["q_band"]))
    for n, name in (("2", "Two"), ("3", "Three"), ("4+", "Four")):
        macro(f"StQ{name}", f"{f[f'real/{n}']['q']:.1f}")
    for kind, name in (("real_long", "RealLong"), ("chance_long", "ChanceLong"), ("real", "Real"), ("chance", "Chance")):
        for m, mn in ((30, "Thirty"), (60, "Sixty"), (120, "TwoHours")):
            macro(f"StSep{name}{mn}", f"{s[kind]['median'][m // 5] / 1000:.1f}")


def hierarchy_tex():
    r = load(MAIN, "hierarchy")
    o, s, w = r["order"], r["switches"], r["who_leads"]
    macro("StCyclic", pct(o["cyclic"]["value"], 1))
    macro("StCyclicBand", band_pct(o["cyclic"]["band"]))
    macro("StCyclicNull", pct(o["cyclic_null"]["value"]))
    macro("StFullShare", pct(o["full_share"]))
    macro("StAcyclic", pct(o["acyclic"]["value"], 1))
    macro("StAcyclicNull", pct(o["acyclic_null"]["value"]))
    macro("StOneLeader", pct(o["one_leader"]["value"], 1))
    macro("StOneLeaderBand", band_pct(o["one_leader"]["band"]))
    macro("StOneLeaderNull", pct(o["one_leader_null"]["value"], 1))
    macro("StOneLeaderNullBand", band_pct(o["one_leader_null"]["band"]))
    for k, name in (("3", "Three"), ("4", "Four"), ("5", "Five")):
        macro(f"StOneLeader{name}", pct(o["by_size"][k]["one_leader"]))
        macro(f"StLine{name}", pct(o["by_size"][k]["line"]))
    for kind, name in (("real", ""), ("chance", "Chance")):
        macro(f"StSwitches{name}", f"{s[kind]['per_hour']:.1f}")
        macro(f"StSwitches{name}Band", "[{:.1f}, {:.1f}]".format(*s[kind]["per_hour_band"]))
        macro(f"StDominant{name}", pct(s[kind]["dominant_median"]))
        macro(f"StNeverSwitch{name}", pct(s[kind]["never_switch"]))
    for key, name in (("experience", "Exp"), ("wing", "Wing")):
        macro(f"St{name}Pairs", num(w[key]["pairs"], 0))
        macro(f"St{name}", pct(w[key]["value"], 1))
        macro(f"St{name}Band", band_pct(w[key]["band"]))
    macro("StExpRatio", f"{w['experience']['median_ratio']:.1f}")


def main():
    GEN.mkdir(exist_ok=True)
    describe_tex()
    cohesion_tex()
    hierarchy_tex()
    chains_tex()
    partners_tex()
    fission_tex()
    (GEN / "structure_macros.tex").write_text(
        "\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in sorted(macros.items())) + "\n")
    print(f"{len(macros)} macros")


if __name__ == "__main__":
    main()
