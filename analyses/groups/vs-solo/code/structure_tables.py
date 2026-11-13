"""The tables and the numbers of the note on the structure of groups
(structure/group_structure.tex), from the JSON of the analyses:
vs-solo/generated/structure_*.tex. Every number of the note is a macro of
generated/structure_macros.tex or of generated/macros.tex.

    uv run python vs-solo/code/structure_tables.py    # from analyses/groups, after describe.py
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


def value(name, x, digits):
    """A {value, band} as two macros: name and nameBand."""
    macro(name, f"{x['value']:.{digits}f}")
    macro(f"{name}Band", f"[{x['band'][0]:.{digits}f}, {x['band'][1]:.{digits}f}]")


def age_tex():
    r = load(MAIN, "age")
    rate, day = r["rate"], "all day"
    band = lambda kind, s, i: f"[{r['band_5'][day][kind][s][i]:.2f}, {r['band_95'][day][kind][s][i]:.2f}]"
    ratio = lambda q, h, kind, s: {"value": q["ratio"][h][kind][s], "band": [q["ratio_5"][h][kind][s], q["ratio_95"][h][kind][s]]}
    for s, name in (("2", "Two"), ("8+", "Max")):
        macro(f"StAgeJoin{name}First", f"{rate[day]['join'][s][0]:.2f}")
        macro(f"StAgeJoin{name}Last", f"{rate[day]['join'][s][-1]:.2f}")
    macro("StAgeLeaveTwoFirst", f"{rate[day]['leave']['2'][2]:.2f}")
    macro("StAgeLeaveTwoFirstBand", band("leave", "2", 2))
    macro("StAgeLeaveTwoLast", f"{rate[day]['leave']['2'][-1]:.2f}")
    macro("StAgeLeaveTwoLastBand", band("leave", "2", -1))
    value("StAgeLeaveTwoRatio", ratio(r, day, "leave", "2"), 2)
    hours = [r["ratio"][h]["leave"]["2"] for h in r["hours"][1:]]
    macro("StAgeLeaveHoursLow", f"{min(hours):.2f}")
    macro("StAgeLeaveHoursHigh", f"{max(hours):.2f}")
    large = [r["ratio"][day]["leave"][s] for s in ("5", "6", "7", "8+")]
    macro("StAgeLeaveLargeLow", f"{min(large):.2f}")
    macro("StAgeLeaveLargeHigh", f"{max(large):.2f}")
    for c, name in (("repeat", "Repeat"), ("once", "Once")):
        q = r["partners"][c]
        macro(f"StAge{name}First", f"{q['rate'][day]['leave']['2'][2]:.2f}")
        value(f"StAge{name}Ratio", ratio(q, day, "leave", "2"), 2)
    macro("StAgeLandTwoFirst", f"{rate[day]['landing']['2'][2]:.3f}")
    macro("StAgeLandTwoLast", f"{rate[day]['landing']['2'][-1]:.3f}")
    value("StAgeLandLate", ratio(r, r["hours"][-1], "landing", "2"), 2)
    macro("StTauStar", f"{r['tau_star_min']:.1f}")


def breathing_tex():
    r = load(MAIN, "breathing")
    macro("StBrGroups", num(r["groups"]["all"]["groups"], 0))
    macro("StBrHours", num(r["groups"]["all"]["hours"], 0))
    f = r["by_f"]["3"]
    for key, name in (("L_par", "Par"), ("L_perp", "Perp"), ("L_z", "Z")):
        macro(f"StBr{name}Glide", f"{f[key][0]['value']:.0f}")
        macro(f"StBr{name}Half", f"{f[key][2]['value']:.0f}")
        macro(f"StBr{name}Climb", f"{f[key][3]['value']:.0f}")
    p = r["prediction"]
    value("StBrRatioGlide", p["3"]["f = 0"]["ratio"], 2)
    value("StBrRatioClimb", p["3"]["f = 1"]["ratio"], 2)
    large = [p["5+"][c]["ratio"]["value"] for c in r["classes"]]
    macro("StBrRatioLargeLow", f"{min(large):.2f}")
    macro("StBrRatioLargeHigh", f"{max(large):.2f}")
    macro("StBrPredShare", pct(r["groups"]["3"]["prediction_share"]))
    e = r["entries"]["3"]
    i = e["offsets_min"].index(1.5)
    macro("StBrEntries", num(e["entries"], 0))
    value("StBrEntryPar", e["L_par_over_before"][i], 2)
    value("StBrEntryPred", e["L_par_pred_over_before"][i], 2)
    value("StBrEntrySigma", e["sigma_l_over_before"][i], 2)


def glide_tex():
    r = load(MAIN, "glide")
    g = r["glides"]
    macro("StGlPairs", num(g["pairs"]["glides"], 0))
    macro("StGlThree", num(g["3+"]["glides"], 0))
    macro("StGlChance", num(g["chance"]["glides"], 0))
    macro("StGlMedian", f"{g['pairs']['median_min']:.1f}")
    for s, name in (("pairs", "Pairs"), ("3+", "Three")):
        a = r["along"][s]["120"]
        macro(f"StGlSigma{name}", f"{a['attr_sigma_l']['value']:.2f}")
        macro(f"StGlV{name}", f"{a['attr_v_head']['value']:.2f}")
        d = r["profile"][s]["s"]["d_rms"]
        for i, at in ((0, "Start"), (5, "Mid"), (9, "End")):
            macro(f"StGlD{name}{at}", f"{d[i]['value']:.0f}")
        z = [x["value"] for x in r["profile"][s]["s"]["L_z"]]
        macro(f"StGlZ{name}Low", f"{min(z):.0f}")
        macro(f"StGlZ{name}High", f"{max(z):.0f}")
    d = r["profile"]["chance"]["s"]["d_rms"]
    for i, at in ((0, "Start"), (5, "Mid"), (9, "End")):
        macro(f"StGlDChance{at}", f"{d[i]['value']:.0f}")
    value("StGlRatioStart", r["chance_ratio"]["s"][0], 2)
    value("StGlRatioMid", r["chance_ratio"]["s"][5], 2)
    value("StGlRatioEnd", r["chance_ratio"]["s"][-1], 2)


def rotation_tex():
    r = load(MAIN, "rotation")
    g = r["groups"]["all"]
    macro("StRotGroups", num(g["groups"], 0))
    macro("StRotHours", num(g["hours"], 0))
    macro("StRotOrdered", pct(g["ordered_share"]))
    t = r["tau"]["all"]
    for lag, name in ((1, "One"), (5, "Five"), (15, "Fifteen"), (30, "Thirty")):
        value(f"StRotTau{name}", t["tau"][t["lags_min"].index(lag)], 2)
    h = r["head"]["3"]
    macro("StRotDominant", pct(h["dominant"]["value"]))
    macro("StRotDominantBand", band_pct(h["dominant"]["band"], 0))
    macro("StRotDominantNull", pct(h["dominant_null"]["value"]))
    macro("StRotDominantNullBand", band_pct(h["dominant_null"]["band"], 0))
    macro("StRotSpell", f"{h['spell_median_min']:.1f}")
    value("StRotChanges", r["head"]["all"]["changes_per_hour"], 1)
    c = r["climbs"]["all"]
    for key, name in (("gliding", "Gliding"), ("mixed", "Mixed"), ("climbing", "Climbing")):
        value(f"StRot{name}", c["by_group_phase_per_hour"][key], 1)
    value("StRotFirstOverMid", c["first_over_mid"], 2)
    macro("StRotFirst", f"{100 * c['first_climber_new_head']['value']:.1f}")
    macro("StRotFirstBand", band_pct(c["first_climber_new_head"]["band"]))
    macro("StRotFirstExp", f"{100 * c['expected_if_any_but_old_head']['value']:.1f}")


def exits_tex():
    r = load(MAIN, "exits")
    macro("StExThermals", num(r["thermals"]["all"]["climbers_2plus"], 0))
    a = r["leaders"]["before"]["all"]
    macro("StExRows", num(a["rows"], 0))
    value("StExDz", a["dz"], 1)
    value("StExDzAfter", r["leaders"]["after"]["all"]["dz"], 1)
    value("StExDgain", a["dgain"], 0)
    macro("StExDentry", f"{a['dz_entry']['value']:.1f}")
    macro("StExDt", f"{a['dt']['value']:.0f}")
    s, n = a["shares"]["20"], r["leaders"]["random"]["all"]["shares"]["20"]
    macro("StExAbove", pct(s["above"]["value"]))
    macro("StExBelow", pct(s["below"]["value"]))
    macro("StExRandomSide", pct((n["above"]["value"] + n["below"]["value"]) / 2))
    macro("StExLater", pct(a["follower_later"]["value"]))
    k = r["link"]["all"]
    macro("StExFirstHead", pct(k["first_is_head_after"]["share"]["value"]))
    macro("StExHighHead", pct(k["highest_is_head_after"]["share"]["value"]))
    macro("StExHeadExp", pct(k["highest_is_head_after"]["expected_climbers"]))


def curvature_tex():
    r = load(MAIN, "curvature")
    p, a = r["params"], r["leaders"]["before"]["all"]

    def share(name, x, digits=1):
        macro(name, pct(x["value"], digits))
        macro(f"{name}Band", band_pct(x["band"]))

    def diff(name, x, digits):
        """Percent difference follower over leader from a median of r = log10(follower / leader)."""
        f = lambda v: f"{100 * (10 ** v - 1):.{digits}f}"
        macro(name, f(x["value"]))
        macro(f"{name}Band", f"[{f(x['band'][0])}, {f(x['band'][1])}]")

    macro("StCuRows", num(a["own"]["dk"]["rows"], 0))
    macro("StCuThermals", num(r["members_by_size"]["all"]["thermals"], 0))
    share("StCuShare", a["own"]["dk"]["leader_above"])
    macro("StCuShareRandom", pct(r["leaders"]["random"]["all"]["own"]["dk"]["leader_above"]["value"], 1))
    value("StCuR", a["own"]["dk"]["median"], 3)
    value("StCuAfterR", r["leaders"]["after"]["all"]["own"]["dk"]["median"], 3)
    diff("StCuCommon", a["common"]["dk"]["median"], 1)
    macro("StCuCommonShare", pct(a["common"]["dk"]["leader_above"]["value"], 1))
    diff("StCuClimbCommon", a["common"]["vz"]["median"], 1)
    c = r["centring"]["log10_first_over_later"]
    diff("StCuFirst", c["median"], 0)
    lo, hi = c["leader_above"]["band"]
    macro("StCuFirstShare", pct(1 - c["leader_above"]["value"], 0))
    macro("StCuFirstShareBand", band_pct([1 - hi, 1 - lo]))
    macro("StCuRadius", f"{r['plausibility']['median_radius_m']:.0f}")
    macro("StCuPeriod", f"{r['plausibility']['median_turn_period_s']:.0f}")
    macro("StCuSensMax", f"{max(abs(x['median']['value']) for x in a['sensitivity'].values()):.3f}")
    macro("StCuSG", str(p["sg_window_s"]))
    macro("StCuVmin", str(p["v_min_ms"]))
    macro("StCuMinSec", str(p["min_seconds"]))


def headings_tex():
    r = load(MAIN, "headings")
    c, p = r["classes"], r["params"]
    macro("StHdGlides", num(r["measured"]["n_real"], 0))
    macro("StHdChance", num(sum(x["n"] for x in c["chance"]), 0))
    macro("StHdTauMax", str(p["tau_max"]))
    for edge, name in zip(p["l_edges"], ("One", "Two", "Three")):
        macro(f"StHdEdge{name}", f"{edge:.0f}")
    for i, name in enumerate(("One", "Two", "Three", "Four")):
        macro(f"StHdL{name}", f"{c['pairs'][i]['l_median']['value']:.0f}")
        macro(f"StHdPeak{name}", sig(c["pairs"][i]["heading"]["tau_max"]["value"]))
        macro(f"StHdLChance{name}", f"{c['chance'][i]['l_median']['value']:.0f}")
        value(f"StHdCChance{name}", c["chance"][i]["heading"]["C_at_l"], 2)
    for i, name in ((0, "One"), (2, "Three"), (3, "Four")):
        value(f"StHdCPair{name}", c["pairs"][i]["heading"]["C_at_l"], 2)
    value("StHdCThreeFour", c["3+"][3]["heading"]["C_at_l"], 2)
    macro("StHdLThreeFour", f"{c['3+'][3]['l_median']['value']:.0f}")
    for k in macros:
        if k.startswith("StHdC"):
            macros[k] = macros[k].replace("-", "$-$")
    macro("StHdSpeedGap", str(int(np.ceil(max(abs(x["speed"]["tau_max"]["value"] - x["l_median"]["value"]) for x in c["pairs"])))))


def marking_tex():
    r = load(MAIN, "marking")
    p, m, s = r["params"], r["measured"], r["summary"]
    macro("StMkR", f"{p['R_m']:.0f}")
    macro("StMkDT", f"{p['DT_s']:.0f}")
    macro("StMkPercentile", str(p["percentile"]))
    macro("StMkPairs", num(m["pairs"], 0))
    macro("StMkNear", f"{r['S_near_min']:g}")
    macro("StMkGroupStarts", num(s["group"]["starts"], 0))
    macro("StMkSoloStarts", num(s["solo"]["starts"], 0))
    macro("StMkShareGroup", pct(s["group"]["any"]["real"]["value"], 1))
    macro("StMkShareSolo", pct(s["solo"]["any"]["real"]["value"], 1))
    value("StMkNowGroup", s["group"]["any"]["now"], 2)
    value("StMkNowGroupOther", s["group"]["group"]["now"], 2)
    value("StMkNowGroupSolo", s["group"]["solo"]["now"], 2)
    value("StMkNowSolo", s["solo"]["any"]["now"], 2)
    macro("StMkHalfGroup", f"{s['group']['any']['s_half']['value']:.0f}")
    macro("StMkHalfSolo", f"{s['solo']['any']['s_half']['value']:.0f}")
    speed = load(MAIN, "performance")["solo"]["glide speed (km/h)"]["value"]
    macro("StMkGlideSpeed", f"{speed:.0f}")
    macro("StMkReach", f"{p['DT_s'] * speed / 3600:.1f}")


def encounters_tex():
    r = load(MAIN, "encounters")
    a, p, m = r["summary"]["all"], r["params"], r["summary"]
    macro("StEnR", f"{p['R_ENC_m']:.0f}")
    macro("StEnBins", str(p["n_bins"]))
    macro("StEnBin", f"{p['R_ENC_m'] / p['n_bins']:.0f}")
    macro("StEnTheta", f"{p['theta_max']:.0f}")
    macro("StEnDelta", f"{p['delta_s']:.0f}")
    macro("StEnEpsZ", f"{p['eps_z']:.0f}")
    for kind, name in (("solo-solo", "SoloSolo"), ("solo-group", "SoloGroup"), ("group-group", "GroupGroup")):
        macro(f"StEnPasses{name}", num(a["passes"][kind], 0))
        macro(f"StEnShare{name}", pct(a["share"][kind]["value"], 1))
    macro("StEnBelowSoloSolo", pct(m["below"]["share"]["solo-solo"]["value"], 1))
    macro("StEnAboveSoloSolo", pct(m["above"]["share"]["solo-solo"]["value"], 1))
    macro("StEnNearGroupGroup", pct(a["curve"]["group-group"][0]["value"]))
    macro("StEnFarGroupGroup", pct(a["curve"]["group-group"][-1]["value"]))
    for kind, name in (("solo-group", "SoloGroup"), ("group-group", "GroupGroup")):
        value(f"StEnRatio{name}", a["ratio"][kind], 2)
        value(f"StEnObsExp{name}", a["obs_exp"][kind], 2)
        value(f"StEnAbove{name}", m["above"]["obs_exp"][kind], 2)
    value("StEnSoloPair", a["size"]["solo-pair"], 2)
    value("StEnSoloThree", a["size"]["solo-3+"], 2)


def network_tex():
    r = load(MAIN, "network")
    m, d1, d2, d3, par = r["measured"], r["d1"], r["d2"], r["d3"], r["params"]
    macro("StNwDraws", str(par["N_NULL"]))
    macro("StNwTries", str(par["TRIES"]))
    macro("StNwQLow", f"{par['Q_LOW']:g}")
    macro("StNwSMin", str(par["S_MIN"]))
    macro("StNwPilots", num(m["pilots_s_min"], 0))
    macro("StNwEdgeLow", str(m["s_edges"][0]))
    macro("StNwEdgeHigh", str(m["s_edges"][1]))
    for c, name in zip(d1, ("Few", "Mid", "Many")):
        macro(f"StNwPilots{name}", num(c["pilots"], 0))
        macro(f"StNwLow{name}", pct(c["low"]))
        macro(f"StNwRatio{name}", f"{c['median_ratio']:.2f}")
    macro("StNwRatioNullMany", f"{d1[-1]['median_ratio_null']:.2f}")
    real, null = d2["real"], d2["null_mean"]
    for i, name in enumerate(("Same", "Diff")):
        macro(f"StNwExcess{name}", num(d2["excess"][i], 0))
        macro(f"StNwExcess{name}Band", f"[{num(d2['excess_band'][0][i], 0)}, {num(d2['excess_band'][1][i], 0)}]")
        macro(f"StNwShare{name}", pct(d2["share"][i]))
    macro("StNwNullSame", pct(null[0] / sum(null)))
    macro("StNwRealSame", pct(real[0] / sum(real)))
    for key, name in (("x", "X"), ("wing", "Wing")):
        d = d3[key]
        macro(f"StNwCorr{name}", f"{d['real']['corr']:.2f}")
        value(f"StNwCorr{name}Null", {"value": d["null"]["corr"]["mean"], "band": d["null"]["corr"]["band"]}, 2)
        if key == "wing":
            macro("StNwSameWing", pct(d["real"]["same"]))
            macro("StNwSameWingNull", pct(d["null"]["same"]["mean"]))
            macro("StNwSameWingNullBand", band_pct(d["null"]["same"]["band"], 0))
    x = d3["x"]
    macro("StNwBeginner", f"{x['real']['mean'][0]:.2f}")
    value("StNwBeginnerNull", {"value": x["null"]["mean"]["mean"][0], "band": [x["null"]["mean"]["band"][0][0], x["null"]["mean"]["band"][1][0]]}, 2)
    macro("StNwVeteran", f"{x['real']['mean'][-1]:.2f}")
    value("StNwVeteranNull", {"value": x["null"]["mean"]["mean"][-1], "band": [x["null"]["mean"]["band"][0][-1], x["null"]["mean"]["band"][1][-1]]}, 2)


def main():
    GEN.mkdir(exist_ok=True)
    describe_tex()
    cohesion_tex()
    hierarchy_tex()
    chains_tex()
    partners_tex()
    fission_tex()
    age_tex()
    breathing_tex()
    glide_tex()
    rotation_tex()
    exits_tex()
    curvature_tex()
    headings_tex()
    marking_tex()
    encounters_tex()
    network_tex()
    (GEN / "structure_macros.tex").write_text(
        "\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in sorted(macros.items())) + "\n")
    print(f"{len(macros)} macros")


if __name__ == "__main__":
    main()
