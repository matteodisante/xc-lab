"""Phase 4 of the note: configurations, episodes and dependency blocks of one date.

Input: the pair intervals of the date that last at least tau*, as (a, b, t_on, t_off) with
a, b flights. contacts.py restricts every interval to the time both flights are between
departure and landing, so the components of the active graph change only at the ends of
intervals.
"""

import numpy as np
import pandas as pd


def components(edges):
    """Connected components of a graph given by its edges (all of size two or more)."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        parent[find(a)] = find(b)
    groups = {}
    for x in list(parent):
        groups.setdefault(find(x), set()).add(x)
    return {frozenset(g) for g in groups.values()}


def configurations(intervals):
    """Every configuration (members, start, end): a set of at least two flights that is a
    connected component of the active graph over a maximal interval. Also the links between
    configurations, as index pairs: c -> c' when c ends at the instant c' starts and they
    share a member."""
    events = {}
    for a, b, on, off in intervals:
        events.setdefault(on, []).append((min(a, b), max(a, b), 1))
        events.setdefault(off, []).append((min(a, b), max(a, b), -1))
    active, opened, configs, links = {}, {}, [], []
    for te in sorted(events):
        for a, b, sign in events[te]:
            active[a, b] = active.get((a, b), 0) + sign
            if not active[a, b]:
                del active[a, b]
        comps = components(active)
        closed = [m for m in opened if m not in comps]
        ended = []
        for m in closed:
            start, index = opened.pop(m)
            configs[index] = (m, start, te)
            ended.append(index)
        for m in comps - set(opened):
            opened[m] = (te, len(configs))
            configs.append((m, te, None))
            links += [(c, len(configs) - 1) for c in ended if configs[c][0] & m]
    assert not opened, "an interval never ended"
    return configs, links


def find_root(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def episodes(n_configs, links):
    """Episode of each configuration: the connected components of the links, numbered in
    the order of their first configuration."""
    parent = list(range(n_configs))
    for a, b in links:
        parent[find_root(parent, a)] = find_root(parent, b)
    roots = [find_root(parent, c) for c in range(n_configs)]
    number = {}
    return np.array([number.setdefault(r, len(number)) for r in roots], np.int64)


def blocks(n_flights, configs, episode):
    """Dependency block of each flight: flights tied by sharing an episode, numbered in the
    order of their first flight; a flight never in a group is a block on its own."""
    parent = list(range(n_flights))
    members = {}
    for (m, _, _), e in zip(configs, episode):
        members.setdefault(e, set()).update(m)
    for m in members.values():
        m = sorted(m)
        for f in m[1:]:
            parent[find_root(parent, f)] = find_root(parent, m[0])
    roots = [find_root(parent, f) for f in range(n_flights)]
    number = {}
    return np.array([number.setdefault(r, len(number)) for r in roots], np.int64)


def membership(table, configs):
    """Stretches of constant membership of every flight: flight, t_start, t_end, state ('L'
    from takeoff to departure, or the whole flight if local; 'S' solo; 'G' in a group) and
    conf, the index of the configuration (-1 unless G). Times in UTC seconds."""
    of = {}
    for c, (m, start, end) in enumerate(configs):
        for f in m:
            of.setdefault(f, []).append((start, end, c))
    rows = []
    for f, flight in enumerate(table.itertuples()):
        if not np.isfinite(flight.departure):
            rows.append((f, flight.takeoff, flight.landing, "L", -1))
            continue
        rows.append((f, flight.takeoff, flight.departure, "L", -1))
        t = flight.departure
        for start, end, c in sorted(of.get(f, [])):
            if start > t:
                rows.append((f, t, start, "S", -1))
            rows.append((f, start, end, "G", c))
            t = end
        if flight.landing > t:
            rows.append((f, t, flight.landing, "S", -1))
    return pd.DataFrame(rows, columns=["flight", "t_start", "t_end", "state", "conf"])
