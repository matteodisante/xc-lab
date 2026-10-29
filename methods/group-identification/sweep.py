"""Phase 4 of the note: groups, configurations, episodes and dependency blocks of one date.

Input: the pair intervals of the date that last at least tau*, as (a, b, t_on, t_off) with
a, b flights. contacts.py restricts every interval to the time both flights are between
departure and landing, so the components of the active graph change only at the ends of
intervals.
"""

import numpy as np
import pandas as pd


def components(edges):
    """Connected components of a graph given by its edges (all of size two or more), as
    {members: edges of the component}."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        parent[find(a)] = find(b)
    members, own = {}, {}
    for x in list(parent):
        members.setdefault(find(x), set()).add(x)
    for a, b in edges:
        own.setdefault(find(a), set()).add((a, b))
    return {frozenset(m): frozenset(own[root]) for root, m in members.items()}


def groups(intervals):
    """Every group (members, start, end): a set of at least two flights that is a connected
    component of the active graph over a maximal interval. Every configuration (members,
    edges, start, end, group): a component with the same edges over a maximal interval, and
    the index of its group. The links between groups, as index pairs: g -> g' when g ends at
    the instant g' starts and they share a member."""
    events = {}
    for a, b, on, off in intervals:
        events.setdefault(on, []).append((min(a, b), max(a, b), 1))
        events.setdefault(off, []).append((min(a, b), max(a, b), -1))
    active, opened, shaped, groups, configs, links = {}, {}, {}, [], [], []
    for te in sorted(events):
        for a, b, sign in events[te]:
            active[a, b] = active.get((a, b), 0) + sign
            if not active[a, b]:
                del active[a, b]
        comps = components(active)
        ended = []
        for m in [m for m in opened if m not in comps]:
            start, index = opened.pop(m)
            groups[index] = (m, start, te)
            ended.append(index)
        for m in comps.keys() - opened.keys():
            opened[m] = (te, len(groups))
            groups.append((m, te, None))
            links += [(g, len(groups) - 1) for g in ended if groups[g][0] & m]
        for key in [k for k in shaped if comps.get(k[0]) != k[1]]:
            start, index = shaped.pop(key)
            configs[index] = (*key, start, te, configs[index][4])
        for key in comps.items() - shaped.keys():
            shaped[key] = (te, len(configs))
            configs.append((*key, te, None, opened[key[0]][1]))
    assert not opened, "an interval never ended"
    return groups, configs, links


def find_root(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def episodes(n_groups, links):
    """Episode of each group: the connected components of the links, numbered in the order
    of their first group."""
    parent = list(range(n_groups))
    for a, b in links:
        parent[find_root(parent, a)] = find_root(parent, b)
    roots = [find_root(parent, g) for g in range(n_groups)]
    number = {}
    return np.array([number.setdefault(r, len(number)) for r in roots], np.int64)


def blocks(n_flights, groups, episode):
    """Dependency block of each flight: flights tied by sharing an episode, numbered in the
    order of their first flight; a flight never in a group is a block on its own."""
    parent = list(range(n_flights))
    members = {}
    for (m, _, _), e in zip(groups, episode):
        members.setdefault(e, set()).update(m)
    for m in members.values():
        m = sorted(m)
        for f in m[1:]:
            parent[find_root(parent, f)] = find_root(parent, m[0])
    roots = [find_root(parent, f) for f in range(n_flights)]
    number = {}
    return np.array([number.setdefault(r, len(number)) for r in roots], np.int64)


def membership(table, groups):
    """Stretches of constant membership of every flight: flight, t_start, t_end, state ('L'
    from takeoff to departure, or the whole flight if local; 'S' solo; 'G' in a group) and
    group, the index of the group (-1 unless G). Times in UTC seconds."""
    of = {}
    for g, (m, start, end) in enumerate(groups):
        for f in m:
            of.setdefault(f, []).append((start, end, g))
    rows = []
    for f, flight in enumerate(table.itertuples()):
        if not np.isfinite(flight.departure):
            rows.append((f, flight.takeoff, flight.landing, "L", -1))
            continue
        rows.append((f, flight.takeoff, flight.departure, "L", -1))
        t = flight.departure
        for start, end, g in sorted(of.get(f, [])):
            if start > t:
                rows.append((f, t, start, "S", -1))
            rows.append((f, start, end, "G", g))
            t = end
        if flight.landing > t:
            rows.append((f, t, flight.landing, "S", -1))
    return pd.DataFrame(rows, columns=["flight", "t_start", "t_end", "state", "group"])
