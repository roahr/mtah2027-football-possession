"""Tracking-derived control features and home/away sequence decoding.

The release uses local geometric control proxies as model features and viterbi_team to decode
externally trained ownership probabilities. In the release configuration, direct control,
shape and event contributions to the decoder have zero weight; supervised ownership evidence
and the transition penalty determine the path. Other event-based combinations are retained
for experimental comparisons.
"""
import numpy as np

from features import FPS, HOME, AWAY

REACH = 1.6          # m: ball within a stride of the player
MARGIN = 0.6         # m: and the nearest opponent at least this much further away
MAX_CARRY = 9.0      # m/s: faster than a sprinting dribble is a ball in flight
MIN_CONTROL = 6      # frames (0.24 s) of control for a lone event to count as possession
REPEAT_WINDOW = 3 * FPS  # a second shorter event by the same team within 3 s confirms the first
BRIDGE = 3           # frames of tracking flicker tolerated inside one event


def control_frames(f):
    """Per frame: HOME, AWAY or -1 for 'nobody controls the ball here'."""
    dH, dA, bs = f["dH"], f["dA"], f["bspeed"]
    out = np.full(f["n"], -1, np.int8)
    ok = f["have"] & ~np.isnan(dH) & ~np.isnan(dA)
    bs = np.where(np.isnan(bs), 0.0, bs)
    home = ok & (dH < REACH) & (dA - dH > MARGIN) & (bs < MAX_CARRY)
    away = ok & (dA < REACH) & (dH - dA > MARGIN) & (bs < MAX_CARRY)
    out[home] = HOME
    out[away] = AWAY
    return out


def control_events(ctrl):
    """List of (start, end_exclusive, team) runs, bridging flicker of up to BRIDGE frames."""
    events = []
    i, n = 0, len(ctrl)
    while i < n:
        if ctrl[i] < 0:
            i += 1
            continue
        team, start, last = ctrl[i], i, i
        j = i + 1
        while j < n and j - last <= BRIDGE + 1:
            if ctrl[j] == team:
                last = j
            elif ctrl[j] >= 0:
                break
            j += 1
        events.append((start, last + 1, int(team)))
        i = last + 1
    return events


def confirmed_events(events):
    """Keep the events that are control rather than a touch."""
    keep = []
    for k, (a, b, t) in enumerate(events):
        if b - a >= MIN_CONTROL:
            keep.append((a, b, t))
            continue
        # short: confirmed if the same team has the next event soon after
        if k + 1 < len(events):
            a2, b2, t2 = events[k + 1]
            if t2 == t and a2 - b <= REPEAT_WINDOW:
                keep.append((a, b, t))
    return keep


def assign_alive(f, alive_segments, events, prior=None):
    """Label each alive segment frame-by-frame with HOME/AWAY.

    `prior` is an optional per-frame P(home) used only where the tracking gives no control event
    to decide by (a whole segment without a visible touch).
    """
    labels = np.full(f["n"], -1, np.int8)
    starts = np.array([e[0] for e in events]) if events else np.zeros(0, int)
    for a, b in alive_segments:
        lo, hi = np.searchsorted(starts, a), np.searchsorted(starts, b)
        seg_events = events[lo:hi]
        if not seg_events:
            if prior is not None:
                p = np.nanmean(prior[a:b])
                labels[a:b] = HOME if (np.isnan(p) or p >= 0.5) else AWAY
            continue
        # from the restart up to the first control, the restarting team has the ball
        current = seg_events[0][2]
        cursor = a
        for (ea, eb, t) in seg_events:
            if t != current:
                labels[cursor:ea] = current
                cursor = ea
                current = t
        labels[cursor:b] = current
    return labels


def viterbi_team(f, alive_segments, ctrl, events, p_home, w_ctrl=1.0, w_shape=0.08,
                 switch=12.0, w_event=0.0, p_sup=None, w_sup=0.0):
    """Two-state (HOME/AWAY) Viterbi inside each alive segment.

    Evidence per frame, as a log-likelihood ratio home-vs-away:
      control   +/- w_ctrl on every frame a team controls the ball (control_frames)
      events    +/- w_event extra on frames inside a confirmed control event
      shape     w_shape * logit(P_home from the shape classifier)
    and a fixed cost `switch` for every change of possession. Viterbi then finds the labelling
    with the best evidence-minus-switches, which is exactly the trade-off spellF1 rewards: a
    change of possession has to be paid for by sustained evidence, not one noisy frame.
    """
    llr = np.zeros(f["n"])
    llr += np.where(ctrl == HOME, w_ctrl, 0.0) - np.where(ctrl == AWAY, w_ctrl, 0.0)
    if w_event:
        for a, b, t in events:
            llr[a:b] += w_event if t == HOME else -w_event
    if p_home is not None and w_shape:
        p = np.clip(np.nan_to_num(p_home, nan=0.5), 0.05, 0.95)
        llr += w_shape * np.log(p / (1 - p))
    if p_sup is not None and w_sup:
        q = np.clip(np.nan_to_num(p_sup, nan=0.5), 0.02, 0.98)
        llr += w_sup * np.log(q / (1 - q))

    labels = np.full(f["n"], -1, np.int8)
    for a, b in alive_segments:
        e = llr[a:b]
        T = len(e)
        # scores for states 0 (home) and 1 (away)
        s0, s1 = e[0] / 2, -e[0] / 2
        back = np.zeros((T, 2), np.bool_)   # True: came from the other state
        for t in range(1, T):
            h_stay, h_sw = s0, s1 - switch
            a_stay, a_sw = s1, s0 - switch
            back[t, 0] = h_sw > h_stay
            back[t, 1] = a_sw > a_stay
            s0 = max(h_stay, h_sw) + e[t] / 2
            s1 = max(a_stay, a_sw) - e[t] / 2
        state = 0 if s0 >= s1 else 1
        out = np.empty(T, np.int8)
        for t in range(T - 1, -1, -1):
            out[t] = state
            if back[t, state]:
                state = 1 - state
        labels[a:b] = out
    return labels


def place_switches_at_touch(f, team, ctrl, touch=2.0, min_visible=0.9):
    """Put each change of possession at the new team's first touch.

    A change of possession happens when the other team wins the ball: in a duel, the moment the
    tackler gets to it; for a pass that is cut out, the interception. Between the old team's last
    controlled frame p and the new team's first controlled frame c, the change is placed at the
    first frame a player of the new team is within `touch` metres of the ball (c if none is). The
    decoder's own placement inside that gap rests only on the slow team-shape evidence. Applied
    only when the ball is in view for the whole gap.
    """
    team = team.copy()
    n = len(team)
    near = {HOME: f["dH"], AWAY: f["dA"]}
    sw = np.where((team[1:] >= 0) & (team[:-1] >= 0) & (team[1:] != team[:-1]))[0] + 1
    for s in sw:
        new, old = team[s], team[s - 1]
        o0 = s - 1                       # the old team's run is [o0, s)
        while o0 > 0 and team[o0 - 1] == old:
            o0 -= 1
        e = s                            # the new team's run is [s, e)
        while e < n and team[e] == new:
            e += 1
        po = np.where(ctrl[o0:s] == old)[0]
        if len(po) == 0:
            continue
        p = o0 + po[-1]
        cn = np.where(ctrl[p + 1:e] == new)[0]
        if len(cn) == 0:
            continue
        c = p + 1 + cn[0]
        if f["have"][p + 1:c + 1].mean() < min_visible:
            continue
        hit = np.where(near[new][p + 1:c + 1] < touch)[0]
        s2 = p + 1 + hit[0] if len(hit) else c
        if s2 < s:
            team[s2:s] = new
        elif s2 > s:
            team[s:s2] = old
    return team


def control_from_broadcast(f):
    """Control frames from the broadcast camera's ball, where the tactical ball is missing.

    The broadcast ball is used only on frames where the broadcast shows live play (see
    broadcast.py); positions are compared with the tactical players at the same frame.
    """
    import broadcast
    from parse_atd import parse
    from features import L, W
    out = np.full(f["n"], -1, np.int8)
    bb = broadcast.build(verbose=False)["ball"]
    idx = np.where(~f["have"] & ~np.isnan(bb[:, 0]) & f["graded"])[0]
    if len(idx) == 0:
        return out
    d = parse("tactical")
    P, teams = d["players"][idx, :, :2].astype(float), d["teams"]
    dist = np.hypot((P[:, :, 0] - bb[idx, None, 0]) * L, (P[:, :, 1] - bb[idx, None, 1]) * W)
    dist = np.where(np.isnan(dist), np.inf, dist)
    dH = np.min(np.where(teams[None, :] == HOME, dist, np.inf), 1)
    dA = np.min(np.where(teams[None, :] == AWAY, dist, np.inf), 1)
    # ball speed from the broadcast track itself, where consecutive frames exist
    bx, by = bb[:, 0] * L, bb[:, 1] * W
    bs = np.hypot(np.gradient(bx), np.gradient(by)) * FPS
    bs = np.where(np.isnan(bs[idx]), 0.0, bs[idx])
    home = (dH < REACH) & (dA - dH > MARGIN) & (bs < MAX_CARRY)
    away = (dA < REACH) & (dH - dA > MARGIN) & (bs < MAX_CARRY)
    out[idx[home]] = HOME
    out[idx[away]] = AWAY
    return out
