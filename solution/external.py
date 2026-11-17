"""Public tracking data from OTHER matches, with frame labels, for training and validation.

The sources below provide labelled data from matches other than the competition match:

  IDSSE   DFL / Sportec Solutions open data: 7 German league matches, 25 Hz optical tracking of all
          players and the ball, CC BY 4.0. Bassek, Rein, Weber & Memmert, "An integrated dataset of
          spatiotemporal and event data in elite soccer", Scientific Data 12, 195 (2025),
          https://doi.org/10.1038/s41597-025-04505-y. The ball track carries BallStatus (alive /
          dead) and BallPossession (home / away) on every frame -- the same kind of label the
          competition derives from corrected tracking.
  Metrica Metrica Sports sample data: 3 anonymised matches, 25 Hz tracking of all players and the
          ball, plus an on-ball event log (github.com/metrica-sports/sample-data). Frame labels are
          derived from the events, see metrica_labels().
  SkillCorner  SkillCorner open data: 20 A-League 2024/25 matches of tracking computed from the
          broadcast video (10 Hz, resampled to 25 Hz), MIT licence (github.com/SkillCorner/opendata).
          Automatic tracking with its real noise; labels from SkillCorner's phases of play, see
          _skillcorner(). Listed separately by available_skillcorner().

Download with solution/download_external.sh. Every loader returns the same dict:

  players (N, K, 2)  normalised pitch coordinates (0..1, origin top-left), NaN when off the pitch
  ball    (N, 3)     normalised x, y and height in metres (NaN height when unknown)
  teams   (K,)       0 home, 1 away
  labels  (N,)       'H' / 'A' / 'D', 'X' outside play
  pitch   (L, W)     metres
  tracked            (SkillCorner only) True: automatic tracking already, used as is by augment.py
"""
import json
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXT = os.path.join(ROOT, "data", "external")
CACHE = os.path.join(HERE, "cache")

IDSSE_MATCHES = {
    "J03WMX": "DFL-COM-000001", "J03WN1": "DFL-COM-000001", "J03WOH": "DFL-COM-000002",
    "J03WOY": "DFL-COM-000002", "J03WPY": "DFL-COM-000002", "J03WQQ": "DFL-COM-000002",
    "J03WR9": "DFL-COM-000002",
}
METRICA_GAMES = (1, 2, 3)
HALFTIME_GAP = 500  # ungraded frames inserted between halves


def _cached(name, fn):
    path = os.path.join(CACHE, "ext_{}.npz".format(name))
    if os.path.exists(path):
        d = dict(np.load(path, allow_pickle=False))
        d["pitch"] = tuple(d["pitch"])
        return d
    d = fn()
    os.makedirs(CACHE, exist_ok=True)
    from cacheio import save_npz
    save_npz(path, **d)
    return d


# -- IDSSE ------------------------------------------------------------------------------------

_ATTR = re.compile(r'(\w+)="([^"]*)"')


def _idsse(match_id):
    comp = IDSSE_MATCHES[match_id]
    info = ET.parse(os.path.join(EXT, "idsse", "DFL_02_01_matchinformation_{}_DFL-MAT-{}.xml".format(
        comp, match_id))).getroot()
    roles = {t.get("TeamId"): t.get("Role") for t in info.iter("Team")}
    env = next(info.iter("Environment"), None)
    L = float(env.get("PitchX")) if env is not None and env.get("PitchX") else 105.0
    W = float(env.get("PitchY")) if env is not None and env.get("PitchY") else 68.0

    # stream the positions file: one FrameSet per person (or the ball) per half
    tracks = {}           # (section, person) -> (team, list of (N, X, Y))
    ball = {}             # section -> list of (N, X, Y, Z, possession, status)
    cur = None
    path = os.path.join(EXT, "idsse", "DFL_04_03_positions_raw_observed_{}_DFL-MAT-{}.xml".format(
        comp, match_id))
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            s = line.lstrip()
            if s.startswith("<Frame "):
                a = dict(_ATTR.findall(s))
                if cur[1] == "BALL":
                    cur[2].append((int(a["N"]), float(a["X"]), float(a["Y"]), float(a.get("Z", "nan")),
                                   int(a.get("BallPossession", "0")), int(a.get("BallStatus", "0"))))
                else:
                    cur[2].append((int(a["N"]), float(a["X"]), float(a["Y"])))
            elif s.startswith("<FrameSet "):
                a = dict(_ATTR.findall(s))
                sec, team = a["GameSection"], a["TeamId"]
                if team.upper() == "BALL" or a.get("PersonId", "").endswith("0000XT"):
                    cur = (sec, "BALL", ball.setdefault(sec, []))
                else:
                    key = (sec, a["PersonId"])
                    if key not in tracks:
                        tracks[key] = (team, [])
                    cur = (sec, team, tracks[key][1])
    sections = [s for s in ("firstHalf", "secondHalf") if s in ball]
    team_of = {p: t for (_, p), (t, _) in tracks.items()}
    # players of the two teams only (referees are tracked too)
    persons = sorted({p for (_, p) in tracks if roles.get(team_of[p]) in ("home", "guest")})
    teams = np.array([0 if roles.get(team_of[p]) == "home" else 1 for p in persons], np.int8)
    col = {p: k for k, p in enumerate(persons)}

    blocks = []
    for sec in sections:
        b = np.array(ball[sec], dtype=np.float64)
        b = b[np.argsort(b[:, 0], kind="stable")]
        n0, n1 = int(b[0, 0]), int(b[-1, 0])
        n = n1 - n0 + 1
        P = np.full((n, len(persons), 2), np.nan)
        B = np.full((n, 3), np.nan)
        lab = np.full(n, "X", dtype="<U1")
        i = (b[:, 0] - n0).astype(int)
        B[i, 0] = (b[:, 1] + L / 2) / L
        B[i, 1] = (W / 2 - b[:, 2]) / W
        B[i, 2] = b[:, 3]
        alive = b[:, 5] == 1
        lab[i] = np.where(~alive, "D", np.where(b[:, 4] == 1, "H", np.where(b[:, 4] == 2, "A", "?")))
        for (s2, p), (_, fr) in tracks.items():
            if s2 != sec or not fr or p not in col:
                continue
            f = np.array(fr, dtype=np.float64)
            j = (f[:, 0] - n0).astype(int)
            ok = (j >= 0) & (j < n)
            P[j[ok], col[p], 0] = (f[ok, 1] + L / 2) / L
            P[j[ok], col[p], 1] = (W / 2 - f[ok, 2]) / W
        # alive frames with no possession flag keep the previous team
        unk = np.where(lab == "?")[0]
        for k in unk:
            lab[k] = lab[k - 1] if k > 0 and lab[k - 1] in "HA" else "X"
        blocks.append((P, B, lab))

    gapP = np.full((HALFTIME_GAP, len(persons), 2), np.nan)
    gapB = np.full((HALFTIME_GAP, 3), np.nan)
    gapL = np.full(HALFTIME_GAP, "X", dtype="<U1")
    P = np.concatenate([x for blk in blocks for x in (blk[0], gapP)][:-1])
    B = np.concatenate([x for blk in blocks for x in (blk[1], gapB)][:-1])
    lab = np.concatenate([x for blk in blocks for x in (blk[2], gapL)][:-1])
    return dict(players=P.astype(np.float32), ball=B.astype(np.float32), teams=teams, labels=lab,
                pitch=np.array([L, W]))


def load_idsse(match_id):
    return _cached("idsse_" + match_id, lambda: _idsse(match_id))


# -- Metrica ----------------------------------------------------------------------------------

def _metrica_csv_tracking(game):
    out = {}
    for side, t in (("Home", 0), ("Away", 1)):
        path = os.path.join(EXT, "metrica", "Sample_Game_{}_RawTrackingData_{}_Team.csv".format(game, side))
        with open(path) as fh:
            fh.readline()
            fh.readline()
            header = fh.readline().rstrip("\n").split(",")
        data = np.genfromtxt(path, delimiter=",", skip_header=3, dtype=np.float64)
        cols = [k for k, h in enumerate(header) if h.startswith("Player")]
        out[side] = dict(period=data[:, 0].astype(int), frame=data[:, 1].astype(int),
                         xy=np.stack([data[:, [c, c + 1]] for c in cols], 1),
                         ball=data[:, [header.index("Ball"), header.index("Ball") + 1]])
    h, a = out["Home"], out["Away"]
    assert np.array_equal(h["frame"], a["frame"])
    P = np.concatenate([h["xy"], a["xy"]], 1)
    teams = np.array([0] * h["xy"].shape[1] + [1] * a["xy"].shape[1], np.int8)
    return h["frame"], h["period"], P, h["ball"], teams


def _metrica_epts_tracking(game):
    meta = os.path.join(EXT, "metrica", "Sample_Game_{}_metadata.xml".format(game))
    root = ET.parse(meta).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    team_of = {p.get("id"): (0 if p.get("teamId") == team_ids[0] else 1) for p in root.iter("Player")}
    chan_player = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    # each data format specification lists, in order, the players of the tracking lines it covers
    specs = []
    for spec in root.iter("DataFormatSpecification"):
        order = [chan_player[r.get("playerChannelId")] for r in spec.iter("PlayerChannelRef")
                 if r.get("playerChannelId", "").endswith("_x")]
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")), order))
    slots = sorted({p for _, _, order in specs for p in order})
    col = {p: k for k, p in enumerate(slots)}
    teams = np.array([team_of[p] for p in slots], np.int8)
    periods = []
    params = {p.findtext("Name"): p.findtext("Value") for p in root.iter("ProviderParameter")}
    for pid, key in ((1, "first_half"), (2, "second_half")):
        a, b = params.get(key + "_start"), params.get(key + "_end")
        if a and b:
            periods.append((pid, int(a), int(b)))
    frames, P, B = [], [], []
    with open(os.path.join(EXT, "metrica", "Sample_Game_{}_tracking.txt".format(game))) as fh:
        for line in fh:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            fr = int(parts[0])
            order = next((o for a, b, o in specs if a <= fr <= b), specs[-1][2])
            row = np.full((len(slots), 2), np.nan)
            for p, g in zip(order, parts[1].split(";")):
                v = g.split(",")
                if v[0] not in ("NaN", ""):
                    row[col[p]] = (float(v[0]), float(v[1]))
            frames.append(fr)
            P.append(row)
            bv = parts[2].split(",")
            B.append([float(bv[0]), float(bv[1])] if bv[0] not in ("NaN", "") else [np.nan, np.nan])
    frames = np.array(frames)
    period = np.zeros(len(frames), int)
    for pid, a, b in periods:
        period[(frames >= a) & (frames <= b)] = pid
    return frames, period, np.array(P, dtype=np.float64), np.array(B, dtype=np.float64), teams


def _metrica_events(game):
    """[(team 0/1 or -1, type, subtype, period, start_frame, end_frame)] in time order."""
    ev = []
    if game in (1, 2):
        import pandas as pd
        e = pd.read_csv(os.path.join(EXT, "metrica", "Sample_Game_{}_RawEventsData.csv".format(game)))
        for r in e.itertuples(index=False):
            team = {"Home": 0, "Away": 1}.get(r[0], -1)
            ev.append((team, str(r[1]).upper(), "" if not isinstance(r[2], str) else r[2].upper(),
                       int(r[3]), int(r[4]), int(r[6])))
    else:
        with open(os.path.join(EXT, "metrica", "Sample_Game_{}_events.json".format(game))) as fh:
            data = json.load(fh)["data"]
        meta = ET.parse(os.path.join(EXT, "metrica", "Sample_Game_{}_metadata.xml".format(game))).getroot()
        home_id = [t.get("id") for t in meta.iter("Team")][0]
        for r in data:
            st = r.get("subtypes")
            if isinstance(st, list):
                sub = "-".join(s["name"] for s in st if s)
            elif isinstance(st, dict):
                sub = st.get("name", "")
            else:
                sub = ""
            tid = (r.get("team") or {}).get("id")
            team = -1 if tid is None else (0 if tid == home_id else 1)
            ev.append((team, r["type"]["name"].upper(), (sub or "").upper(), int(r["period"]),
                       int(r["start"]["frame"]), int(r["end"]["frame"] or r["start"]["frame"])))
    ev.sort(key=lambda x: (x[4], x[5]))
    return ev


def metrica_labels(frames, period, events):
    """Frame labels from Metrica's on-ball event log.

    * dead from the frame the ball goes out (end of BALL OUT, or of a SHOT/BALL LOST that ends
      'OUT', 'GOAL' or 'WOODWORK-GOAL'), from a whistle (FAULT RECEIVED, offside) and before the
      kick-off, until the next restart (SET PIECE, or any event whose subtype is a GOAL KICK)
    * otherwise the team of the latest on-ball action (SET PIECE, PASS, RECOVERY, SHOT, BALL LOST,
      BALL OUT); a CHALLENGE alone changes nothing, so a touch or duel without control keeps the
      previous team, as the challenge's possession rules ask
    """
    n = len(frames)
    pos = {f: k for k, f in enumerate(frames)}
    lab = np.full(n, "X", dtype="<U1")
    on_ball = {"SET PIECE", "PASS", "RECOVERY", "SHOT", "BALL LOST", "BALL OUT"}
    for per in sorted(set(period[period > 0])):
        idx = np.where(period == per)[0]
        pev = [e for e in events if e[3] == per]
        if not pev:
            continue
        kick = pos.get(pev[0][4], idx[0])
        end = idx[-1] + 1
        state = np.full(n, -2, np.int8)       # -2 unknown, -1 dead, 0/1 team
        marks = []                            # (frame index, value)
        prev_end, dead_logged = None, False
        for team, typ, sub, _, s, e in pev:
            si, ei = pos.get(s), pos.get(e)
            if si is None:
                continue
            ei = si if ei is None or ei < si else ei
            restart = typ == "SET PIECE" or "GOAL KICK" in sub
            if restart and team >= 0:
                # a restart means the ball was dead: when no stoppage was logged since the last
                # on-ball action, the ball went dead when that action ended
                if not dead_logged and prev_end is not None and prev_end + 1 < si:
                    marks.append((prev_end + 1, -1, 2))
                marks.append((si, team, 1))
                dead_logged = False
            elif typ in on_ball and team >= 0:
                marks.append((si, team, 0))
            out = (typ == "BALL OUT" or (typ in ("SHOT", "BALL LOST") and
                                         ("OUT" in sub or "GOAL" in sub.replace("GOAL KICK", ""))))
            if out:
                marks.append((ei + 1, -1, 2))
                dead_logged = True
            if typ == "FAULT RECEIVED" or "OFFSIDE" in sub:
                marks.append((si, -1, 2))
                dead_logged = True
            if typ in on_ball and not restart:
                prev_end = ei
                if not out:
                    dead_logged = False
        marks.sort(key=lambda m: (m[0], m[2] != 2))
        # play ends with the last logged action, not with the end of the tracking file
        end = min(end, max(m[0] for m in marks) + 25)
        cur, dead, dead_since = -2, True, kick
        last = kick
        for at, val, kind in marks + [(end, -3, 3)]:
            at = min(max(at, kick), end)
            state[last:at] = -1 if dead else cur
            last = at
            if kind == 3:
                break
            if kind == 2:
                if not dead:
                    dead, dead_since = True, at
            elif kind == 1:
                dead, cur = False, val
            elif not dead:
                cur = val
            elif at - dead_since > 25:
                # an on-ball action well after a stoppage with no logged restart: play resumed
                dead, cur = False, val
        seg = state[kick:end]
        lab[kick:end] = np.where(seg == -1, "D", np.where(seg == 0, "H", np.where(seg == 1, "A", "X")))
    return lab


def _metrica(game):
    if game in (1, 2):
        frames, period, P, B, teams = _metrica_csv_tracking(game)
    else:
        frames, period, P, B, teams = _metrica_epts_tracking(game)
    lab = metrica_labels(frames, period, _metrica_events(game))
    # insert a halftime gap so the halves are separate blocks
    cut = np.where(np.diff(period) != 0)[0] + 1
    parts = np.split(np.arange(len(frames)), cut)
    keep = [p for p in parts if period[p[0]] > 0]
    gapP = np.full((HALFTIME_GAP,) + P.shape[1:], np.nan)
    Ps, Bs, Ls = [], [], []
    for k, p in enumerate(keep):
        Ps.append(P[p]); Bs.append(B[p]); Ls.append(lab[p])
        if k < len(keep) - 1:
            Ps.append(gapP); Bs.append(np.full((HALFTIME_GAP, 2), np.nan)); Ls.append(np.full(HALFTIME_GAP, "X"))
    P, B, lab = np.concatenate(Ps), np.concatenate(Bs), np.concatenate(Ls)
    B = np.concatenate([B, np.full((len(B), 1), np.nan)], 1)
    return dict(players=P.astype(np.float32), ball=B.astype(np.float32), teams=teams, labels=lab,
                pitch=np.array([105.0, 68.0]))


def load_metrica(game):
    return _cached("metrica_{}".format(game), lambda: _metrica(game))


# -- SkillCorner ------------------------------------------------------------------------------

SKILLCORNER_MATCHES = (
    "1874553", "1886347", "1899585", "1925299", "1927964", "1953632", "1959846", "1986691", "1996435",
    "1996436", "2006229", "2006363", "2007448", "2007721", "2010085", "2011166", "2013725", "2015213",
    "2016236", "2017461",
)
SKILLCORNER_FPS = 10


def _skillcorner(match_id):
    """Broadcast tracking at 10 fps, resampled to 25 fps; labels from the phases of play.

    SkillCorner defines a phase of play only while the ball is in play, each with its team in
    possession, so a frame inside a phase is that team's and a frame of a period outside every phase
    is dead. Players are kept whether detected or extrapolated by SkillCorner (the competition's
    tracking has most players on most frames); the ball only where it is detected, as a tracker
    reports it.
    """
    base = os.path.join(EXT, "skillcorner", match_id)
    with open(base + "_match.json", encoding="utf-8") as fh:
        info = json.load(fh)
    L, W = float(info["pitch_length"]), float(info["pitch_width"])
    home = info["home_team"]["id"]
    persons = sorted(p["id"] for p in info["players"])
    col = {p: k for k, p in enumerate(persons)}
    teams = np.array([0 if p["team_id"] == home else 1 for p in sorted(info["players"], key=lambda p: p["id"])],
                     np.int8)
    frames = {}
    with open(base + "_tracking_extrapolated.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            frames[r["frame"]] = r
    phases = []
    with open(base + "_phases_of_play.csv", encoding="utf-8") as fh:
        import csv
        for row in csv.DictReader(fh):
            phases.append((int(row["frame_start"]), int(row["frame_end"]), int(row["team_in_possession_id"])))

    step = SKILLCORNER_FPS / 25.0
    blocks = []
    for per in sorted(info["match_periods"], key=lambda p: p["period"]):
        a, b = int(per["start_frame"]), int(per["end_frame"])
        n10 = b - a + 1
        P10 = np.full((n10, len(persons), 2), np.nan)
        B10 = np.full((n10, 3), np.nan)
        for i in range(n10):
            r = frames.get(a + i)
            if r is None:
                continue
            bd = r["ball_data"]
            if bd.get("is_detected") and bd["x"] is not None and bd["y"] is not None:
                B10[i] = ((bd["x"] + L / 2) / L, (W / 2 - bd["y"]) / W,
                          np.nan if bd.get("z") is None else bd["z"])
            for p in r["player_data"] or ():
                k = col.get(p["player_id"])
                if k is not None and p["x"] is not None and p["y"] is not None:
                    P10[i, k] = ((p["x"] + L / 2) / L, (W / 2 - p["y"]) / W)
        lab10 = np.full(n10, "D", dtype="<U1")
        for s, e, team in phases:
            s, e = max(s, a), min(e, b + 1)
            if s < e:
                lab10[s - a:e - a] = "H" if team == home else "A"
        # 10 fps -> 25 fps: positions interpolated between neighbouring frames (NaN unless both are
        # known), labels from the nearest frame
        t = np.arange(int((n10 - 1) / step) + 1) * step
        i0 = np.floor(t).astype(int)
        i1 = np.minimum(i0 + 1, n10 - 1)
        w = (t - i0)
        exact = w < 1e-9
        P = np.where(exact[:, None, None], P10[i0], P10[i0] * (1 - w)[:, None, None] + P10[i1] * w[:, None, None])
        B = np.where(exact[:, None], B10[i0], B10[i0] * (1 - w)[:, None] + B10[i1] * w[:, None])
        lab = lab10[np.minimum(np.rint(t).astype(int), n10 - 1)]
        blocks.append((P, B, lab))

    gapP = np.full((HALFTIME_GAP, len(persons), 2), np.nan)
    gapB = np.full((HALFTIME_GAP, 3), np.nan)
    gapL = np.full(HALFTIME_GAP, "X", dtype="<U1")
    P = np.concatenate([x for blk in blocks for x in (blk[0], gapP)][:-1])
    B = np.concatenate([x for blk in blocks for x in (blk[1], gapB)][:-1])
    lab = np.concatenate([x for blk in blocks for x in (blk[2], gapL)][:-1])
    return dict(players=P.astype(np.float32), ball=B.astype(np.float32), teams=teams, labels=lab,
                pitch=np.array([L, W]), tracked=np.array(True))


def load_skillcorner(match_id):
    return _cached("skillcorner_" + match_id, lambda: _skillcorner(match_id))


def controlled(labels, min_frames=25):
    """Apply the competition's written possession rule to labels that switch on every touch.

    CHALLENGE.md: a change of possession needs *controlled* possession -- a touch without control,
    a deflection, or an aerial duel won without control does not count. A team's spell shorter
    than `min_frames` that is followed by the previous team again, or by a stoppage, is such a
    touch: it is given back to the team that had the ball. A short spell right after a restart is
    kept (the restarting team has the ball by definition).
    """
    lab = labels.copy()
    for _ in range(3):
        seg = []
        k = 0
        n = len(lab)
        while k < n:
            j = k
            while j < n and lab[j] == lab[k]:
                j += 1
            seg.append((lab[k], k, j))
            k = j
        changed = False
        for i in range(1, len(seg) - 1):
            (lp, _, _), (l, a, b), (ln, _, _) = seg[i - 1], seg[i], seg[i + 1]
            if l not in "HA" or lp not in "HA" or lp == l or b - a >= min_frames:
                continue
            if ln == lp or ln == "D":
                lab[a:b] = lp
                changed = True
        if not changed:
            break
    return lab


def available():
    """List available IDSSE and Metrica training matches."""
    names = []
    for m, c in IDSSE_MATCHES.items():
        if os.path.exists(os.path.join(EXT, "idsse", "DFL_04_03_positions_raw_observed_{}_DFL-MAT-{}.xml".format(c, m))):
            names.append("idsse_" + m)
    for g in METRICA_GAMES:
        f = ("Sample_Game_{}_tracking.txt" if g == 3 else "Sample_Game_{}_RawTrackingData_Away_Team.csv").format(g)
        if os.path.exists(os.path.join(EXT, "metrica", f)):
            names.append("metrica_{}".format(g))
    return names


def available_skillcorner():
    return ["skillcorner_" + m for m in SKILLCORNER_MATCHES
            if os.path.exists(os.path.join(EXT, "skillcorner", m + "_tracking_extrapolated.jsonl"))]


def load(name):
    kind, key = name.split("_", 1)
    if kind == "skillcorner":
        return load_skillcorner(key)
    return load_idsse(key) if kind == "idsse" else load_metrica(int(key))


if __name__ == "__main__":
    for name in sys.argv[1:] or available() + available_skillcorner():
        d = load(name)
        lab = d["labels"]
        g = lab != "X"
        runs = np.flatnonzero(np.diff(np.concatenate(([0], g.astype(np.int8), [0]))))
        poss = []
        for a, b in zip(runs[::2], runs[1::2]):
            seg = lab[a:b]
            ch = np.flatnonzero(seg[1:] != seg[:-1]) + 1
            st = np.concatenate(([0], ch)); en = np.concatenate((ch, [len(seg)]))
            poss += [(e - s) / 25 for s, e in zip(st, en) if seg[s] in "HA"]
        poss = np.array(poss)
        print("{:14s} frames {:6d} graded {:6d}  dead {:.3f}  H {:.3f}  ball seen {:.3f}  players/frame {:.1f}  "
              "possessions {:4d} mean {:4.1f}s median {:4.1f}s <2s {:.2f}".format(
                  name, len(lab), g.sum(), np.mean(lab[g] == "D"), np.mean(lab[g] == "H"),
                  np.mean(~np.isnan(d["ball"][g, 0])), np.mean(np.sum(~np.isnan(d["players"][g, :, 0]), 1)),
                  len(poss), poss.mean(), np.median(poss), np.mean(poss < 2)))
