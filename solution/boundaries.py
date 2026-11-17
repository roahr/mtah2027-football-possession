"""Snap HMM dead-ball boundaries to the physical event that causes them.

The HMM works on >= 1 s smoothed movement features, so its boundaries lag the
real event by 1-4 s. Two events are directly visible in the tracking data:

* the ball crossing a line  -> the stoppage starts there;
* a restart kick: the ball, until then still/missing/at a player's feet, is
  moving away from the nearest player and is more than SEPARATION metres from
  them -> play resumes there.
"""
import numpy as np

from features import FPS, rolling_nanmean
from alive_dead import segments_from_mask

SEPARATION = 2.0      # m from the nearest player: the ball has left the taker
KICK_SPEED = 3.0      # m/s
STILL_SPEED = 2.0     # m/s: a placed ball (tracking jitter included)
STOP_BEFORE = 4 * FPS  # search window around an HMM stoppage onset
STOP_AFTER = 1 * FPS
KICK_BEFORE = 3 * FPS  # search window around an HMM restart
KICK_AFTER = 2 * FPS


def _kick_candidates(f):
    d = np.fmin(f["dH"], f["dA"])
    moving = f["have"] & (f["bspeed"] > KICK_SPEED) & (d > SEPARATION)
    still = ~f["have"] | (f["bspeed"] < STILL_SPEED) | (d < 1.0)
    # still at some point in the previous second
    k = np.ones(FPS)
    recent_still = np.convolve(still.astype(float), k, "full")[:len(still)] > 0
    prev = np.roll(moving, 1); prev[0] = False
    return np.where(moving & ~prev & np.roll(recent_still, 1))[0]


def refine(f, dead):
    dead = dead.copy()
    n = f["n"]
    out = f["have"] & (f["oob"] > 0)
    kicks = _kick_candidates(f)
    moved = []
    for a, b in segments_from_mask(dead):
        new_a, new_b = a, b
        if a > 0:
            lo, hi = max(0, a - STOP_BEFORE), min(n, a + STOP_AFTER)
            w = np.where(out[lo:hi])[0]
            if len(w):
                new_a = lo + w[0]
        if b < n:
            c = kicks[(kicks >= b - KICK_BEFORE) & (kicks <= b + KICK_AFTER) & (kicks > new_a + FPS)]
            if len(c):
                new_b = c[np.argmin(np.abs(c - b))]
        dead[new_a:a] = True
        dead[a:new_a] = False
        dead[new_b:b] = False
        dead[b:new_b] = True
        moved.append((new_a - a, new_b - b))
    return dead, np.array(moved)


def restart_kind(f, r):
    """What a predicted restart at frame r looks like, from the ball in the 4 s before it.

    Every real restart is one of a few set plays with a fixed place or a still ball: a throw-in on
    a touchline, a corner, a goal kick from the six-yard box, a kick-off from the centre spot, or a
    free kick from a stationary ball. A 'restart' in the middle of open play matches none of them.
    Returns None when the ball is not seen around r.
    """
    fps, n = 25, f["n"]
    lo, hi = max(0, r - 4 * fps), min(n, r + fps // 2)
    seen = lo + np.where(f["have"][lo:hi])[0]
    if len(seen) == 0:
        return None
    x, y = f["bx"][seen], f["by"][seen]
    if np.any(np.minimum(y, 68.0 - y) < 2.5):
        return "throw-in"
    if np.any((x < 3) | (x > 102.3)) and np.any((y < 3) | (y > 65)):
        return "corner"
    if np.any(((x < 7) | (x > 98.3)) & (y > 20) & (y < 48)):
        return "goal kick"
    if np.any((np.abs(x - 52.65) < 3) & (np.abs(y - 34.0) < 3)):
        return "kick-off"
    still = f["bspeed"][lo:r] < 1.0
    run = 0
    if still.any():
        e = np.diff(np.concatenate(([0], still.astype(np.int8), [0])))
        run = int((np.where(e == -1)[0] - np.where(e == 1)[0]).max())
    if run >= 1.5 * fps:
        return "still ball"
    return "open play"


def drop_unexplained(f, dead, max_dur=5 * FPS):
    """Remove short 'stoppages' that nothing explains.

    A stoppage needs a reason and ends in a restart. A short one (< 5 s) with the ball in view,
    well inside the pitch throughout, the broadcast showing live play, and play resuming from open
    play rather than a set-play position is a slow spell of possession, not a stoppage.
    """
    dead = dead.copy()
    n = f["n"]
    oob = np.where(f["have"], f["oob"], -99.0)
    cam = f["cam"]
    for a, b in segments_from_mask(dead):
        if b - a >= max_dur or a == 0 or b >= n or not (f["graded"][a - 1] and f["graded"][b]):
            continue
        if f["have"][a:b].mean() < 0.7:
            continue
        if np.max(oob[max(0, a - 2 * FPS):min(n, b + FPS)]) > -1.0:
            continue
        c = cam[a:min(n, b + 4 * FPS)]
        c = c[c >= 0]
        if len(c) == 0 or np.mean(c == 0) >= 0.2:
            continue
        if restart_kind(f, b) == "open play":
            dead[a:b] = False
    return dead



KICKOFF_STILL = 3 * FPS     # players stand in their own halves for at least this long
KICKOFF_GROUP = 30 * FPS    # formations this close together belong to one kick-off


def kickoffs(f):
    """Kick-offs, found from the players alone: everyone in their own half, standing still.

    Returns [(first_frame_of_formation, kick_off_frame)]. When players line up, break and line up
    again (a delayed restart), the formations are grouped and play starts when the last one breaks.
    """
    form = rolling_nanmean((f["own_half"] >= 0.85).astype(float), FPS) >= 0.8
    still = rolling_nanmean(f["pspeed"], FPS) < 1.0
    runs = [(a, b) for a, b in segments_from_mask(form & still & (f["npl"] >= 16))
            if b - a >= KICKOFF_STILL]
    out = []
    for a, b in runs:
        if out and a - out[-1][1] < KICKOFF_GROUP:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


def apply_kickoffs(f, dead):
    """The ball is dead while teams wait for a kick-off and comes alive when the formation breaks.

    Graded frames of a half before its opening kick-off are dead too (tracking can start minutes
    before the kick-off), and a stoppage still running when the kick-off formation breaks ends there.
    """
    dead = dead.copy()
    n, g = f["n"], f["graded"]
    halves = segments_from_mask(g)
    for a, k in kickoffs(f):
        half = [(h0, h1) for h0, h1 in halves if h0 <= k < h1]
        if not half:
            continue
        h0, h1 = half[0]
        if k - h0 <= FPS:          # the half starts with this kick-off
            continue
        first = not any(h0 <= k2 < k for _, k2 in kickoffs(f))
        start = h0 if first and a - h0 < 10 * 60 * FPS else max(h0, a)
        dead[start:k] = True
        e = k
        while e < n and dead[e] and g[e]:
            e += 1
        if e - k < KICKOFF_GROUP:
            dead[k:e] = False
    return dead



def fill_replay_gaps(f, dead, max_dur=8 * FPS):
    """A short 'live' spell between two stoppages, shown on TV entirely as replays, is one stoppage.

    Directors cut to replays while the ball is dead; a few seconds of apparent play between two
    stoppages that the broadcast never showed live is players jogging back, not a restart.
    """
    dead = dead.copy()
    n, cam = f["n"], f["cam"]
    for a, b in segments_from_mask(f["graded"] & ~dead):
        if b - a >= max_dur or a == 0 or b >= n or not (dead[a - 1] and dead[b]):
            continue
        c = cam[a:b]
        if np.mean(c >= 0) < 0.8 or np.mean(c[c >= 0] == 0) < 0.9:
            continue
        dead[a:b] = True
    return dead
