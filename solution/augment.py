"""Make clean tracking from other matches look like automatic, uncorrected tracking.

The external matches (external.py) are corrected, complete tracking; the competition's tracking is
automatic. To train and validate on them, their tracking is degraded with randomised artifacts.
Every degraded copy draws its own noise level from wide ranges, so the models learn to cope with
any plausible camera instead of being calibrated on one. Nothing here is measured on the provided
match: the ranges are anchored on the statistics published in DATA.md (ball present on 59% of
graded frames, ~20.5 of 22 players tracked, duplicated and switched tracks) and on the fact that a
ball out of play, in the air or in a crowd is the one a single camera loses.

  ball dropout    two-state (seen / missing) Markov chain whose rates depend on whether the ball
                  is in play, in the air, or dead
  player dropout  players missing in runs
  ghosts          a second track duplicating a player, sometimes under the other team (an
                  identity switch)
  speed channel   recomputed from the degraded positions, like a tracker would

All draws come from a seeded generator, so a degraded match is reproducible.
"""
import numpy as np

from features import FPS

VERSION = "r1"   # part of every cache key built on degraded data

# (low, high) of a uniform draw per degraded copy; run lengths in frames
RANGES = dict(
    dead_missing=(0.40, 0.90),        # share of dead-ball frames without a ball
    dead_missing_run=(150.0, 600.0),
    alive_missing=(0.05, 0.30),       # share of live-play frames without a ball
    alive_missing_run=(30.0, 200.0),
    air_missing=(0.30, 0.80),         # share of frames with the ball high in the air (when known)
    air_missing_run=(20.0, 90.0),
    player_missing_frac=(0.03, 0.12),
    player_missing_run=(25.0, 200.0),
    ghost_rate=(0.02, 0.12),
    ghost_run=(40.0, 200.0),
    ghost_wrong_team=(0.2, 0.8),
)
AIR_HEIGHT = 2.5   # m


def random_profile(rng):
    u = {k: float(rng.uniform(*v)) for k, v in RANGES.items()}
    prof = dict(ball_missing_run={}, ball_visible_run={}, air_height=AIR_HEIGHT)
    for state, key in (("dead", "dead"), ("alive", "alive"), ("air", "air")):
        m, run = u[key + "_missing"], u[key + "_missing_run"]
        prof["ball_missing_run"][state] = run
        prof["ball_visible_run"][state] = run * (1 - m) / m
    for k in ("player_missing_frac", "player_missing_run", "ghost_rate", "ghost_run", "ghost_wrong_team"):
        prof[k] = u[k]
    return prof


def _markov_mask(rng, n, p_on, p_off, start_on=False):
    """Boolean run mask from per-frame switching probabilities p_on (off->on), p_off (on->off)."""
    u = rng.random(n)
    out = np.empty(n, bool)
    s = start_on
    for t in range(n):
        if s:
            s = u[t] >= p_off[t]
        else:
            s = u[t] < p_on[t]
        out[t] = s
    return out


def _speed(px, py):
    """Per-player speed (m/s) from positions, the way a tracker reports it."""
    vx = np.gradient(px, axis=0) * FPS
    vy = np.gradient(py, axis=0) * FPS
    sp = np.hypot(vx, vy)
    k = np.ones(5) / 5
    out = np.full_like(sp, np.nan)
    for j in range(sp.shape[1]):
        col = sp[:, j]
        ok = ~np.isnan(col)
        if ok.sum() < 5:
            continue
        num = np.convolve(np.where(ok, col, 0.0), k, "same")
        den = np.convolve(ok.astype(float), k, "same")
        out[:, j] = np.where(ok & (den > 0), num / np.maximum(den, 1e-9), np.nan)
    return out


def as_tracked(d):
    """A match whose tracking is automatic already (SkillCorner's broadcast tracking): no artificial
    noise, only the speed channel a tracker reports."""
    L, W = d["pitch"]
    P = d["players"].astype(np.float64)
    speed = _speed(P[:, :, 0] * L, P[:, :, 1] * W)
    return dict(players=np.concatenate([P, speed[:, :, None]], 2), ball=d["ball"].astype(np.float64),
                teams=d["teams"].astype(np.int8), labels=d["labels"], pitch=(L, W))


def degrade(d, seed, profile=None, strength=1.0):
    """Degraded copy of an external match: players (N, K+2, 3) with speed, ball (N, 3).

    The noise level is drawn from RANGES with the same seed unless `profile` is given. Matches that
    carry real tracking noise already (d["tracked"]) are returned as tracked.
    """
    if bool(d.get("tracked", False)):
        return as_tracked(d)
    rng = np.random.default_rng(seed)
    if profile is None:
        profile = random_profile(rng)
    L, W = d["pitch"]
    lab = d["labels"]
    P = d["players"].astype(np.float64).copy()
    B = d["ball"].astype(np.float64).copy()
    n, K = P.shape[:2]
    teams = d["teams"].astype(np.int8)

    # ball dropout ------------------------------------------------------------------------------
    state = np.where(lab == "D", "dead", "alive")
    z = B[:, 2]
    air = ~np.isnan(z) & (z > profile["air_height"]) & (state == "alive")
    state = np.where(air, "air", state)
    p_miss = np.zeros(n)
    p_back = np.zeros(n)
    for s in ("alive", "dead", "air"):
        m = state == s
        p_miss[m] = strength / profile["ball_visible_run"][s]
        p_back[m] = 1.0 / profile["ball_missing_run"][s]
    missing = _markov_mask(rng, n, p_miss, p_back)
    B[missing, :2] = np.nan

    # player dropout ---------------------------------------------------------------------------
    f = profile["player_missing_frac"] * strength
    run = profile["player_missing_run"]
    p_back = 1.0 / run
    p_miss = f / ((1 - f) * run)
    u = rng.random((n, K))
    gone = np.zeros(K, bool)
    drop = np.zeros((n, K), bool)
    for t in range(n):
        gone = np.where(gone, u[t] >= p_back, u[t] < p_miss)
        drop[t] = gone
    P[drop] = np.nan

    # ghost tracks: two extra slots, one per team, that sometimes copy a player -------------------
    ghosts = np.full((n, 2, 2), np.nan)
    g_on = _markov_mask(rng, n, np.full(n, profile["ghost_rate"] * strength /
                                        ((1 - profile["ghost_rate"]) * profile["ghost_run"])),
                        np.full(n, 1.0 / profile["ghost_run"]))
    edges = np.diff(np.concatenate(([0], g_on.astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        src = rng.integers(K)
        wrong = rng.random() < profile["ghost_wrong_team"]
        slot = (1 - teams[src]) if wrong else teams[src]
        off = rng.normal(0, 0.8, 2) / np.array([L, W])
        ghosts[a:b, slot] = P[a:b, src] + off
        if wrong and rng.random() < 0.5:
            P[a:b, src] = np.nan          # an identity switch: the player now sits on the other team
    P = np.concatenate([P, ghosts], 1)
    teams = np.concatenate([teams, np.array([0, 1], np.int8)])

    speed = _speed(P[:, :, 0] * L, P[:, :, 1] * W)
    players = np.concatenate([P, speed[:, :, None]], 2)
    return dict(players=players, ball=B, teams=teams, labels=lab, pitch=(L, W))
