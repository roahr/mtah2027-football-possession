"""Per-frame physical features derived from the tactical ATD (metres, seconds)."""
import os
import sys

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "solution"))
from games import GAMES, GAME, half_video_blocks  # noqa: E402
from parse_atd import parse  # noqa: E402

FPS = 25
L, W = GAMES[GAME]["pitch"]
HOME, AWAY = 0, 1


def interp_gaps(x, max_gap):
    """Linearly fill NaN gaps of at most `max_gap` frames that have data on both sides."""
    x = x.copy()
    isn = np.isnan(x)
    if not isn.any():
        return x
    idx = np.arange(len(x))
    good = ~isn
    filled = np.interp(idx, idx[good], x[good])
    # length of each NaN run
    edges = np.diff(np.concatenate(([0], isn.astype(np.int8), [0])))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    for s, e in zip(starts, ends):
        if s > 0 and e < len(x) and e - s <= max_gap:
            x[s:e] = filled[s:e]
    return x


def rolling_nanmean(x, win):
    v = np.nan_to_num(x)
    m = (~np.isnan(x)).astype(float)
    num = uniform_filter1d(v, win, mode="nearest")
    den = uniform_filter1d(m, win, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 1e-6, num / den, np.nan)


def graded_mask(n):
    mask = np.zeros(n, bool)
    for _, a, b in half_video_blocks(GAME):
        mask[a:b + 1] = True
    return mask


def build(with_camera=False):
    """Features of the competition match from its tactical tracking.

    with_camera is an optional feature input. Submission inference uses tactical tracking.
    """
    d = parse("tactical")
    n = len(d["ball"])
    cam = None
    if with_camera:
        try:
            import broadcast
            cam = broadcast.build(verbose=False)["cam"].astype(np.int8)
        except FileNotFoundError:
            pass
    return build_from(d["players"], d["ball"], d["teams"], graded_mask(n), cam)


def build_from(players, ball, teams, graded, cam=None, pitch=(L, W)):
    """Per-frame features from tracking arrays of any match.

    players (N, K, 3) and ball (N, >=2) are in normalised pitch coordinates, as in the ATD; the
    third player channel is speed in m/s (NaN where unknown). `graded` marks the frames of play,
    `cam` is the broadcast camera state (-1 where there is none).
    """
    L, W = pitch
    P = np.asarray(players, dtype=np.float64)
    B = np.asarray(ball, dtype=np.float64)
    n = len(B)
    if cam is None:
        cam = np.full(n, -1, np.int8)

    px, py, ps = P[:, :, 0] * L, P[:, :, 1] * W, P[:, :, 2]
    present = ~np.isnan(px)

    # --- ball -------------------------------------------------------------------------------
    seen = ~np.isnan(B[:, 0])
    bx_raw, by_raw = B[:, 0] * L, B[:, 1] * W
    bx = interp_gaps(bx_raw, 8)
    by = interp_gaps(by_raw, 8)
    have = ~np.isnan(bx)
    sbx = np.where(have, median_filter(np.nan_to_num(bx), 5), np.nan)
    sby = np.where(have, median_filter(np.nan_to_num(by), 5), np.nan)
    bvx = np.gradient(sbx) * FPS
    bvy = np.gradient(sby) * FPS
    bspeed = np.hypot(bvx, bvy)
    bspeed[bspeed > 45] = np.nan  # teleports are tracker errors, not shots

    oob = np.maximum.reduce([-sbx, sbx - L, -sby, sby - W])  # >0 means outside the lines

    # --- players relative to ball --------------------------------------------------------------
    dx = px - sbx[:, None]
    dy = py - sby[:, None]
    dist = np.hypot(dx, dy)
    dist[~present] = np.inf
    dH = np.min(np.where(teams[None, :] == HOME, dist, np.inf), axis=1)
    dA = np.min(np.where(teams[None, :] == AWAY, dist, np.inf), axis=1)
    dH[~have] = np.nan
    dA[~have] = np.nan
    dH[np.isinf(dH)] = np.nan
    dA[np.isinf(dA)] = np.nan
    n_within5 = np.sum(dist < 5.0, axis=1).astype(float)

    # --- collective movement ----------------------------------------------------------------
    pspeed = np.nanmean(np.where(present, ps, np.nan), axis=1)
    # speed channel can be NaN while position exists; fall back to finite differences
    vx = np.gradient(px, axis=0) * FPS
    vy = np.gradient(py, axis=0) * FPS
    pspeed_fd = np.nanmedian(np.where(present, np.hypot(vx, vy), np.nan), axis=1)
    pspeed = np.where(np.isnan(pspeed), pspeed_fd, pspeed)
    npl = present.sum(1).astype(float)

    # team shape (for the self-supervised team classifier)
    shape = {}
    for t, name in ((HOME, "h"), (AWAY, "a")):
        m = present & (teams[None, :] == t)
        x = np.where(m, px, np.nan)
        y = np.where(m, py, np.nan)
        shape[name + "_cx"] = np.nanmean(x, 1)
        shape[name + "_cy"] = np.nanmean(y, 1)
        shape[name + "_sx"] = np.nanstd(x, 1)
        shape[name + "_sy"] = np.nanstd(y, 1)
        shape[name + "_vx"] = np.nanmean(np.where(m, vx, np.nan), 1)
        shape[name + "_spd"] = np.nanmean(np.where(m, ps, np.nan), 1)
        shape[name + "_n"] = m.sum(1).astype(float)

    # share of players standing in their own half (all of them do at a kick-off). Which half is
    # whose comes from where each team's players are on average in each graded half
    hx = shape["h_cx"]
    ax_ = shape["a_cx"]
    sign = np.zeros(n)
    edges = np.diff(np.concatenate(([0], graded.astype(np.int8), [0])))
    blocks = list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0]))
    for a, b in blocks:
        sign[a:b] = 1.0 if np.nanmean(ax_[a:b] - hx[a:b]) > 0 else -1.0
    known = np.where(sign != 0)[0]
    sign = sign[known[np.clip(np.searchsorted(known, np.arange(n)), 0, len(known) - 1)]]
    mid = L / 2
    own = np.where(teams[None, :] == HOME, (px - mid) * sign[:, None] < 1.0,
                   (px - mid) * sign[:, None] > -1.0)
    own_half = np.sum(own & present, 1) / np.maximum(present.sum(1), 1)

    return dict(
        n=n, seen=seen, have=have, bx=sbx, by=sby, bspeed=bspeed, oob=oob,
        dH=dH, dA=dA, n_within5=n_within5, pspeed=pspeed, npl=npl, cam=cam, own_half=own_half,
        graded=graded, **shape,
    )


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    f = build()
    g = f["graded"]
    for k, v in f.items():
        if isinstance(v, np.ndarray) and v.dtype != bool:
            print("{:10s} nan {:.2f}  median {:.2f}".format(k, np.mean(np.isnan(v[g])), np.nanmedian(v[g])))
