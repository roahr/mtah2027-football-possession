"""Put the broadcast recording on the tactical timeline.

The broadcast tracking uses the same normalised pitch coordinates as the tactical tracking, but its
own frame counter, with cuts, replays and close-ups. It is a live feed, so each half is the tactical
timeline plus a (nearly) constant offset. Everything below is found from the data:

  1. coarse offsets   per 100 s window, the shift that best overlays the two ball tracks (FFT
                      cross-correlation); windows that agree form one span per half
  2. fine mapping     per 5 s window, the shift (+-OFFSET_SEARCH frames) that best overlays the two
                      sets of players; a robust straight line through those shifts per span
  3. camera state     each broadcast frame is LIVE if its players sit on the tactical players at
                      the mapped frame, NO_MATCH if it has a pitch homography but shows something
                      else (a replay or a tight close-up), CUT if it has no usable homography
                      (replays, close-ups, crowd shots)

TV directors cut away from live play mostly when the ball is dead, so the camera state is an
independent cue for the alive/dead model. The broadcast ball is kept too, but only on LIVE frames.

Output on the tactical timeline: `cam` (-1 not covered, CUT, NO_MATCH, LIVE) and `ball` (NaN unless
the broadcast saw the ball on a LIVE frame).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))

from parse_atd import parse, CACHE  # noqa: E402

L, W = 105.3, 68.0
UNCOVERED, CUT, NO_MATCH, LIVE = -1, 0, 1, 2
COARSE_WIN = 2500        # frames per coarse window
COARSE_RMS = 0.05        # normalised units: worse best-overlay than this is noise (replays etc.)
FINE_WIN = 125           # frames per fine window
OFFSET_SEARCH = 8        # frames searched either side of a span's coarse offset
MAX_REPROJ = 5.0         # px: broadcast homography error above this is not trusted
LIVE_MATCH = 1.5         # m: median distance broadcast player -> nearest tactical player when live
MIN_PLAYERS = 4          # broadcast players needed to judge a frame


def _coarse_spans(T, B):
    """[(b_start, b_end, offset)] -- broadcast ranges that share one coarse offset."""
    mt = ~np.isnan(T[:, 0])
    Tz = np.nan_to_num(T)
    n_fft = 1 << int(np.ceil(np.log2(len(T) + COARSE_WIN)))
    fT = {k: np.fft.rfft(v, n_fft) for k, v in dict(
        m=mt.astype(float), x=Tz[:, 0], y=Tz[:, 1], s=(Tz ** 2).sum(1) * mt).items()}

    def xc(fa, b):
        return np.fft.irfft(fa * np.conj(np.fft.rfft(b, n_fft)), n_fft)

    found = []
    for s in range(0, len(B) - COARSE_WIN, COARSE_WIN):
        Bw = B[s:s + COARSE_WIN]
        mb = ~np.isnan(Bw[:, 0])
        if mb.sum() < 300:
            continue
        Bz = np.nan_to_num(Bw)
        cnt = xc(fT["m"], mb.astype(float))
        mse = (xc(fT["s"], mb.astype(float)) + xc(fT["m"], (Bz ** 2).sum(1) * mb)
               - 2 * (xc(fT["x"], Bz[:, 0]) + xc(fT["y"], Bz[:, 1]))) / np.maximum(cnt, 1)
        mse[cnt < 0.3 * mb.sum()] = np.inf
        k = int(np.argmin(mse[:len(T)]))
        if np.sqrt(mse[k]) < COARSE_RMS:
            found.append((s, k - s))
    found = np.array(found)
    order = np.argsort(found[:, 1], kind="stable")
    clusters, cur = [], [order[0]]
    for a, b in zip(order[:-1], order[1:]):
        if found[b, 1] - found[a, 1] <= 60:
            cur.append(b)
        else:
            clusters.append(cur)
            cur = [b]
    clusters.append(cur)
    spans = [(int(found[c, 0].min()), int(found[c, 0].max()) + COARSE_WIN, int(np.median(found[c, 1])))
             for c in clusters if len(c) >= 5]
    return sorted(spans)


def _player_cost(Pb, Pt):
    """Per frame: median distance (m) broadcast player -> nearest tactical player, and #players."""
    d = np.hypot((Pb[:, :, None, 0] - Pt[:, None, :, 0]) * L, (Pb[:, :, None, 1] - Pt[:, None, :, 1]) * W)
    d = np.where(np.isnan(d), np.inf, d)
    nn = d.min(2)
    nn[~np.isfinite(nn)] = np.nan
    n = np.sum(~np.isnan(nn), 1)
    with np.errstate(all="ignore"):
        med = np.nanmedian(nn, 1)
    return med, n


def _cost_at(Pb, Pt, b_idx, t_idx, chunk=4096):
    med = np.full(len(b_idx), np.nan)
    n = np.zeros(len(b_idx), int)
    for s in range(0, len(b_idx), chunk):
        m, k = _player_cost(Pb[b_idx[s:s + chunk]], Pt[t_idx[s:s + chunk]])
        med[s:s + chunk], n[s:s + chunk] = m, k
    return med, n


def _fit_span(Pb, Pt, usable, a, b, base, n_t):
    """Robust line offset(b) = c0 + c1*b over one span, from per-window best player overlays."""
    xs, ys = [], []
    for s in range(max(0, a - COARSE_WIN), min(len(Pb), b + COARSE_WIN), FINE_WIN):
        idx = np.arange(s, min(len(Pb), s + FINE_WIN))
        idx = idx[usable[idx]]
        if len(idx) < 10:
            continue
        best = None
        for o in range(base - OFFSET_SEARCH, base + OFFSET_SEARCH + 1):
            t = idx + o
            ok = (t >= 0) & (t < n_t)
            if ok.sum() < 10:
                continue
            med, n = _player_cost(Pb[idx[ok]], Pt[t[ok]])
            m = n >= MIN_PLAYERS
            if m.sum() < 10:
                continue
            c = float(np.median(med[m]))
            if best is None or c < best[0]:
                best = (c, o)
        if best is not None and best[0] < LIVE_MATCH:
            xs.append(s + FINE_WIN / 2)
            ys.append(best[1])
    xs, ys = np.array(xs), np.array(ys, float)
    keep = np.abs(ys - np.median(ys)) <= 3
    for _ in range(3):
        c1, c0 = np.polyfit(xs[keep], ys[keep], 1)
        keep = np.abs(ys - (c0 + c1 * xs)) <= 2
    return c0, c1


def build(verbose=True):
    out_path = os.path.join(CACHE, "broadcast_on_tactical.npz")
    if os.path.exists(out_path):
        return dict(np.load(out_path))
    import load_homography
    td, bd = parse("tactical"), parse("broadcast")
    T, B = td["ball"][:, :2].astype(float), bd["ball"][:, :2].astype(float)
    Pt, Pb = td["players"][:, :, :2].astype(float), bd["players"][:, :, :2].astype(float)
    _, err, valid = load_homography.load(source="broadcast")
    n_t, n_b = len(T), len(B)
    usable = valid.astype(bool) & (err < MAX_REPROJ)

    spans = _coarse_spans(T, B)
    lines = [(a, b) + _fit_span(Pb, Pt, usable, a, b, o, n_t) for a, b, o in spans]
    if verbose:
        for a, b, c0, c1 in lines:
            print("  broadcast ~{}-{}: tactical = broadcast + {:.2f} {:+.3f}/10k frames".format(
                a, b, c0, c1 * 1e4))

    # each broadcast frame belongs to one span; where two spans could both apply, switch at the
    # point that best explains which mapping the players follow
    bidx = np.arange(n_b)
    owner = np.full(n_b, -1)
    for k, (a, b, c0, c1) in enumerate(lines):
        lo = 0 if k == 0 else (lines[k - 1][1] + a) // 2
        hi = n_b if k == len(lines) - 1 else None
        owner[lo:hi] = k
    for k in range(len(lines) - 1):
        a0, b0, c00, c10 = lines[k]
        a1, b1, c01, c11 = lines[k + 1]
        lo, hi = b0 - COARSE_WIN, a1 + COARSE_WIN
        zone = np.arange(max(0, lo), min(n_b, hi))
        zone = zone[usable[zone]]
        if len(zone) == 0:
            continue
        good = []
        for c0, c1 in ((c00, c10), (c01, c11)):
            t = zone + np.round(c0 + c1 * zone).astype(int)
            ok = (t >= 0) & (t < n_t)
            med, n = _cost_at(Pb, Pt, zone[ok], t[ok])
            g = np.zeros(len(zone), bool)
            g[np.where(ok)[0]] = (n >= MIN_PLAYERS) & (med < LIVE_MATCH)
            good.append(g)
        # switch index maximising matches of span k before it and span k+1 after it
        score = np.concatenate(([0], np.cumsum(good[0]))) + \
            (good[1].sum() - np.concatenate(([0], np.cumsum(good[1]))))
        cut = zone[min(int(np.argmax(score)), len(zone) - 1)]
        owner[max(0, lo):cut] = k
        owner[cut:min(n_b, hi)] = k + 1

    c0 = np.array([lines[k][2] for k in owner])
    c1 = np.array([lines[k][3] for k in owner])
    tgt = bidx + np.round(c0 + c1 * bidx).astype(int)
    inside = (tgt >= 0) & (tgt < n_t)
    state = np.full(n_b, CUT)
    chk = np.where(inside & usable)[0]
    med, n = _cost_at(Pb, Pt, chk, tgt[chk])
    state[chk] = np.where((n >= MIN_PLAYERS) & (med < LIVE_MATCH), LIVE, NO_MATCH)

    cam = np.full(n_t, UNCOVERED, np.int8)
    ball = np.full((n_t, 2), np.nan)
    w = np.where(inside)[0]
    cam[tgt[w]] = state[w]
    lv = w[(state[w] == LIVE) & ~np.isnan(B[w, 0])]
    ball[tgt[lv]] = B[lv]
    if verbose:
        print("  tactical frames covered {:.3f}, live {:.3f}".format(np.mean(cam >= 0), np.mean(cam == LIVE)))
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(out_path, cam=cam, ball=ball)
    return dict(cam=cam, ball=ball)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    from features import graded_mask
    d = build()
    T = parse("tactical")["ball"][:, :2].astype(float)
    g = graded_mask(len(T))
    for k, name in ((UNCOVERED, "uncovered"), (CUT, "cut"), (NO_MATCH, "no match"), (LIVE, "live")):
        print("  graded frames {:10s} {:.3f}".format(name, np.mean(d["cam"][g] == k)))
    bt, bb = ~np.isnan(T[:, 0]), ~np.isnan(d["ball"][:, 0])
    both = bt & bb & g
    dist = np.hypot((T[both, 0] - d["ball"][both, 0]) * L, (T[both, 1] - d["ball"][both, 1]) * W)
    print("  ball: tactical {:.3f} of graded, broadcast adds {:.3f}; both seen {} frames, median gap {:.2f} m".format(
        np.mean(bt[g]), np.mean((bb & ~bt)[g]), both.sum(), np.median(dist)))
