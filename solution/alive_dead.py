"""Segmentation utilities and experimental unsupervised status modelling.

The release decoder uses segments_from_mask to form contiguous intervals. Its live/dead
probabilities come from the packaged, externally trained tree and GRU models. The HMM fitting
functions in this module are retained from earlier experiments and are not called by the
submission entry point.
"""
import numpy as np
from scipy.ndimage import uniform_filter1d, maximum_filter1d, minimum_filter1d

from features import FPS, rolling_nanmean

STEP = 5  # run the HMM at 5 Hz; features are smoothed over >= 1 s anyway


def _digitize(x, edges):
    """Bin index, with NaN mapped to an extra last bin."""
    out = np.digitize(x, edges)
    out[np.isnan(x)] = len(edges) + 1
    return out


def observation_matrix(f, use_cam=True, p_sup=None):
    seen = f["seen"].astype(float)
    feats = []
    # collective speed: the strongest single cue, players walk during stoppages
    ps = rolling_nanmean(f["pspeed"], 2 * FPS)
    feats.append(_digitize(ps, [0.8, 1.0, 1.2, 1.4, 1.6, 1.9, 2.2, 2.6, 3.0, 3.5]))
    # change in collective speed over the last/next 3 s (deceleration at a whistle)
    ps_past = rolling_nanmean(np.concatenate((np.full(3 * FPS, np.nan), f["pspeed"][:-3 * FPS])), 3 * FPS)
    feats.append(_digitize(ps - ps_past, [-1.0, -0.5, -0.2, 0.2, 0.5, 1.0]))
    # ball visibility over 2 s
    vis = uniform_filter1d(seen, 2 * FPS, mode="nearest")
    feats.append(_digitize(vis, [0.05, 0.3, 0.6, 0.9]))
    # ball speed (rolling median-ish over 1 s)
    bs = rolling_nanmean(f["bspeed"], FPS)
    feats.append(_digitize(bs, [1.0, 2.5, 5.0, 8.0, 12.0, 18.0]))
    # ball outside the lines, ever in the last 2 s
    oob = np.nan_to_num(f["oob"], nan=-99.0)
    oob_recent = maximum_filter1d(oob, 2 * FPS, origin=FPS - 1)
    feats.append(_digitize(np.where(np.isnan(f["oob"]) & (oob_recent < -50), np.nan, oob_recent),
                           [-3.0, -1.0, 0.3, 1.5]))
    # how crowded the ball is (set pieces in the box, walls) -- only where the ball is seen
    feats.append(_digitize(np.where(f["have"], f["n_within5"], np.nan), [0.5, 1.5, 3.5, 6.5]))
    ignore = [False, False, False, True, True, True]
    # broadcast camera: replays and close-ups are shown mostly while the ball is dead. Uncovered
    # frames carry no information, like a missing ball
    if use_cam:
        cam = np.where(f["cam"] >= 0, f["cam"], np.nan).astype(float)
        feats.append(_digitize(cam, [0.5, 1.5]))
        ignore.append(True)
    # P(dead) from the model trained on other matches (supervised.py), self-calibrated here like
    # every other observation
    if p_sup is not None:
        feats.append(_digitize(p_sup, [0.1, 0.3, 0.5, 0.7, 0.9]))
        ignore.append(True)
    # ball-derived features carry no information when the ball is missing: visibility already
    # says that once, and repeating it four times would let it outvote the players
    ignore_nan = np.array(ignore)
    return np.stack(feats, 1), ignore_nan


def _forward_backward(logB, logA, logpi):
    T, S = logB.shape
    la = np.empty((T, S))
    la[0] = logpi + logB[0]
    for t in range(1, T):
        la[t] = logB[t] + np.logaddexp.reduce(la[t - 1][:, None] + logA, axis=0)
    lb = np.zeros((T, S))
    for t in range(T - 2, -1, -1):
        lb[t] = np.logaddexp.reduce(logA + (logB[t + 1] + lb[t + 1])[None, :], axis=1)
    ll = np.logaddexp.reduce(la[-1])
    post = np.exp(la + lb - ll)
    xi = np.zeros((S, S))
    for t in range(T - 1):
        m = la[t][:, None] + logA + (logB[t + 1] + lb[t + 1])[None, :] - ll
        xi += np.exp(m)
    return post, xi, ll


def fit_alive_dead(f, n_iter=25, verbose=True, use_cam=True, p_sup=None):
    """Return per-frame posterior P(dead) over the whole video (NaN outside graded halves)."""
    O, ignore_nan = observation_matrix(f, use_cam, p_sup)
    n = f["n"]
    graded = f["graded"]
    # contiguous graded blocks, subsampled
    blocks = []
    edges = np.diff(np.concatenate(([0], graded.astype(np.int8), [0])))
    for a, b in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
        blocks.append(np.arange(a, b, STEP))
    nbins = O.max(0) + 1
    S = 2
    # initialise from the speed cue alone: slow means dead
    ps = rolling_nanmean(f["pspeed"], 2 * FPS)
    init = [(ps[ix] < 1.3).astype(float) for ix in blocks]
    emis = []
    for k in range(O.shape[1]):
        e = np.ones((S, nbins[k]))
        for ix, g in zip(blocks, init):
            for s, w in ((0, 1 - g), (1, g)):
                np.add.at(e[s], O[ix, k], w)
        if ignore_nan[k]:
            e[:, -1] = 0.0
        emis.append(e / e.sum(1, keepdims=True))
    for k in np.where(ignore_nan)[0]:
        emis[k][:, -1] = 1.0
    stay = 1 - 1.0 / (15 * FPS / STEP)  # mean dwell ~15 s a priori
    A = np.array([[stay, 1 - stay], [1 - stay, stay]])
    pi = np.array([0.5, 0.5])

    for it in range(n_iter):
        acc = [np.full((S, nb), 1e-3) for nb in nbins]
        xi_tot = np.full((S, S), 1e-3)
        ll_tot = 0.0
        posts = []
        for ix in blocks:
            logB = sum(np.log(emis[k][:, O[ix, k]]).T for k in range(O.shape[1]))
            post, xi, ll = _forward_backward(logB, np.log(A), np.log(pi))
            posts.append(post)
            xi_tot += xi
            ll_tot += ll
            for k in range(O.shape[1]):
                for s in range(S):
                    np.add.at(acc[k][s], O[ix, k], post[:, s])
        for k in np.where(ignore_nan)[0]:
            acc[k][:, -1] = 0.0
        emis = [a / a.sum(1, keepdims=True) for a in acc]
        for k in np.where(ignore_nan)[0]:
            emis[k][:, -1] = 1.0
        A = xi_tot / xi_tot.sum(1, keepdims=True)
        if verbose and (it % 5 == 0 or it == n_iter - 1):
            print("  hmm iter {:2d}  loglik {:.1f}  stay {:.4f} {:.4f}".format(it, ll_tot, A[0, 0], A[1, 1]))

    # name the states: dead is the one whose players move slower
    mean_speed_bin = [np.sum(emis[0][s] * np.arange(nbins[0])) for s in range(S)]
    dead_state = int(np.argmin(mean_speed_bin))
    p_dead = np.full(n, np.nan)
    for ix, post in zip(blocks, posts):
        a, b = ix[0], ix[-1] + STEP
        full = np.arange(a, min(b, n))
        p_dead[full] = np.interp(full, ix, post[:, dead_state])
    model = dict(emis=emis, A=A, dead_state=dead_state)
    return p_dead, model


def segments_from_mask(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0]))


def dead_mask(f, p_dead, thr=0.5, min_dead=2 * FPS, min_alive=2 * FPS):
    """Threshold the posterior, then clean up very short runs of either state."""
    dead = (p_dead > thr) & f["graded"]
    # a ball clearly seen outside the lines is dead, whatever the HMM says
    oob = np.nan_to_num(f["oob"], nan=-99.0) > 1.0
    dead |= oob & f["graded"]
    for a, b in segments_from_mask(~dead & f["graded"]):
        if b - a < min_alive:
            dead[a:b] = True
    for a, b in segments_from_mask(dead):
        if b - a < min_dead:
            dead[a:b] = False
    return dead
