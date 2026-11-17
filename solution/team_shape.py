"""Collective-team geometry for the externally trained possession models.

The release uses shape_features and attack_sign to derive deterministic team-position and
movement features. These features feed the packaged ownership models. The pseudo-label and
cross-fitting functions below describe a separate experimental classifier; that classifier
is not fitted or used by the submission entry point.
"""
import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.ensemble import HistGradientBoostingClassifier

from features import FPS, L, rolling_nanmean
from half_utils import half_blocks


def attack_sign(f):
    """+1 where home attacks towards +x, per frame (derived from where each team sits)."""
    sign = np.zeros(f["n"])
    for a, b in half_blocks(f):
        sign[a:b] = 1.0 if np.nanmean(f["a_cx"][a:b] - f["h_cx"][a:b]) > 0 else -1.0
    return sign


def shape_features(f):
    s = attack_sign(f)

    def ax(x):  # x along home's attacking direction, 0 = home goal
        return np.where(s > 0, x, L - x)

    cols = {}
    hx, axx = ax(f["h_cx"]), ax(f["a_cx"])
    cols["h_x"], cols["a_x"], cols["dx"] = hx, axx, hx - axx
    cols["h_sx"], cols["h_sy"], cols["a_sx"], cols["a_sy"] = f["h_sx"], f["h_sy"], f["a_sx"], f["a_sy"]
    cols["area_ratio"] = np.log((f["h_sx"] * f["h_sy"] + 1) / (f["a_sx"] * f["a_sy"] + 1))
    cols["h_vx"] = s * f["h_vx"]            # home moving towards the away goal
    cols["a_vx"] = -s * f["a_vx"]           # away moving towards the home goal
    cols["h_spd"], cols["a_spd"] = f["h_spd"], f["a_spd"]
    cols["spd_diff"] = f["h_spd"] - f["a_spd"]
    bx = ax(f["bx"])
    cols["ball_x"] = bx
    cols["ball_y_c"] = np.abs(f["by"] - 34.0)
    cols["ball_seen"] = uniform_filter1d(f["seen"].astype(float), FPS, mode="nearest")
    # last known ball position, held through gaps, and how stale it is
    idx = np.where(f["have"], np.arange(f["n"]), 0)
    idx = np.maximum.accumulate(idx)
    cols["ball_x_last"] = bx[idx]
    cols["ball_age"] = np.minimum((np.arange(f["n"]) - idx) / FPS, 30.0)

    X = []
    for k, v in cols.items():
        X.append(rolling_nanmean(v, FPS))
        if k in ("dx", "h_vx", "a_vx", "spd_diff", "area_ratio", "h_x", "a_x"):
            X.append(rolling_nanmean(v, 4 * FPS))
    return np.stack(X, 1)


def pseudo_labels(f, team, events, guard=FPS):
    """+1/0 on frames where the state machine is confident, -1 elsewhere."""
    y = np.full(f["n"], -1, np.int8)
    alive = team >= 0
    y[alive] = team[alive]
    # drop frames near a change of possession or near a dead ball
    change = np.where(np.diff(team.astype(int)) != 0)[0]
    for c in change:
        y[max(0, c - guard):c + guard + 1] = -1
    # keep only frames within 4 s of a confirmed control event of the labelled team
    near = np.zeros(f["n"], bool)
    for a, b, t in events:
        lo, hi = max(0, a - 4 * FPS), min(f["n"], b + 4 * FPS)
        near[lo:hi] |= team[lo:hi] == t
    y[~near] = -1
    return y


def cross_fit(X, y, graded, n_folds=6, seed=0, n_bags=4):
    """Out-of-fold P(home) for every graded frame; folds are contiguous time blocks.

    Each fold averages `n_bags` models trained on interleaved subsets of the training frames
    (every n_bags-th frame, different offsets): neighbouring frames are near-duplicates, so this
    loses little information but makes the output far less sensitive to small changes in the
    pseudo-labels.
    """
    p = np.full(len(y), np.nan)
    idx = np.where(graded)[0]
    folds = np.array_split(idx, n_folds)
    for k, test in enumerate(folds):
        train = np.concatenate([folds[j] for j in range(n_folds) if j != k])
        train = train[y[train] >= 0]
        acc = np.zeros(len(test))
        for b in range(n_bags):
            sub = train[b::n_bags]
            clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                                 l2_regularization=1.0, random_state=seed + b)
            clf.fit(X[sub], y[sub])
            # class 1 is AWAY in our coding; accumulate P(home)
            acc += clf.predict_proba(X[test])[:, list(clf.classes_).index(0)]
        p[test] = acc / n_bags
    return p
