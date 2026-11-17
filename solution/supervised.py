"""Supervised evidence learned on OTHER matches (external.py), applied to any match.

Two frame classifiers, trained on the public external matches after degrading their tracking to
look like the competition's (augment.py):

  dead model   P(ball dead | collective movement, ball visibility, ball speed, ball position,
               crowding, kick-off shape)
  team model   P(home team in possession | ball-to-player distances, recent and upcoming ball
               control by each team, team shapes) -- trained on alive frames, with the
               competition's 'controlled possession' rule applied to the labels

Neither replaces the match's own models: their outputs enter the alive/dead HMM as one more
observation and the team Viterbi as one more evidence term, so evidence specific to the
competition match (the broadcast camera, its own team-shape model) still counts.
"""
import os
import sys

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d
from sklearn.ensemble import HistGradientBoostingClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from features import FPS, HOME, AWAY, rolling_nanmean  # noqa: E402
import possession  # noqa: E402
import team_shape  # noqa: E402

SUBSAMPLE = 5        # training frames: every 5th (neighbouring frames are near-duplicates)


def _shift(x, k):
    """x delayed by k frames (k > 0: value from k frames ago), edges held."""
    if k == 0:
        return x
    out = np.empty_like(x)
    if k > 0:
        out[k:] = x[:-k]
        out[:k] = x[0]
    else:
        out[:k] = x[-k:]
        out[k:] = x[-1]
    return out


def _past(x, win):
    return _shift(rolling_nanmean(x, win), win // 2)


def _future(x, win):
    return _shift(rolling_nanmean(x, win), -(win // 2))


def _since(mask, cap):
    """Seconds since mask was last True (capped)."""
    n = len(mask)
    idx = np.where(mask, np.arange(n), -10 ** 9)
    last = np.maximum.accumulate(idx)
    return np.minimum((np.arange(n) - last) / FPS, cap)


def _until(mask, cap):
    return _since(mask[::-1], cap)[::-1]


def dead_features(f):
    ps = f["pspeed"]
    seen = f["seen"].astype(float)
    bs = f["bspeed"]
    oob = np.where(f["have"], f["oob"], np.nan)
    oobz = np.nan_to_num(oob, nan=-20.0)
    # last known ball position, held through gaps
    idx = np.maximum.accumulate(np.where(f["have"], np.arange(f["n"]), 0))
    bx, by = f["bx"][idx], f["by"][idx]
    L, W = 105.0, 68.0
    cols = [
        rolling_nanmean(ps, FPS), rolling_nanmean(ps, 3 * FPS), rolling_nanmean(ps, 6 * FPS),
        _past(ps, 3 * FPS), _future(ps, 3 * FPS), _future(ps, 3 * FPS) - _past(ps, 3 * FPS),
        uniform_filter1d(seen, FPS, mode="nearest"), uniform_filter1d(seen, 3 * FPS, mode="nearest"),
        _since(f["seen"], 20.0), _until(f["seen"], 20.0),
        rolling_nanmean(bs, FPS), maximum_filter1d(np.nan_to_num(bs), FPS),
        np.clip(oobz, -20, 10), maximum_filter1d(oobz, 2 * FPS, origin=FPS - 1),
        maximum_filter1d(oobz, 2 * FPS, origin=-(FPS - 1)),
        np.minimum(by, W - by), np.minimum(bx, L - bx), _since(f["have"], 20.0),
        rolling_nanmean(np.where(f["have"], f["n_within5"], np.nan), FPS),
        rolling_nanmean(f["npl"], FPS), rolling_nanmean(f["own_half"], FPS),
        rolling_nanmean((f["h_sx"] + f["a_sx"]) / 2, FPS), rolling_nanmean((f["h_sy"] + f["a_sy"]) / 2, FPS),
        rolling_nanmean(np.abs(f["h_cx"] - f["a_cx"]), FPS),
    ]
    return np.stack(cols, 1).astype(np.float32)


def team_features(f, ctrl=None):
    if ctrl is None:
        ctrl = possession.control_frames(f)
    big = 50.0
    dH = np.nan_to_num(f["dH"], nan=big)
    dA = np.nan_to_num(f["dA"], nan=big)
    ch = (ctrl == HOME).astype(float)
    ca = (ctrl == AWAY).astype(float)
    n = f["n"]
    any_c = ctrl >= 0
    li = np.maximum.accumulate(np.where(any_c, np.arange(n), -1))
    last_team = np.where(li >= 0, np.where(ctrl[np.maximum(li, 0)] == HOME, 1.0, -1.0), 0.0)
    ni = np.minimum.accumulate(np.where(any_c, np.arange(n), n)[::-1])[::-1]
    nxt_team = np.where(ni < n, np.where(ctrl[np.minimum(ni, n - 1)] == HOME, 1.0, -1.0), 0.0)
    cols = [
        np.minimum(dH, big), np.minimum(dA, big), dA - dH,
        _past(dA - dH, FPS), _future(dA - dH, FPS),
        minimum_filter1d(dH, 10, origin=4), minimum_filter1d(dA, 10, origin=4),
        minimum_filter1d(dH, 10, origin=-5), minimum_filter1d(dA, 10, origin=-5),
        _past(ch, FPS), _past(ca, FPS), _past(ch, 3 * FPS), _past(ca, 3 * FPS),
        _future(ch, FPS), _future(ca, FPS), _future(ch, 3 * FPS), _future(ca, 3 * FPS),
        last_team, _since(any_c, 20.0), nxt_team, _until(any_c, 20.0),
        rolling_nanmean(f["bspeed"], 10),
    ]
    X = np.concatenate([np.stack(cols, 1), team_shape.shape_features(f)], 1)
    return X.astype(np.float32)


# Hyperparameters of the two gradient-boosted tree models used in the submission.
# early_stopping='auto' (the scikit-learn default) holds out a random 10% of the training rows.
DEAD_PARAMS = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=100,
                   l2_regularization=1.0, early_stopping="auto")
TEAM_PARAMS = dict(DEAD_PARAMS)


def params_tag():
    """Short hash of the tree hyperparameters, part of every cache key built on the trees."""
    import hashlib
    import json
    blob = json.dumps([DEAD_PARAMS, TEAM_PARAMS], sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:8]


def _clf(params, seed):
    return HistGradientBoostingClassifier(random_state=seed, **params)


def match_matrices(name, seed):
    """Training rows of one degraded external match (cached): dead X, y and team X, y."""
    import augment
    path = os.path.join(HERE, "cache", "supx_{}_{}_{}.npz".format(name, seed, augment.VERSION))
    if os.path.exists(path):
        z = np.load(path)
        return z["Xd"], z["yd"], z["Xt"], z["yt"]
    import augment
    import external
    from features import build_from
    d = external.load(name)
    lab = external.controlled(d["labels"])
    dd = augment.degrade(d, seed)
    g = lab != "X"
    f = build_from(dd["players"], dd["ball"], dd["teams"], g, None, dd["pitch"])
    idx = np.where(g)[0][::SUBSAMPLE]
    Xd = dead_features(f)[idx]
    yd = (lab[idx] == "D").astype(np.int8)
    alive = idx[np.isin(lab[idx], ["H", "A"])]
    Xt = team_features(f)[alive]
    yt = (lab[alive] == "H").astype(np.int8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    from cacheio import save_npz
    save_npz(path, Xd=Xd, yd=yd, Xt=Xt, yt=yt)
    return Xd, yd, Xt, yt


TRAIN_SEEDS = (0, 1)   # two independently degraded copies of every training match


def training_set(names, seeds=TRAIN_SEEDS):
    Xd, yd, Xt, yt = [], [], [], []
    for name in names:
        for s in seeds:
            a, b, c, d = match_matrices(name, s)
            Xd.append(a); yd.append(b); Xt.append(c); yt.append(d)
    return Xd, yd, Xt, yt


def fit(names, seeds=TRAIN_SEEDS, seed=0, dead_params=None, team_params=None):
    Xd, yd, Xt, yt = training_set(names, seeds)
    dead = _clf(DEAD_PARAMS if dead_params is None else dead_params, seed).fit(np.concatenate(Xd), np.concatenate(yd))
    team = _clf(TEAM_PARAMS if team_params is None else team_params, seed).fit(np.concatenate(Xt), np.concatenate(yt))
    return dict(dead=dead, team=team)


def apply(models, f):
    """Per-frame P(dead) and P(home in possession) for one match."""
    p_dead = models["dead"].predict_proba(dead_features(f))[:, 1]
    p_home = models["team"].predict_proba(team_features(f))[:, 1]
    return p_dead, p_home


def models_for(exclude=None, seeds=TRAIN_SEEDS):
    """Models trained on every available external match except `exclude`, cached on disk."""
    import pickle
    import augment
    import external
    names = [n for n in external.available() if n != exclude]
    if len(names) < len(external.IDSSE_MATCHES) + len(external.METRICA_GAMES) - (exclude is not None):
        raise FileNotFoundError("external training data incomplete -- run solution/download_external.sh")
    key = "sup_{}_{}_{}_{}.pkl".format("all" if exclude is None else "no_" + exclude, "-".join(map(str, seeds)),
                                       augment.VERSION, params_tag())
    path = os.path.join(HERE, "cache", key)
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    m = fit(names, seeds)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    from cacheio import save_pickle
    save_pickle(path, m)
    return m
