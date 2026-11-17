"""Fuse externally trained model evidence and decode possession in two stages.

The status branch combines a boosted tree and the seed-0 bidirectional GRU.
The ownership branch combines a boosted tree with a multi-seed GRU group,
a longer-trained GRU, a dilated convolution network and a transformer.
Two Viterbi passes place live/dead boundaries and assign a team within each
live segment. The switch costs were chosen on external matches.
"""
import numpy as np

from alive_dead import segments_from_mask
import boundaries
import possession
import supervised

# Chosen using leave-one-match-out evaluation on the external matches.
PARAMS = dict(dead_bias=0.0, dead_switch=30.0, refine=False, kickoffs=False,
              w_ctrl=0.0, w_sup=1.0, switch=6.0)


def two_state_viterbi(e, switch):
    """Boolean path maximising sum(+-e/2) - switch * changes (True where e favours the state)."""
    T = len(e)
    s1, s0 = e[0] / 2, -e[0] / 2
    back = np.zeros((T, 2), np.bool_)
    for t in range(1, T):
        st1, sw1 = s1, s0 - switch
        st0, sw0 = s0, s1 - switch
        back[t, 1] = sw1 > st1
        back[t, 0] = sw0 > st0
        s1 = max(st1, sw1) + e[t] / 2
        s0 = max(st0, sw0) - e[t] / 2
    state = 1 if s1 >= s0 else 0
    out = np.empty(T, np.bool_)
    for t in range(T - 1, -1, -1):
        out[t] = state == 1
        if back[t, state]:
            state = 1 - state
    return out


def dead_mask(f, p_dead, dead_bias=0.0, dead_switch=30.0, refine=True, kickoffs=True, **_):
    p = np.clip(p_dead, 1e-3, 1 - 1e-3)
    llr = np.log(p / (1 - p)) - dead_bias
    dead = np.zeros(f["n"], bool)
    for a, b in segments_from_mask(f["graded"]):
        dead[a:b] = two_state_viterbi(llr[a:b], dead_switch)
    if refine:
        dead, _ = boundaries.refine(f, dead)
    if kickoffs:
        dead = boundaries.apply_kickoffs(f, dead)
    return dead & f["graded"]


def team_labels(f, dead, p_home, ctrl=None, events=None, w_ctrl=1.0, w_sup=1.0, switch=6.0, **_):
    if ctrl is None:
        ctrl = possession.control_frames(f)
    if events is None:
        events = possession.confirmed_events(possession.control_events(ctrl))
    alive = f["graded"] & ~dead
    team = possession.viterbi_team(f, segments_from_mask(alive), ctrl, events, None, w_ctrl=w_ctrl,
                                   w_shape=0.0, switch=switch, w_event=0.0, p_sup=p_home, w_sup=w_sup)
    labels = np.full(f["n"], "X", dtype="<U1")
    labels[dead] = "D"
    labels[alive & (team == possession.HOME)] = "H"
    labels[alive & (team == possession.AWAY)] = "A"
    return labels


def _logit(p):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def probabilities(f, models=None, net=None, use_net=True, ensemble=None, use_ensemble=True):
    """Return per-frame dead-ball and live home-possession probabilities."""
    import models_store
    models = models_store.boosting() if models is None else models
    p_dead, p_home = supervised.apply(models, f)
    if use_net:
        import temporal
        net = models_store.network() if net is None else net
        X = temporal.features_of(f)
        q_dead, q_home = temporal.predict(net, X, f["n"])
        p_dead = 1 / (1 + np.exp(-(_logit(p_dead) + _logit(q_dead)) / 2))
        if not use_ensemble:
            p_home = 1 / (1 + np.exp(-(_logit(p_home) + _logit(q_home)) / 2))
        else:
            ens = models_store.ensemble() if ensemble is None else ensemble
            q = {k: _logit(temporal.predict(m, X, f["n"])[1]) for k, m in ens.items()}
            gru5 = np.mean([_logit(q_home)] + [q[k] for k in ("gru1", "gru2", "gru3", "gru4")], 0)
            team = np.mean([_logit(p_home), gru5, q["gru_long"], q["tcn"], q["transformer"]], 0)
            p_home = 1 / (1 + np.exp(-team))
    return p_dead, p_home


def predict(f, models=None, net=None, use_net=True, use_ensemble=True, **overrides):
    P = dict(PARAMS, **overrides)
    p_dead, p_home = probabilities(f, models, net, use_net, use_ensemble=use_ensemble)
    dead = dead_mask(f, p_dead, **P)
    return team_labels(f, dead, p_home, **P), dict(p_dead=p_dead, p_home=p_home, dead=dead)
