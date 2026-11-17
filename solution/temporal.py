"""A temporal neural network over the frame features, trained on OTHER matches only.

A bidirectional GRU reads the same per-frame features as the gradient-boosting models
(supervised.dead_features + team_features) at 5 Hz and outputs P(home), P(away), P(dead) for every
step, with minutes of context in both directions. Training data are the degraded external matches
(augment.py), labels with the controlled-possession rule; nothing comes from the competition match.
The resulting evidence goes through the Viterbi decoders in decode.py.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

STEP = 5             # frames per network step (5 Hz)
WINDOW = 512         # steps per training window (~100 s)
HIDDEN = 64
EPOCHS = 12
BATCHES = 40         # per epoch
BATCH = 32
LR = 2e-3
SEED = 0

# Additional networks used with the tree and seed-0 GRU in the ownership ensemble.
ENSEMBLE = {"gru1": dict(arch="gru", seed=1), "gru2": dict(arch="gru", seed=2), "gru3": dict(arch="gru", seed=3),
            "gru4": dict(arch="gru", seed=4), "gru_long": dict(arch="gru", seed=0, epochs=30),
            "tcn": dict(arch="tcn", seed=0), "transformer": dict(arch="transformer", seed=0)}


def sequence(name, seed):
    """Features (T, F) at 5 Hz and targets (T,) in {0 H, 1 A, 2 D, -1 none} of a degraded match."""
    import augment
    path = os.path.join(HERE, "cache", "seq_{}_{}_{}.npz".format(name, seed, augment.VERSION))
    if os.path.exists(path):
        z = np.load(path)
        return z["X"], z["y"]
    import external
    import supervised
    from features import build_from
    d = external.load(name)
    lab = external.controlled(d["labels"])
    dd = augment.degrade(d, seed)
    f = build_from(dd["players"], dd["ball"], dd["teams"], lab != "X", None, dd["pitch"])
    X = np.concatenate([supervised.dead_features(f), supervised.team_features(f)], 1)[::STEP]
    y = np.select([lab == "H", lab == "A", lab == "D"], [0, 1, 2], -1).astype(np.int8)[::STEP]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    from cacheio import save_npz
    save_npz(path, X=X.astype(np.float32), y=y)
    return X, y


def features_of(f):
    import supervised
    return np.concatenate([supervised.dead_features(f), supervised.team_features(f)], 1)[::STEP]


class Normaliser:
    def __init__(self, Xs):
        X = np.concatenate(Xs)
        self.mean = np.nanmean(X, 0)
        self.std = np.nanstd(X, 0) + 1e-6
        self.nan_cols = np.flatnonzero(np.isnan(X).mean(0) > 0.01)

    def __call__(self, X):
        Z = np.clip((X - self.mean) / self.std, -5, 5)
        miss = np.isnan(X[:, self.nan_cols]).astype(np.float32)
        return np.concatenate([np.nan_to_num(Z), miss], 1).astype(np.float32)


def _net(n_in, arch="gru", hidden=HIDDEN):
    """gru: bidirectional GRU. tcn: dilated residual 1-D convolutions (receptive field ~200 s).
    transformer: self-attention over 512-step windows, with a convolution for local order."""
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Dropout(0.1))
            self.gru = nn.GRU(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
            self.out = nn.Linear(2 * hidden, 3)

        def forward(self, x):
            return self.out(self.gru(self.inp(x))[0])

    class Block(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.c1 = nn.Conv1d(hidden, hidden, 3, padding=d, dilation=d)
            self.c2 = nn.Conv1d(hidden, hidden, 3, padding=d, dilation=d)
            self.n1, self.n2 = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
            self.drop = nn.Dropout(0.1)

        def forward(self, h):                  # (B, T, C)
            y = self.drop(nn.functional.gelu(self.n1(self.c1(h.transpose(1, 2)).transpose(1, 2))))
            y = self.drop(nn.functional.gelu(self.n2(self.c2(y.transpose(1, 2)).transpose(1, 2))))
            return h + y

    class TCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Dropout(0.1))
            self.blocks = nn.Sequential(*[Block(2 ** i) for i in range(8)])
            self.out = nn.Linear(hidden, 3)

        def forward(self, x):
            return self.out(self.blocks(self.inp(x)))

    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Dropout(0.1))
            self.local = nn.Conv1d(hidden, hidden, 9, padding=4)
            layer = nn.TransformerEncoderLayer(hidden, 4, 4 * hidden, 0.1, activation="gelu", batch_first=True,
                                               norm_first=True)
            self.enc = nn.TransformerEncoder(layer, 3, enable_nested_tensor=False)
            self.out = nn.Linear(hidden, 3)
            pos = torch.arange(WINDOW)[:, None] / torch.pow(10000.0, torch.arange(0, hidden, 2) / hidden)
            pe = torch.zeros(WINDOW, hidden)
            pe[:, 0::2], pe[:, 1::2] = torch.sin(pos), torch.cos(pos)
            self.register_buffer("pe", 0.1 * pe)

        def forward(self, x):
            h = self.inp(x)
            h = h + self.local(h.transpose(1, 2)).transpose(1, 2) + self.pe[:h.shape[1]]
            return self.out(self.enc(h))

    return {"gru": Net, "tcn": TCN, "transformer": Transformer}[arch]()


def _partial_loss(logits, y):
    """Cross-entropy over H / A / D, where a target of 3 means 'in play, team not labelled':
    its loss is -log(P(H) + P(A))."""
    import torch
    logp = torch.log_softmax(logits, -1)
    full = (y >= 0) & (y <= 2)
    alive = y == 3
    terms = [-logp[full].gather(1, y[full][:, None])[:, 0], -torch.logsumexp(logp[alive][:, :2], -1)]
    return torch.cat(terms).mean()


def train(names, seeds=(0, 1), verbose=False, data=None, arch="gru", seed=SEED, epochs=EPOCHS, hidden=HIDDEN):
    """Fit a temporal model using named matches or precomputed training sequences."""
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    rng = np.random.default_rng(seed)
    data = [sequence(n, s) for n in names for s in seeds] if data is None else data
    partial = any((y == 3).any() for _, y in data)
    norm = Normaliser([X for X, _ in data])
    seqs = [(torch.from_numpy(norm(X)), torch.from_numpy(y.astype(np.int64))) for X, y in data]
    net = _net(seqs[0][0].shape[1], arch, hidden)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=epochs * BATCHES)
    lossf = _partial_loss if partial else torch.nn.CrossEntropyLoss(ignore_index=-1)
    for ep in range(epochs):
        net.train()
        tot = 0.0
        for _ in range(BATCHES):
            xb, yb = [], []
            for _ in range(BATCH):
                X, y = seqs[rng.integers(len(seqs))]
                s = int(rng.integers(0, len(y) - WINDOW))
                xb.append(X[s:s + WINDOW])
                yb.append(y[s:s + WINDOW])
            xb, yb = torch.stack(xb), torch.stack(yb)
            loss = lossf(net(xb).reshape(-1, 3), yb.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss)
        if verbose:
            print("  epoch {:2d} loss {:.4f}".format(ep, tot / BATCHES), flush=True)
    net.eval()
    return dict(net=net, norm=norm, arch=arch, hidden=hidden)


def _logits(net, Z, arch):
    """Network outputs (T, 3) over a whole half; the transformer reads overlapping 512-step windows
    blended with a Hann taper, as in training."""
    import torch
    with torch.no_grad():
        if arch != "transformer":
            return net(torch.from_numpy(Z)[None])[0].numpy()
        T = len(Z)
        if T <= WINDOW:
            return net(torch.from_numpy(Z)[None])[0].numpy()
        starts = list(range(0, T - WINDOW + 1, WINDOW // 2))
        if starts[-1] != T - WINDOW:
            starts.append(T - WINDOW)
        w = np.sin(np.pi * (np.arange(WINDOW) + 0.5) / WINDOW) ** 2 + 1e-3
        num, den = np.zeros((T, 3)), np.zeros(T)
        for i in range(0, len(starts), 64):
            ss = starts[i:i + 64]
            out = net(torch.from_numpy(np.stack([Z[s:s + WINDOW] for s in ss])))
            for s, o in zip(ss, out.numpy()):
                num[s:s + WINDOW] += w[:, None] * o
                den[s:s + WINDOW] += w
        return num / den[:, None]


def predict(model, X, n):
    """Per-frame P(dead), P(home | alive) at 25 Hz from 5 Hz network outputs."""
    logits = _logits(model["net"], model["norm"](X), model.get("arch", "gru"))
    t5 = np.arange(len(X)) * STEP
    t = np.arange(n)
    lg = np.stack([np.interp(t, t5, logits[:, k]) for k in range(3)], 1)
    p = np.exp(lg - lg.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    p_dead = p[:, 2]
    p_home = p[:, 0] / np.maximum(p[:, 0] + p[:, 1], 1e-9)
    return p_dead, p_home


def model_for(exclude=None, seeds=(0, 1)):
    """Network trained on every external match except `exclude`, cached on disk."""
    import pickle
    import augment
    import external
    import torch
    names = [n for n in external.available() if n != exclude]
    key = "gru_{}_{}_{}.pt".format("all" if exclude is None else "no_" + exclude, "-".join(map(str, seeds)),
                                   augment.VERSION)
    path = os.path.join(HERE, "cache", key)
    if os.path.exists(path):
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        net = _net(blob["n_in"])
        net.load_state_dict(blob["state"])
        net.eval()
        return dict(net=net, norm=blob["norm"])
    m = train(names, seeds)
    from cacheio import save_pickle
    save_pickle(path, dict(state=m["net"].state_dict(), norm=m["norm"], n_in=m["net"].inp[0].in_features))
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    return m
