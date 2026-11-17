"""Load the packaged models fitted on the ten external training matches.

  boosting_all.pkl   gradient boosting for dead-ball and ownership evidence
  gru_all.pkl        seed-0 bidirectional GRU
  net_*.pkl          seven additional networks for the ownership ensemble
"""
import os
import pickle

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, "models")
BOOSTING = os.path.join(DIR, "boosting_all.pkl")
NETWORK = os.path.join(DIR, "gru_all.pkl")


def _retrain():
    return os.environ.get("MTAH_RETRAIN") == "1"


def boosting():
    if os.path.exists(BOOSTING) and not _retrain():
        with open(BOOSTING, "rb") as fh:
            return pickle.load(fh)
    import supervised
    m = supervised.models_for()
    os.makedirs(DIR, exist_ok=True)
    from cacheio import save_pickle
    save_pickle(BOOSTING, m)
    return m


def network():
    import temporal
    if os.path.exists(NETWORK) and not _retrain():
        with open(NETWORK, "rb") as fh:
            blob = pickle.load(fh)
        net = temporal._net(blob["n_in"])
        net.load_state_dict(blob["state"])
        net.eval()
        return dict(net=net, norm=blob["norm"])
    m = temporal.model_for()
    os.makedirs(DIR, exist_ok=True)
    from cacheio import save_pickle
    save_pickle(NETWORK, dict(state=m["net"].state_dict(), norm=m["norm"], n_in=m["net"].inp[0].in_features))
    return m


def ensemble_member(name):
    """One network of the ownership ensemble, trained on the ten external
    matches, two degraded copies of each."""
    import temporal
    path = os.path.join(DIR, "net_{}.pkl".format(name))
    if os.path.exists(path) and not _retrain():
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        net = temporal._net(blob["n_in"], blob["arch"], blob["hidden"])
        net.load_state_dict(blob["state"])
        net.eval()
        return dict(net=net, norm=blob["norm"], arch=blob["arch"])
    import external
    import supervised
    names = external.available()
    if len(names) < len(external.IDSSE_MATCHES) + len(external.METRICA_GAMES):
        raise FileNotFoundError("external training data incomplete -- run solution/download_external.sh")
    m = temporal.train(None, data=[temporal.sequence(n, s) for n in names for s in supervised.TRAIN_SEEDS],
                       **temporal.ENSEMBLE[name])
    os.makedirs(DIR, exist_ok=True)
    from cacheio import save_pickle
    save_pickle(path, dict(state=m["net"].state_dict(), norm=m["norm"], n_in=m["net"].inp[0].in_features,
                           arch=m["arch"], hidden=m["hidden"]))
    return m


def ensemble():
    """Load the seven additional networks in the ownership ensemble."""
    import temporal
    return {name: ensemble_member(name) for name in temporal.ENSEMBLE}
