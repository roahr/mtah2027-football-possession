"""Parse the raw ATD text into a compact .npz cache (players, ball, team of each slot)."""
import os, sys, xml.etree.ElementTree as ET
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import require, source_frames

CACHE = os.path.join(ROOT, "solution", "cache")


def slot_teams(meta_path):
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    return np.array([0 if p.get("teamId") == team_ids[0] else 1 for p in root.iter("Player")], np.int8)


def parse(source="tactical"):
    os.makedirs(CACHE, exist_ok=True)
    out = os.path.join(CACHE, "atd_{}.npz".format(source))
    if os.path.exists(out):
        return dict(np.load(out))
    meta, raw = require(["atd_meta", "atd_raw"], source=source)
    teams = slot_teams(meta)
    n = source_frames(source)
    P = np.full((n, len(teams), 3), np.nan, np.float32)
    B = np.full((n, 3), np.nan, np.float32)
    with open(raw) as fh:
        for line in fh:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            i = int(parts[0]) - 1
            B[i] = [float(v) for v in parts[2].split(",")]
            for s, g in enumerate(parts[1].split(";")):
                if g[0] != "N":
                    P[i, s] = [float(v) for v in g.split(",")]
    np.savez_compressed(out, players=P, ball=B, teams=teams)
    return dict(players=P, ball=B, teams=teams)


if __name__ == "__main__":
    for s in sys.argv[1:] or ["tactical"]:
        d = parse(s)
        print(s, d["players"].shape, "ball seen", np.mean(~np.isnan(d["ball"][:, 0])))
