"""Generate the submission CSV from the tactical ATD and packaged checkpoints."""

import argparse
import os
from pathlib import Path
import sys
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from decode import predict
from features import build
from models_store import BOOSTING, DIR, NETWORK
from temporal import ENSEMBLE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="submission", help="Output CSV filename stem")
    parser.add_argument("--out", type=Path, default=ROOT / "evaluation" / "results")
    args = parser.parse_args()

    if os.environ.get("MTAH_RETRAIN") == "1":
        parser.error("Unset MTAH_RETRAIN to run inference with the packaged checkpoints.")
    checkpoints = [Path(BOOSTING), Path(NETWORK)]
    checkpoints.extend(Path(DIR) / f"net_{name}.pkl" for name in ENSEMBLE)
    missing = [str(path.relative_to(ROOT)) for path in checkpoints if not path.is_file()]
    if missing:
        parser.error(f"Missing packaged checkpoints: {', '.join(missing)}")

    # Empty reductions are expected where the source has no observations.
    with warnings.catch_warnings():
        for message in ("Mean of empty slice", "All-NaN slice encountered", "Degrees of freedom <= 0 for slice"):
            warnings.filterwarnings("ignore", message=message, category=RuntimeWarning)
        labels, _ = predict(build())
    output = args.out / f"{args.name}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines(f"{frame},{label}\n" for frame, label in enumerate(labels))
    print(f"Wrote {len(labels):,} frames to {output}")


if __name__ == "__main__":
    main()
