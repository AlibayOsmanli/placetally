"""Rebuild the place partitions from the stored fragment statistics and compare them with the paper's.

Each file in fragments/ holds, for one stream, the six cut statistics of every cut, the raw SigLIP 2 chamfer
similarity between fragments, and the place label the paper used for each fragment. This replays the
continuation model and the best-link rule; cutting the stream needs the frames (see extract_features.py).
"""

import argparse
from pathlib import Path

import numpy as np

from placetally.places import continuation_probabilities, link_fragments
from placetally.weights import load_weights

ROOT = Path(__file__).resolve().parent.parent


def replay(path, continuation):
    z = np.load(path)
    stats = [dict(zip(continuation["features"], map(float, row))) for row in z["cut_stats"]]
    p_continue = continuation_probabilities(continuation, stats)
    labels = link_fragments(p_continue, z["raw_similarity"].astype(np.float64), continuation["level"])
    return np.unique(labels, return_inverse=True)[1], z["place_of_fragment"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fragments", type=Path, default=ROOT / "fragments")
    args = parser.parse_args()

    continuation = load_weights(ROOT / "weights").continuation
    failed = []
    for folder in sorted(p for p in args.fragments.iterdir() if p.is_dir()):
        files = sorted(folder.glob("*.npz"))
        different = []
        for path in files:
            rebuilt, stored = replay(path, continuation)
            if not np.array_equal(rebuilt, stored):
                different.append(path.stem)
        print(f"{folder.name}: {len(files) - len(different)} of {len(files)} streams identical")
        failed += different
    if failed:
        raise SystemExit("different: " + ", ".join(failed))


if __name__ == "__main__":
    main()
