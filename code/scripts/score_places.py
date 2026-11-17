"""Place F1, number of places and exact streams on ARKit test, from stored frame-pair counts."""

import argparse
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=ROOT / "results/place_pairs.jsonl")
    parser.add_argument("--output", type=Path, help="also write the scores as JSON")
    args = parser.parse_args()

    by_method = collections.defaultdict(list)
    for line in open(args.pairs):
        r = json.loads(line)
        by_method[r["method"]].append(r)

    scores = {}
    for method, rows in by_method.items():
        both = sum(r["pairs_in_both"] for r in rows)
        f1 = 2 * both / (sum(r["pairs_in_place"] for r in rows) + sum(r["pairs_in_room"] for r in rows))
        scores[method] = {
            "streams": len(rows),
            "f1": f1,
            "places": sum(r["places"] for r in rows),
            "rooms": sum(r["rooms"] for r in rows),
            "exact_streams": sum(r["places"] == r["rooms"] for r in rows),
        }
        s = scores[method]
        print(f"{method:36s} F1 {f1:.3f}  places {s['places']:5d}  exact {s['exact_streams']:3d} of {len(rows)}")
    if args.output:
        args.output.write_text(json.dumps(scores, indent=1) + "\n")


if __name__ == "__main__":
    main()
