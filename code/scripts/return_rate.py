"""The return-rate study: MRA of each method as visits are removed from 60 test streams (291 questions)."""

import argparse
import collections
import json
from pathlib import Path

from placetally.metrics import bootstrap_ci, mra

ROOT = Path(__file__).resolve().parent.parent
ORDER = ["first visits only", "half the returns", "all visits", "returning rooms only"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, default=ROOT / "results/return_rate.jsonl")
    parser.add_argument("--output", type=Path, help="also write the scores as JSON")
    args = parser.parse_args()

    by_setting = collections.defaultdict(list)
    for line in open(args.answers):
        r = json.loads(line)
        by_setting[r["setting"]].append(r)

    scores = {}
    for setting in ORDER:
        rows = by_setting[setting]
        share = sum(r["return_share"] for r in rows) / len(rows)
        print(f"{setting} (return rate {share:.2f}, {len(rows)} questions)")
        scores[setting] = {"return_rate": share, "questions": len(rows), "mra": {}}
        methods = sorted({m for r in rows for m in r["answers"]})
        for method in methods:
            if not all(method in r["answers"] for r in rows):
                continue
            values = [mra(r["answers"][method] or 0, r["truth"]) for r in rows]
            mean, low, high = bootstrap_ci(values, [r["stream_id"] for r in rows])
            scores[setting]["mra"][method] = [100 * mean, 100 * low, 100 * high]
            print(f"  {method:36s} {100 * mean:5.1f} [{100 * low:.1f}, {100 * high:.1f}]")
    if args.output:
        args.output.write_text(json.dumps(scores, indent=1) + "\n")


if __name__ == "__main__":
    main()
