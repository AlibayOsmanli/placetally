"""Score the stored answers: strict MRA with 95% intervals over streams, for every method and split."""

import argparse
import collections
import json
from pathlib import Path

from placetally.metrics import bootstrap_ci, mra

ROOT = Path(__file__).resolve().parent.parent


def read_jsonl(path):
    return [json.loads(line) for line in open(path)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, default=ROOT / "results/answers.jsonl")
    parser.add_argument("--truth", type=Path, default=ROOT / "results/truth.jsonl")
    parser.add_argument("--output", type=Path, help="also write the scores as JSON")
    args = parser.parse_args()

    truth = {(r["stream_id"], r["category"]): r for r in read_jsonl(args.truth)}
    groups = collections.defaultdict(list)
    for r in read_jsonl(args.answers):
        t = truth[(r["stream_id"], r["category"])]
        groups[(t["split"], r["method"])].append((r["stream_id"], r["category"], r["answer"], t["answer"]))

    scores = []
    for (split, method), rows in sorted(groups.items()):
        rows.sort()
        # An answer that could not be parsed is stored as null and scores as 0.
        values = [mra(0 if answer is None else answer, true) for _, _, answer, true in rows]
        mean, low, high = bootstrap_ci(values, [r[0] for r in rows])
        scores.append(
            {
                "split": split,
                "method": method,
                "questions": len(rows),
                "streams": len({r[0] for r in rows}),
                "mra": [100 * mean, 100 * low, 100 * high],
            }
        )
        print(f"{split:11s} {method:52s} {len(rows):4d}  {100 * mean:5.1f} [{100 * low:.1f}, {100 * high:.1f}]")
    if args.output:
        args.output.write_text(json.dumps(scores, indent=1) + "\n")


if __name__ == "__main__":
    main()
