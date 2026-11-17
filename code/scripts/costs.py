"""Cost numbers from the stored logs: construction, memory, input tokens and query time per method.

Covers the cost table on ARKit test (build speed, memory, input and query time of every row), the same numbers on
the long streams, and the input of the frame-budget sweep on the second training set. Memory is bytes kept after
construction per minute of video (MB = 10^6 bytes); input is the mean number of reader input tokens per question,
gate reads included. Construction and query times were measured on one A100-40GB GPU, query times with nothing
cached, on the same 50 test questions for every method.
"""

import argparse
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LENGTHS = {"arkit-test": "test", "arkit-long30": "30 min", "arkit-long60": "60 min"}
BUDGETS = ("16", "32", "64", "128")


def read_jsonl(path):
    return [json.loads(line) for line in open(path)]


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def length(stream_id):
    return LENGTHS[stream_id.rsplit("-", 1)[0]]


def show(x, fmt):
    width = int(fmt.split(".")[0].rstrip(","))
    return "--".rjust(width) if x is None else format(x, fmt)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--costs", type=Path, default=ROOT / "results/costs")
    parser.add_argument("--output", type=Path, help="also write the numbers as JSON")
    args = parser.parse_args()

    # Construction: decoding, features and places are shared by every K; writing the memory depends on K.
    construction = {}
    by_length = collections.defaultdict(list)
    for r in read_jsonl(args.costs / "construction.jsonl"):
        by_length[length(r["stream_id"])].append(r)
    for size, rows in by_length.items():
        minutes = sum(r["minutes"] for r in rows)
        for k in BUDGETS:
            seconds = sum(r["features_and_places_s"] + r["memory_write_s"][k] for r in rows)
            construction[f"PlaceTally, {k} frames per place / {size}"] = {
                "gpu_s_per_minute": seconds / minutes,
                "times_real_time": 60 * minutes / seconds,
                "streams": len(rows),
            }

    memory = collections.defaultdict(list)
    for r in read_jsonl(args.costs / "memory.jsonl"):
        memory[(r["method"], length(r["stream_id"]))].append(r)
    questions = collections.defaultdict(list)
    for r in read_jsonl(args.costs / "questions.jsonl"):
        questions[(r["method"], length(r["stream_id"]))].append(r)
    query = collections.defaultdict(list)
    for r in read_jsonl(args.costs / "query_times.jsonl"):
        query[r["method"]].append(r["gpu_s"])

    table = {}
    for method, size in sorted(set(questions) | set(memory)):
        kept = memory.get((method, size), [])
        rows = questions.get((method, size), [])
        table[f"{method} / {size}"] = {
            "memory_mb_per_minute": (
                sum(r["total_bytes"] for r in kept) / sum(r["minutes"] for r in kept) / 1e6 if kept else None
            ),
            "memory_streams": len(kept),
            "input_k_tokens": (
                None if mean(r["input_tokens"] for r in rows) is None else mean(r["input_tokens"] for r in rows) / 1000
            ),
            # Only where every question has a stored time (batched reads; on test most were not timed).
            "reader_gpu_s_per_question": (
                mean(r["reader_gpu_s"] for r in rows) if all(r["reader_gpu_s"] is not None for r in rows) else None
            ),
            "questions": len(rows),
        }
    query_table = {method: {"gpu_s": mean(v), "questions": len(v)} for method, v in query.items()}

    budgets = collections.defaultdict(list)
    for r in read_jsonl(args.costs / "budgets.jsonl"):
        budgets[r["method"]].append(r)
    budget_table = {
        method: {
            "questions": len(rows),
            "input_k_tokens": mean(r["input_tokens"] for r in rows) / 1000,
            "gpu_s_per_question": mean(r["batch_share_gpu_s"] for r in rows),
        }
        for method, rows in budgets.items()
    }
    others = json.loads((args.costs / "other_methods.json").read_text())

    # The cost table on ARKit test: build (x real time), memory (MB/min), input (k tokens), query (GPU-s).
    def own(method, k=None):
        test = table[f"{method} / test"]
        build = construction[f"{method} / test"]["times_real_time"] if k else None
        return [build, test["memory_mb_per_minute"], test["input_k_tokens"], query_table[method]["gpu_s"]]

    va, cv, dm = others["VideoAgent"], others["CountVid"], others["DirectMe"]
    cambrian = others["Cambrian-S-7B, surprise memory"]
    cost_table = {
        **{f"PlaceTally, K={k}": own(f"PlaceTally, {k} frames per place", k) for k in ("16", "64", "128")},
        "Whole-video reading": own("whole-video reading"),
        "VideoAgent": [
            60 / va["construction_gpu_s_per_minute"],
            va["input_mb_per_minute_as_jpeg"] + va["state_mb_per_minute"],
            "n/c",
            query_table["VideoAgent"]["gpu_s"],
        ],
        "CountVid": [None, cv["input_mb_per_minute_as_jpeg"], None, cv["query_gpu_s"]],
        "DirectMe": [60 / dm["construction_gpu_s_per_minute"], dm["memory_mb_per_minute_as_jpeg"], None, None],
        "Cambrian-S whole video": [None, "n/m", "n/c", query_table["Cambrian-S-7B, whole-video reading"]["gpu_s"]],
        "Cambrian-S memory": [
            None,
            cambrian["state_mb_per_minute"],
            "n/c",
            query_table["Cambrian-S-7B, surprise memory"]["gpu_s"],
        ],
    }

    def cell(x, fmt):
        return (x if isinstance(x, str) else show(x, fmt)).rjust(int(fmt.split(".")[0].rstrip(",")))

    print("Cost table, ARKit test (11-minute videos, one A100-40GB GPU; --: none, n/c: counted differently, n/m: not measured)")
    print(f"{'method':24s} {'build x real time':>17s} {'memory MB/min':>14s} {'input k tokens':>15s} {'query GPU-s':>12s}")
    for name, (build, mb, tokens, q) in cost_table.items():
        print(f"{name:24s} {cell(build, '17.2f')} {cell(mb, '14.2f')} {cell(tokens, '15.1f')} {cell(q, '12,.1f')}")
    print(
        "  JPEG estimates for VideoAgent, CountVid and DirectMe; as run "
        f"{va['input_mb_per_minute_as_run'] + va['state_mb_per_minute']:.0f}, {cv['input_mb_per_minute_as_run']:.0f} and "
        f"{dm['memory_mb_per_minute_as_run']:.0f} MB/min. CountVid query: GPU timers on {cv['query_gpu_questions']} questions. "
        f"Cambrian-S memory: GPU state while answering. DirectMe answers on a CPU in {dm['answer_cpu_s'] * 1000:.0f} ms."
    )

    print("PlaceTally construction, GPU-s per minute of video (x real time):")
    for size in ("test", "30 min", "60 min"):
        cells = [construction[f"PlaceTally, {k} frames per place / {size}"] for k in BUDGETS]
        print(
            f"  {size:7s} {cells[0]['streams']} streams  "
            + "  ".join(f"K={k} {c['gpu_s_per_minute']:.2f} ({c['times_real_time']:.1f}x)" for k, c in zip(BUDGETS, cells))
        )

    print(f"{'method / length':52s} {'MB/min':>7s} {'k tokens':>9s} {'GPU-s/q':>8s}")
    for key, r in table.items():
        print(
            f"{key:52s} {show(r['memory_mb_per_minute'], '7.2f')} {show(r['input_k_tokens'], '9.1f')} "
            f"{show(r['reader_gpu_s_per_question'], '8.2f')}"
        )
    print("  GPU-s/q: reader time per question, batched requests sharing a batch's time equally")
    print("Query time with nothing cached, GPU-s per question (the same 50 test questions):")
    for method, r in query_table.items():
        print(f"  {method:42s} {r['gpu_s']:6.2f}  ({r['questions']} questions)")
    print("Frame and token budgets, second training set:")
    for method, r in sorted(budget_table.items()):
        print(f"  {method:42s} {r['input_k_tokens']:7.1f} k tokens  {r['gpu_s_per_question']:5.2f} GPU-s")

    if args.output:
        out = {
            "construction": construction,
            "methods": table,
            "query": query_table,
            "cost_table": cost_table,
            "budgets": budget_table,
        }
        args.output.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
