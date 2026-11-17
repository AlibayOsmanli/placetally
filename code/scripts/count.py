"""Count objects in one stream with PlaceTally (needs a GPU and vLLM).

Finds the places from the features written by extract_features.py, then reads every place once per
category: a count read and a presence gate. The answer is the sum of the gated counts.
"""

import argparse
import functools
import io
import json
from pathlib import Path

import numpy as np

from placetally.places import find_places
from placetally.reader import Reader, gated_count, letterbox, make_video, sample_frames
from placetally.weights import load_weights

ROOT = Path(__file__).resolve().parent.parent


@functools.lru_cache(maxsize=4)
def archive(path):
    with np.load(path, allow_pickle=False) as z:
        return z["jpeg_blob"], z["jpeg_offsets"]


def load_frame(frame):
    blob, offsets = archive(frame["archive"])
    i = frame["archive_index"]
    return letterbox(io.BytesIO(blob[offsets[i] : offsets[i + 1]].tobytes()), frame["orientation"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", type=Path, required=True, help="folder written by extract_features.py")
    parser.add_argument("--categories", nargs="+", required=True, help="e.g. chair tv_monitor")
    parser.add_argument("--reader", required=True, help="local Qwen3.5-9B checkpoint (see settings.py)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.features / "features.npz", allow_pickle=False) as z:
        codes = z["codes"].astype(np.float32)
        thumbnail = z["thumbnail"].astype(np.float32)
        histogram = z["histogram"].astype(np.float32)
        detector_codes = z["detector_codes"].astype(np.float32)
        positions = z["positions"].astype(int)
    frames = json.loads((args.features / "frames.json").read_text())
    assert len(frames) == len(codes)
    assert np.allclose(np.linalg.norm(codes, axis=1), 1, atol=1e-4), "codes must be unit length"

    places = find_places(codes, thumbnail, histogram, detector_codes, positions, load_weights(ROOT / "weights"))
    reader = Reader(args.reader)

    totals = {c: 0 for c in args.categories}
    reads = []
    for p, members in enumerate(places):
        chosen = [frames[members[int(i)]] for i in sample_frames(len(members))]
        video = make_video(np.stack([load_frame(f) for f in chosen]), [float(f["time"]) for f in chosen])
        for category in args.categories:
            count, p_yes, tokens = reader.read(video, category, f"place-{p}")
            counted = gated_count(count, p_yes)
            totals[category] += counted
            reads.append(
                {
                    "place": p,
                    "category": category,
                    "count": count,
                    "p_yes": p_yes,
                    "counted": counted,
                    "input_tokens": tokens,
                }
            )

    result = {"places": places, "answers": totals, "reads": reads}
    args.output.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(totals))


if __name__ == "__main__":
    main()
