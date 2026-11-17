"""Compute the features of one benchmark stream from rebuilt recording archives (needs a GPU).

The benchmark's build scripts write one archive per recording: <recording>.npz with the 1-fps frames the
reader sees, and, with --five-fps, <recording>_five_fps.npz with the frames the cut detector sees. This
writes features.npz and frames.json to the output folder; count.py reads both.
"""

import argparse
import io
import json
from pathlib import Path

import numpy as np

from placetally.features import FrameEncoder, frame_features, unit


def encode_recording(path, orientation, encoder):
    with np.load(path, allow_pickle=False) as z:
        blob, offsets = z["jpeg_blob"], z["jpeg_offsets"]
        n = len(offsets) - 1
        feats = [frame_features(io.BytesIO(blob[offsets[i] : offsets[i + 1]].tobytes()), orientation) for i in range(n)]
        times = z["source_times_s"].copy()
        indices = z["recording_frame_indices"].copy() if "recording_frame_indices" in z else np.arange(n)
    # Thumbnails and histograms were stored as float16 for the paper's runs; keep that rounding.
    return {
        "code": unit(encoder.encode(np.stack([f[0] for f in feats]))),
        "thumbnail": np.asarray([f[1] for f in feats], np.float16).astype(np.float32),
        "histogram": np.asarray([f[2] for f in feats], np.float16).astype(np.float32),
        "time": times,
        "index": indices,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--streams", type=Path, required=True, help="a benchmark file, e.g. arkit_test.json")
    parser.add_argument("--stream", required=True, help="stream ID, e.g. arkit-test-0001")
    parser.add_argument("--recordings", type=Path, required=True, help="folder of rebuilt recording archives")
    parser.add_argument("--encoder", required=True, help="local SigLIP 2 checkpoint (see settings.py)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True, help="output folder")
    args = parser.parse_args()

    stream = next(s for s in json.loads(args.streams.read_text())["streams"] if s["stream_id"] == args.stream)
    encoder = FrameEncoder(args.encoder, device=args.device)

    encoded = {}
    codes, positions, frames = [], [], []
    five_fps = {"code": [], "thumbnail": [], "histogram": []}
    offset = 0
    for visit in stream["visits"]:
        recording, orientation = visit["recording_id"], visit["orientation"]
        one_path = args.recordings / f"{recording}.npz"
        if (recording, orientation) not in encoded:
            encoded[recording, orientation] = (
                encode_recording(one_path, orientation, encoder),
                encode_recording(args.recordings / f"{recording}_five_fps.npz", orientation, encoder),
            )
        one, five = encoded[recording, orientation]
        position_of = {int(v): i for i, v in enumerate(one["index"])}

        # The detector sees the whole recording at 5 fps, in the visit's direction.
        order = np.arange(len(five["time"]))
        if visit["reversed"]:
            order = order[::-1]
        for key in five_fps:
            five_fps[key].append(five[key][order])

        for r in visit["frame_ranges"]:
            for j in range(r["count"]):
                i = position_of[r["first"] + j * r["step"]]
                codes.append(one["code"][i])
                positions.append(offset + int(np.argmin(np.abs(five["time"][order] - one["time"][i]))))
                frames.append(
                    {
                        "archive": str(one_path.resolve()),
                        "archive_index": i,
                        "orientation": orientation,
                        "time": r["time_first"] + j * r["time_step"],
                    }
                )
        offset += len(order)

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "features.npz",
        # The paper's runs normalised the gathered codes once more; it changes the last float32 bit.
        codes=unit(np.asarray(codes, np.float32)),
        detector_codes=np.concatenate(five_fps["code"]),
        thumbnail=np.concatenate(five_fps["thumbnail"]),
        histogram=np.concatenate(five_fps["histogram"]),
        positions=np.asarray(positions, int),
    )
    (args.output / "frames.json").write_text(json.dumps(frames) + "\n")


if __name__ == "__main__":
    main()
