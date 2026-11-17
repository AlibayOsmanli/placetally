"""Rebuild the frames and answers of RevisitCount streams from your own copy of a source dataset.

Writes one archive per recording (JPEG frames and the visible object boxes of each frame) to
<output>/recordings, and recomputes the answers. ARKit answers are compared with the released ones.
Runs on CPU.
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import os
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import arkit, categories, mesh, mobile, plyio, surface  # noqa: E402
from lib.visibility import scan_visits  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def jpg(arr):
    f = io.BytesIO()
    Image.fromarray(np.asarray(arr, np.uint8)).save(f, format="JPEG", quality=90)
    return f.getvalue()


def ranges(vis):
    return [r["first"] + j * r["step"] for r in vis["frame_ranges"] for j in range(r["count"])]


def pack_images(blobs):
    offsets = np.cumsum([0] + list(map(len, blobs)), dtype=np.int64)
    return np.frombuffer(b"".join(blobs), np.uint8), offsets


def truth_arrays(labels, observations):
    ptr = [0]
    box = []
    uid = []
    for obs in observations:
        for obj, b in obs:
            uid.append(str(obj))
            box.append(b)
        ptr.append(len(uid))
    return {
        "uids_all": np.array(list(map(str, labels))),
        "labels_all": np.array(list(labels.values())),
        "p0_gt_ptr": np.asarray(ptr, np.int64),
        "p0_gt_box": np.asarray(box, np.float32).reshape(-1, 4),
        "p0_gt_uid": np.asarray(uid, dtype=str),
    }


def save_record(path, indices, blobs, labels, observations, orientation, **extra):
    data, off = pack_images(blobs)
    arrays = truth_arrays(labels, observations)
    np.savez_compressed(
        path,
        recording_frame_indices=np.asarray(indices, np.int64),
        jpeg_blob=data,
        jpeg_offsets=off,
        orientation=np.asarray(orientation),
        **arrays,
        **extra,
    )
    return arrays


def build_arkit(root, scan, indices, out):
    split, vid = scan.split("_", 1)
    arkit.ROOT = str(root / "3dod")
    arkit.RAW = str(root / "raw")
    surface.ROOT = arkit.ROOT
    surface.RAW = arkit.RAW
    sc = arkit.Scan(vid, split, hires=True)
    frames = [f for f in (sc.frame(i) for i in sc.sample(1.0, 0.0)) if f is not None]
    hs = surface.HiresScan(vid, split)
    ids = hs.sample(1.0, 0.0)
    assert len(ids) == len(frames), (scan, "source sampling mismatch")
    assert np.allclose(
        np.asarray(list(map(float, ids))) - float(ids[0]),
        np.asarray([f.t - frames[0].t for f in frames]),
        atol=1e-6,
        rtol=0,
    )
    labels = {str(b["uid"]): b["label"] for b in hs.boxes}
    obs = []
    blobs = []
    for k in indices:
        pose, K, depth = hs.geometry(ids[k])
        w2c = np.linalg.inv(pose)
        here = []
        for u, faces in zip(labels, hs.faces):
            b, _, _ = surface.visible_box(faces, pose, w2c, K, depth)
            if b is not None:
                here.append((u, b))
        obs.append(here)
        blobs.append(jpg(frames[k].image))
    save_record(
        out,
        indices,
        blobs,
        labels,
        obs,
        "from release visit",
        source_times_s=np.asarray([frames[k].t - frames[0].t for k in indices]),
    )
    return labels


def build_rescan(root, scan, indices, out):
    mesh.B = root
    folder = root / "scans" / scan
    with zipfile.ZipFile(folder / "sequence.zip") as z:
        info = z.read("_info.txt").decode()
        names = sorted(n for n in z.namelist() if n.endswith(".color.jpg"))
        native = [int(re.search(r"frame-(\d+)", n)[1]) for n in names]
        assert native == list(range(len(native)))
        poses = np.stack(
            [np.asarray(z.read(f"frame-{k:06d}.pose.txt").decode().split(), float).reshape(4, 4) for k in native]
        )
        K = np.asarray(re.search(r"m_calibrationColorIntrinsic = (.*)", info)[1].split(), float).reshape(4, 4)[:3, :3]
        W = int(re.search(r"m_colorWidth = (\d+)", info)[1])
        H = int(re.search(r"m_colorHeight = (\d+)", info)[1])
        up = poses[:, 2, :3]
        m = up.mean(0)
        axis = int(np.argmax(np.abs(m[:2])))
        sky = {(0, -1.0): "Left", (0, 1.0): "Right", (1, -1.0): "Up", (1, 1.0): "Down"}[(axis, float(np.sign(m[axis])))]
        V, O, F = mesh.load_ply(scan)
        labels = {str(k): v for k, v in mesh.semseg(scan).items()}
        obs = []
        blobs = []
        for k in indices:
            obs.append([(str(u), b) for u, b, f in mesh.frame_truth(V, O, F, poses[k], K, W, H) if str(u) in labels])
            blobs.append(z.read(names[k]))
    save_record(out, indices, blobs, labels, obs, sky)
    return labels


def build_multiscan(root, scan, indices, out):
    import cv2

    folder = root / scan
    pre = folder / scan
    L = [json.loads(l) for l in pre.with_suffix(".jsonl").read_text().splitlines() if l]
    annotation = read(pre.with_suffix(".annotations.json"))
    alignment = read(pre.with_suffix(".align.json"))
    V, F, objects, _ = plyio.read_ply(pre.with_suffix(".ply").read_bytes())
    V2, O2, F2 = mobile.vertex_split(V, F, objects)
    ts = np.asarray([r["timestamp"] for r in L], np.int64)
    sample = mobile.frame_times(ts, 1)
    native = sample[indices]
    poses = np.asarray([r["transform"] for r in L], float)
    R, t = mobile.poses_ply(poses, alignment["coordinate_transform"])
    K = np.asarray(L[len(L) // 2]["intrinsics"], float).reshape(3, 3).T
    K[:2] *= 640 / 1920
    world = np.linalg.inv(np.asarray(alignment["coordinate_transform"], float).reshape(4, 4).T)[:3, :3]
    sky, _ = mobile.sky_rule(np.einsum("ij,njk->nik", np.linalg.inv(world), R))
    labels = {str(o["objectId"]): o["label"] for o in annotation["objects"]}
    obs = []
    for k in native:
        pose = np.eye(4)
        pose[:3, :3] = R[k]
        pose[:3, 3] = t[k]
        obs.append([(str(u), b) for u, b, f in mesh.frame_truth(V2, O2, F2, pose, K, 640, 480) if str(u) in labels])
    cap = cv2.VideoCapture(str(pre.with_suffix(".mp4")))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == len(L)
    wanted = {int(v): j for j, v in enumerate(native)}
    blobs = [None] * len(native)
    for k in range(int(native.max()) + 1):
        if not cap.grab():
            raise RuntimeError((scan, "video stopped", k))
        if k in wanted:
            ok, image = cap.retrieve()
            assert ok
            image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)[:, :, ::-1]
            blobs[wanted[k]] = jpg(np.ascontiguousarray(image))
    cap.release()
    assert all(b is not None for b in blobs)
    save_record(
        out, indices, blobs, labels, obs, sky, native_video_indices=native, source_times_s=(ts[native] - ts[0]) / 1e9
    )
    return labels


def visit_objects(cache, visit):
    with np.load(cache / (visit["recording_id"] + ".npz"), allow_pickle=False) as z:
        positions = {int(v): i for i, v in enumerate(z["recording_frame_indices"])}
        ptr = z["p0_gt_ptr"]
        boxes = []
        uids = []
        offsets = [0]
        for index in ranges(visit):
            i = positions[index]
            boxes.extend(z["p0_gt_box"][ptr[i] : ptr[i + 1]])
            uids.extend(z["p0_gt_uid"][ptr[i] : ptr[i + 1]])
            offsets.append(len(uids))
        args = {
            "uids_all": z["uids_all"],
            "labels_all": z["labels_all"],
            "p0_gt_ptr": np.asarray(offsets),
            "p0_gt_box": np.asarray(boxes, np.float32).reshape(-1, 4),
            "p0_gt_uid": np.asarray(uids, dtype=str),
        }
        return {u: v[0] for u, v in scan_visits(args, 0).items()}


def rescan_identity(root, visits, annotations):
    groups = read(root / "3RScan.json")
    room = {}
    for g in groups:
        reference = g["reference"]
        room[reference] = reference
        for scan in g["scans"]:
            room[scan["reference"]] = reference
    by = collections.defaultdict(list)
    for v in visits:
        scan = v["recording_id"]
        by[room[scan]].append(scan)
    aliases = {}
    excluded = set()
    for recording_list in by.values():
        names = list(dict.fromkeys(recording_list))
        if len(names) < 2:
            continue
        a, b = names[:2]
        la, lb = annotations[a], annotations[b]
        for u in la.keys() & lb.keys():
            if la[u] == lb[u]:
                aliases[f"{b}:{u}"] = f"{a}:{u}"
            else:
                excluded |= {categories.LAB2CAT.get(la[u]), categories.LAB2CAT.get(lb[u])} - {None}
        ua = {categories.LAB2CAT[la[u]] for u in la.keys() - lb.keys() if la[u] in categories.LAB2CAT}
        ub = {categories.LAB2CAT[lb[u]] for u in lb.keys() - la.keys() if lb[u] in categories.LAB2CAT}
        excluded |= ua & ub
    return aliases, excluded, {v["recording_id"]: room[v["recording_id"]] for v in visits}


def answers(dataset, stream, cache, root, annotations):
    if dataset == "multiscan":
        return None
    aliases = stream.get("object_identity_links", {})
    excluded = set(stream.get("excluded_categories", []))
    room_truth = None
    if dataset == "3rscan":
        aliases, excluded, room_truth = rescan_identity(root, stream["visits"], annotations)
    seen = []
    convention = set()
    for visit in stream["visits"]:
        scan = visit["recording_id"]
        observed = visit_objects(cache, visit)
        row = {c: set() for c in categories.CATS}
        for u, lab in observed.items():
            cat = lab if dataset == "arkit" else categories.LAB2CAT.get(lab)
            if cat in row:
                row[cat].add(aliases.get(f"{scan}:{u}", f"{scan}:{u}"))
        if dataset == "3rscan":
            for c, labels in categories.CONF.items():
                if any(lab in labels for lab in observed.values()):
                    convention.add(c)
        seen.append(row)
    result = []
    for c in categories.CATS:
        if c in excluded or c in convention:
            continue
        per = [r[c] for r in seen]
        answer = len(set().union(*per))
        counts = list(map(len, per))
        if answer >= 1 and sum(counts) != answer:
            result.append({"category": c, "answer": answer, "per_visit_counts": counts})
    return {
        "stream_id": stream["stream_id"],
        "questions": result,
        **({"room_id_by_recording": room_truth} if room_truth is not None else {}),
    }


def extract_five_fps(dataset, root, scan, out):
    if dataset == "3rscan":
        raise ValueError("Native keyframes have no timestamp-supported five-fps reconstruction")
    if dataset == "arkit":
        split, vid = scan.split("_", 1)
        arkit.ROOT = str(root / "3dod")
        arkit.RAW = str(root / "raw")
        sc = arkit.Scan(vid, split, hires=True)
        frames = [f for f in (sc.frame(i) for i in sc.sample(5.0, 0.0)) if f is not None]
        blobs = [jpg(f.image) for f in frames]
        times = np.asarray([f.t - frames[0].t for f in frames])
    else:
        import cv2

        pre = root / scan / scan
        rows = [json.loads(l) for l in pre.with_suffix(".jsonl").read_text().splitlines() if l]
        ts = np.asarray([r["timestamp"] for r in rows], np.int64)
        native = mobile.frame_times(ts, 5)
        times = (ts[native] - ts[0]) / 1e9
        blobs = [None] * len(native)
        wanted = {int(k): i for i, k in enumerate(native)}
        assert len(wanted) == len(native), "duplicate frame indices in the 5-fps sampling"
        cap = cv2.VideoCapture(str(pre.with_suffix(".mp4")))
        for k in range(int(native.max()) + 1):
            if not cap.grab():
                raise RuntimeError((scan, "video stopped", k))
            if k in wanted:
                ok, image = cap.retrieve()
                assert ok
                image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)[:, :, ::-1]
                blobs[wanted[k]] = jpg(np.ascontiguousarray(image))
        cap.release()
        assert all(b is not None for b in blobs)
    blob, off = pack_images(blobs)
    np.savez_compressed(out, jpeg_blob=blob, jpeg_offsets=off, source_times_s=times, sampling_rate=np.asarray(5))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True, choices=["arkit", "3rscan", "multiscan"])
    ap.add_argument("--source", required=True, type=Path, help="the source dataset, as downloaded")
    ap.add_argument("--release", required=True, type=Path, help="a benchmark file, e.g. arkit_test.json")
    ap.add_argument("--output", required=True, type=Path, help="an empty folder")
    ap.add_argument("--stream", help="rebuild only this stream")
    ap.add_argument(
        "--five-fps", action="store_true", help="also write the 5-fps frames the cut detector reads (ARKit, MultiScan)"
    )
    a = ap.parse_args()
    dataset = a.dataset
    streams = read(a.release)["streams"]
    if a.stream:
        streams = [s for s in streams if s["stream_id"] == a.stream]
        assert streams, "Stream not found"
    cache = a.output / "recordings"
    os.makedirs(cache, exist_ok=True)
    indices = collections.defaultdict(set)
    for st in streams:
        for v in st["visits"]:
            indices[v["recording_id"]].update(ranges(v))
    builders = {"arkit": build_arkit, "3rscan": build_rescan, "multiscan": build_multiscan}
    annotations = {}
    for scan, ii in sorted(indices.items()):
        destination = cache / (scan + ".npz")
        if destination.exists():
            raise FileExistsError("Use an empty output folder: " + str(destination))
        annotations[scan] = builders[dataset](a.source, scan, np.asarray(sorted(ii), int), destination)
        if a.five_fps:
            extract_five_fps(dataset, a.source, scan, cache / (scan + "_five_fps.npz"))
        print(scan, "complete", flush=True)
    output = []
    mismatches = []
    for st in streams:
        rebuilt = answers(dataset, st, cache, a.source, annotations)
        if rebuilt is not None:
            output.append(rebuilt)
            if dataset == "arkit":
                expected = [{k: q[k] for k in ["category", "answer", "per_visit_counts"]} for q in st["questions"]]
                if rebuilt["questions"] != expected:
                    mismatches.append(st["stream_id"])
    (a.output / "rebuild_summary.json").write_text(
        json.dumps(
            {
                "dataset": dataset,
                "streams": len(streams),
                "recording_archives": len(indices),
                "answer_mismatches": mismatches,
                "orientation": "Frames are stored as recorded; rotate them upright with the orientation label when reading.",
            },
            indent=2,
        )
        + "\n"
    )
    if output:
        (a.output / "locally_derived_answers.json").write_text(json.dumps(output, indent=2) + "\n")
    if mismatches:
        raise SystemExit("Rebuilt answers differ from the released ones: " + str(mismatches))


if __name__ == "__main__":
    main()
