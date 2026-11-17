"""3RScan and MultiScan visibility: object boxes from a z-buffered mesh render."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from numba import njit

B = Path(".")
SCALE = 0.5


def load_ply(scan: str):
    p = B / "scans" / scan / "labels.instances.annotated.v2.ply"
    with open(p, "rb") as f:
        nv = nf = 0
        while True:
            line = f.readline().decode()
            if line.startswith("element vertex"):
                nv = int(line.split()[-1])
            elif line.startswith("element face"):
                nf = int(line.split()[-1])
            elif line.startswith("end_header"):
                break
        txt = f.read().decode().split("\n")
    v = np.array([l.split() for l in txt[:nv]], dtype=np.float64)
    fc = np.array([l.split()[1:4] for l in txt[nv : nv + nf]], dtype=np.int32)
    return (v[:, :3].astype(np.float32), v[:, 6].astype(np.int32), fc)


def semseg(scan: str) -> dict:
    return {int(o["id"]): o["label"] for o in json.load(open(B / "scans" / scan / "semseg.v2.json"))["segGroups"]}


@njit(cache=True)
def raster(u, v, z, faces, H, W):
    depth = np.full((H, W), np.inf, np.float32)
    for m in range(faces.shape[0]):
        a, b, c = (faces[m, 0], faces[m, 1], faces[m, 2])
        if z[a] < 0.05 or z[b] < 0.05 or z[c] < 0.05:
            continue
        x0, y0, x1, y1, x2, y2 = (u[a], v[a], u[b], v[b], u[c], v[c])
        xmin = max(int(np.floor(min(x0, x1, x2))), 0)
        xmax = min(int(np.ceil(max(x0, x1, x2))), W - 1)
        ymin = max(int(np.floor(min(y0, y1, y2))), 0)
        ymax = min(int(np.ceil(max(y0, y1, y2))), H - 1)
        if xmin > xmax or ymin > ymax:
            continue
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(den) < 1e-12:
            continue
        for yy in range(ymin, ymax + 1):
            for xx in range(xmin, xmax + 1):
                px, py = (xx + 0.5, yy + 0.5)
                l0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / den
                l1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / den
                l2 = 1.0 - l0 - l1
                if l0 < -0.02 or l1 < -0.02 or l2 < -0.02:
                    continue
                d = l0 * z[a] + l1 * z[b] + l2 * z[c]
                if d < depth[yy, xx]:
                    depth[yy, xx] = d
    return depth


def project(verts, pose, K):
    R, t = (pose[:3, :3], pose[:3, 3])
    pc = (verts.astype(np.float64) - t) @ R
    z = pc[:, 2]
    zs = np.where(z > 1e-06, z, 1e-06)
    return (K[0, 0] * pc[:, 0] / zs + K[0, 2], K[1, 1] * pc[:, 1] / zs + K[1, 2], z)


def render_depth(verts, faces, pose, K, W, H):
    u, v, z = project(verts, pose, K)
    h, w = (int(H * SCALE), int(W * SCALE))
    d = raster((u * SCALE).astype(np.float32), (v * SCALE).astype(np.float32), z.astype(np.float32), faces, h, w)
    return (d, u, v, z)


def frame_truth(verts, oid, faces, pose, K, W, H, min_vis=0.3, min_frac=0.002):
    d, u, v, z = render_depth(verts, faces, pose, K, W, H)
    ins = (z > 0.1) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    ui, vi = (np.clip((u * SCALE).astype(int), 0, d.shape[1] - 1), np.clip((v * SCALE).astype(int), 0, d.shape[0] - 1))
    dz = d[vi, ui]
    ok = ins & (z <= dz + 0.1 + 0.02 * z)
    out = []
    idx = np.flatnonzero(ins)
    if not len(idx):
        return out
    o_in = oid[idx]
    for o in np.unique(o_in):
        sel = idx[o_in == o]
        vis = sel[ok[sel]]
        if len(sel) < 5 or len(vis) < 3 or len(vis) / len(sel) < min_vis:
            continue
        x1, x2, y1, y2 = (u[vis].min(), u[vis].max(), v[vis].min(), v[vis].max())
        if (x2 - x1) * (y2 - y1) < min_frac * W * H:
            continue
        out.append((int(o), [x1 * 640 / W, y1 * 480 / H, x2 * 640 / W, y2 * 480 / H], float(len(vis) / len(sel))))
    return out
