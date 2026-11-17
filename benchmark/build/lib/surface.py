"""ARKitScenes visibility: which annotated boxes are visible in a frame, checked against sensor depth."""

from __future__ import annotations

import glob
import os

import numpy as np
from PIL import Image
from .arkit import _traj, _boxes, _grid_points

ROOT = ""
RAW = ""
W, H = 640, 480
MIN_PX = 100 * (W / 256.0) ** 2
TOL_ABS, TOL_REL, MIN_FRAC, MIN_N, NS = 0.05, 0.03, 0.15, 12, 8


def face_points(b, n: int = NS):
    g = (np.arange(n) + 0.5) / n - 0.5
    u, v = np.meshgrid(g, g, indexing="ij")
    u, v = (u.reshape(-1), v.reshape(-1))
    pts, nrm, ctr = ([], [], [])
    for ax in range(3):
        o = [a for a in range(3) if a != ax]
        for s in (-0.5, 0.5):
            loc = np.zeros((n * n, 3))
            loc[:, ax], loc[:, o[0]], loc[:, o[1]] = (s, u, v)
            pts.append(loc * b["size"] @ b["R"] + b["c"])
            nrm.append(np.sign(s) * b["R"][ax])
            ctr.append(s * b["size"][ax] * b["R"][ax] + b["c"])
    return (np.stack(pts), np.stack(nrm), np.stack(ctr))


class HiresScan:

    def __init__(self, vid: str, split: str, max_dt: float = 0.02):
        base = os.path.join(ROOT, split, vid)
        self.vid, self.fr, self.raw = (vid, os.path.join(base, f"{vid}_frames"), os.path.join(RAW, split, vid))
        self.poses = _traj(os.path.join(self.fr, "lowres_wide.traj"))
        self.boxes = _boxes(os.path.join(base, f"{vid}_3dod_annotation.json"))
        self.grid = [_grid_points(b) for b in self.boxes]
        self.faces = [face_points(b) for b in self.boxes]
        lo = sorted(
            (os.path.basename(p).split("_")[1][:-4] for p in glob.glob(os.path.join(self.fr, "lowres_wide", "*.png")))
        )
        self._lo_ts = np.array([float(x) for x in lo])
        self._pose_ts = np.array(sorted((float(k) for k in self.poses)))
        self._dep = sorted(
            (os.path.basename(p).split("_")[1][:-4] for p in glob.glob(os.path.join(self.fr, "lowres_depth", "*.png")))
        )
        self._dep_ts = np.array([float(x) for x in self._dep])
        hi = sorted(
            (
                os.path.basename(p).split("_")[1][:-7]
                for p in glob.glob(os.path.join(self.raw, "vga_wide_intrinsics", "*.pincam"))
            )
        )
        self.ids = [
            h
            for h in hi
            if len(self._pose_ts)
            and np.min(np.abs(self._pose_ts - float(h))) <= max_dt
            and (np.min(np.abs(self._lo_ts - float(h))) <= max_dt)
        ]

    def sample(self, fps: float, phase: float) -> list[str]:
        ts = np.array([float(x) for x in self.ids])
        want = np.arange(ts[0] + phase, ts[-1], 1.0 / fps)
        return [self.ids[int(np.argmin(np.abs(ts - w)))] for w in want]

    def geometry(self, fid: str):
        k = self._pose_ts[np.argmin(np.abs(self._pose_ts - float(fid)))]
        pose = self.poses[f"{k:.3f}"]
        _, _, fx, fy, cx, cy = np.loadtxt(os.path.join(self.raw, "vga_wide_intrinsics", f"{self.vid}_{fid}.pincam"))
        lo = self._dep[int(np.argmin(np.abs(self._dep_ts - float(fid))))]
        depth = np.array(Image.open(os.path.join(self.fr, "lowres_depth", f"{self.vid}_{lo}.png")), np.float32) / 1000.0
        return (pose, (fx, fy, cx, cy), depth)


def project(P, w2c, K, depth):
    fx, fy, cx, cy = K
    pc = P @ w2c[:3, :3].T + w2c[:3, 3]
    z = pc[:, 2]
    keep = z > 0.1
    u = np.full(len(P), -1.0)
    v = np.full(len(P), -1.0)
    u[keep] = fx * pc[keep, 0] / z[keep] + cx
    v[keep] = fy * pc[keep, 1] / z[keep] + cy
    keep &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
    ui, vi = (u[keep].astype(int), v[keep].astype(int))
    d = depth[
        np.clip(vi * depth.shape[0] // H, 0, depth.shape[0] - 1),
        np.clip(ui * depth.shape[1] // W, 0, depth.shape[1] - 1),
    ]
    return (keep, ui, vi, z[keep], d)


def visible_box(faces, pose, w2c, K, depth, tol_abs=TOL_ABS, tol_rel=TOL_REL):
    pts, nrm, ctr = faces
    cam = pose[:3, 3]
    front = ((cam - ctr) * nrm).sum(1) > 0
    if not front.any():
        return (None, 0, 0)
    keep, ui, vi, z, d = project(pts[front].reshape(-1, 3), w2c, K, depth)
    known = d > 0
    ui, vi, z, d = (ui[known], vi[known], z[known], d[known])
    if len(z) == 0:
        return (None, 0, 0)
    ok = z <= d + np.maximum(tol_abs, tol_rel * d)
    frac, n = (float(ok.mean()), int(ok.sum()))
    if frac < MIN_FRAC or n < MIN_N:
        return (None, frac, n)
    x1, y1, x2, y2 = (ui[ok].min(), vi[ok].min(), ui[ok].max(), vi[ok].max())
    if (x2 - x1) * (y2 - y1) < MIN_PX:
        return (None, frac, n)
    return ([x1, y1, x2, y2], frac, n)
