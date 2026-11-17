"""ARKitScenes frames, poses and box annotations."""

from __future__ import annotations

import glob
import json
import os

from dataclasses import dataclass
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

ROOT = ""
RAW = ""


def _traj(path: str) -> dict[str, np.ndarray]:
    poses = {}
    for line in open(path):
        tok = line.split()
        if len(tok) != 7:
            continue
        R = Rotation.from_rotvec(np.array(tok[1:4], float)).as_matrix()
        ext = np.eye(4)
        ext[:3, :3], ext[:3, 3] = (R, np.array(tok[4:7], float))
        poses[f"{round(float(tok[0]), 3):.3f}"] = np.linalg.inv(ext)
    return poses


def _boxes(path: str):
    a = json.load(open(path))
    out = []
    for d in a.get("data", []):
        seg = d["segments"]["obbAligned"]
        R = np.array(seg["normalizedAxes"]).reshape(3, 3)
        c = np.array(seg["centroid"]).reshape(3)
        size = np.array(seg["axesLengths"]).reshape(3)
        out.append({"uid": d["uid"], "label": d["label"], "R": R, "c": c, "size": size})
    return out


def _grid_points(b, n: int = 6) -> np.ndarray:
    g = (np.arange(n) + 0.5) / n - 0.5
    local = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3) * b["size"]
    return local @ b["R"] + b["c"]


@dataclass
class Frame:
    t: float
    image: np.ndarray
    gt_box: np.ndarray
    gt_uid: list
    gt_label: list
    gt_vis: np.ndarray


class Scan:

    def __init__(
        self,
        video_id: str,
        split: str = "Training",
        min_vis: float = 0.3,
        min_px: int = 100,
        hires: bool = False,
        max_dt: float = 0.02,
    ):
        base = os.path.join(ROOT, split, video_id)
        fr = os.path.join(base, f"{video_id}_frames")
        self.video_id, self.fr, self.hires, self.max_dt = (video_id, fr, hires, max_dt)
        self.raw = os.path.join(RAW, split, video_id)
        self.poses = _traj(os.path.join(fr, "lowres_wide.traj"))
        self.boxes = _boxes(os.path.join(base, f"{video_id}_3dod_annotation.json"))
        self.points = [_grid_points(b) for b in self.boxes]
        self.ids = sorted(
            (os.path.basename(p).split("_")[1][:-4] for p in glob.glob(os.path.join(fr, "lowres_wide", "*.png")))
        )
        self.min_vis, self.min_px = (min_vis, min_px)
        self._lo_ts = np.array([float(x) for x in self.ids])
        self._pose_ts = np.array(sorted((float(k) for k in self.poses)))
        if hires:
            hi = sorted(
                (os.path.basename(p).split("_")[1][:-4] for p in glob.glob(os.path.join(self.raw, "vga_wide", "*.png")))
            )
            keep = []
            for h in hi:
                th = float(h)
                if (
                    len(self._pose_ts)
                    and np.min(np.abs(self._pose_ts - th)) <= max_dt
                    and (np.min(np.abs(self._lo_ts - th)) <= max_dt)
                ):
                    keep.append(h)
            self.ids = keep

    def _pose(self, fid: str):
        if self.hires:
            k = self._pose_ts[np.argmin(np.abs(self._pose_ts - float(fid)))]
            return self.poses[f"{k:.3f}"]
        if fid in self.poses:
            return self.poses[fid]
        for k in (f"{float(fid) - 0.001:.3f}", f"{float(fid) + 0.001:.3f}"):
            if k in self.poses:
                return self.poses[k]
        return None

    def frame(self, fid: str) -> Frame | None:
        pose = self._pose(fid)
        name = f"{self.video_id}_{fid}"
        if self.hires:
            kf = os.path.join(self.raw, "vga_wide_intrinsics", f"{name}.pincam")
            img_path = os.path.join(self.raw, "vga_wide", f"{name}.png")
            lo = self.ids_lo_nearest(fid)
            depth_path = os.path.join(self.fr, "lowres_depth", f"{self.video_id}_{lo}.png")
        else:
            kf = os.path.join(self.fr, "lowres_wide_intrinsics", f"{name}.pincam")
            img_path = os.path.join(self.fr, "lowres_wide", f"{name}.png")
            depth_path = os.path.join(self.fr, "lowres_depth", f"{name}.png")
        if pose is None or not os.path.exists(kf):
            return None
        w, h, fx, fy, cx, cy = np.loadtxt(kf)
        img = np.array(Image.open(img_path).convert("RGB"))
        depth = np.array(Image.open(depth_path), np.float32) / 1000.0
        H, W = img.shape[:2]
        w2c = np.linalg.inv(pose)
        boxes, uids, labels, vis = ([], [], [], [])
        for b, P in zip(self.boxes, self.points):
            pc = P @ w2c[:3, :3].T + w2c[:3, 3]
            z = pc[:, 2]
            front = z > 0.1
            if not front.any():
                continue
            u = fx * pc[front, 0] / z[front] + cx
            v = fy * pc[front, 1] / z[front] + cy
            inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            if inside.sum() < 5:
                continue
            ui, vi, zi = (u[inside].astype(int), v[inside].astype(int), z[front][inside])
            d = depth[
                np.clip(vi * depth.shape[0] // H, 0, depth.shape[0] - 1),
                np.clip(ui * depth.shape[1] // W, 0, depth.shape[1] - 1),
            ]
            ok = (d > 0) & (zi <= d + 0.15)
            frac = ok.mean()
            if frac < self.min_vis or ok.sum() < 3:
                continue
            x1, y1, x2, y2 = (ui[ok].min(), vi[ok].min(), ui[ok].max(), vi[ok].max())
            if (x2 - x1) * (y2 - y1) < self.min_px * (W / 256.0) ** 2:
                continue
            boxes.append([x1, y1, x2, y2])
            uids.append(b["uid"])
            labels.append(b["label"])
            vis.append(frac)
        return Frame(
            float(fid), img, np.array(boxes, np.float32).reshape(-1, 4), uids, labels, np.array(vis, np.float32)
        )

    def ids_lo_nearest(self, fid: str) -> str:
        los = (
            sorted(
                (
                    os.path.basename(p).split("_")[1][:-4]
                    for p in glob.glob(os.path.join(self.fr, "lowres_depth", "*.png"))
                )
            )
            if not hasattr(self, "_lo_depth")
            else self._lo_depth
        )
        self._lo_depth = los
        ts = np.array([float(x) for x in los])
        return los[int(np.argmin(np.abs(ts - float(fid))))]

    def sample(self, fps: float = 1.0, phase: float = 0.0) -> list[str]:
        ts = np.array([float(x) for x in self.ids])
        want = np.arange(ts[0] + phase, ts[-1], 1.0 / fps)
        return [self.ids[int(np.argmin(np.abs(ts - w)))] for w in want]
