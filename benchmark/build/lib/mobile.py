"""MultiScan frame times, poses and orientation."""

from __future__ import annotations

import numpy as np

DIAG = np.diag([1.0, -1.0, -1.0])


def frame_times(ts_ns: np.ndarray, fps: int) -> np.ndarray:
    t = (ts_ns - ts_ns[0]) / 1000000000.0
    k = np.arange(0, int(np.floor(t[-1] * fps + 1e-09)) + 1)
    tt = k / fps
    idx = np.searchsorted(t, tt)
    idx = np.clip(idx, 1, len(t) - 1)
    idx = np.where(np.abs(t[idx - 1] - tt) <= np.abs(t[idx] - tt), idx - 1, idx)
    return idx.astype(np.int32)


def poses_ply(T16: np.ndarray, coord: np.ndarray):
    Tcm = T16.reshape(-1, 4, 4).transpose(0, 2, 1)
    M = np.asarray(coord, float).reshape(4, 4).T
    P = np.linalg.inv(M) @ Tcm
    return (P[:, :3, :3] @ DIAG, P[:, :3, 3])


def sky_rule(R_cv_arkit_world: np.ndarray) -> tuple[str, dict]:
    up = R_cv_arkit_world[:, 1, :3]
    m = up.mean(0)
    ax = int(np.argmax(np.abs(m[:2])))
    sign = float(np.sign(m[ax]))
    sky = {(0, -1.0): "Left", (0, 1.0): "Right", (1, -1.0): "Up", (1, 1.0): "Down"}[ax, sign]
    agree = float(np.mean(np.argmax(np.abs(up[:, :2]), 1) == ax))
    return (sky, {"up_mean_cam": m.tolist(), "axis_agree": agree, "up_z_mean": float(abs(m[2]))})


def vertex_split(V, F, oid):
    key = F.astype(np.int64) * 65536 + oid[:, None]
    flat = key.ravel()
    uk, inv = np.unique(flat, return_inverse=True)
    V2 = V[(uk // 65536).astype(np.int64)].astype(np.float32)
    O2 = (uk % 65536).astype(np.int32)
    return (V2, O2, inv.reshape(-1, 3).astype(np.int32))
