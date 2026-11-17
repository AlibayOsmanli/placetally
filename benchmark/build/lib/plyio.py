"""Reading MultiScan PLY meshes."""

import numpy as np


def read_ply(buf):
    i = buf.index(b"end_header\n") + len(b"end_header\n")
    hdr = buf[:i].decode()
    nv = int(hdr.split("element vertex ")[1].split()[0])
    nf = int(hdr.split("element face ")[1].split()[0])
    vd = np.dtype(
        [("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("nx", "<f8"), ("ny", "<f8"), ("nz", "<f8"), ("c", "u1", (4,))]
    )
    v = np.frombuffer(buf, vd, nv, i)
    j = i + nv * vd.itemsize
    fd = np.dtype([("n", "u1"), ("a", "<i4"), ("b", "<i4"), ("c", "<i4"), ("o", "<u2"), ("p", "<u2")])
    f = np.frombuffer(buf, fd, nf, j)
    assert (f["n"] == 3).all() and j + nf * fd.itemsize == len(buf), (hdr, len(buf))
    V = np.stack([v["x"], v["y"], v["z"]], 1)
    return (V, np.stack([f["a"], f["b"], f["c"]], 1).astype(np.int32), f["o"].astype(np.int32), f["p"].astype(np.int32))
