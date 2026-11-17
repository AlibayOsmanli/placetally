"""Objects that count as seen in a visit: a box of at least 0.5% of the image in at least two frames."""

from collections import defaultdict


def scan_visits(z: dict, p: int, min_area: float = 0.005, min_frames: int = 2, img_area: float = 640 * 480):
    labels = {str(u): str(l) for (u, l) in zip(z["uids_all"], z["labels_all"])}
    ptr, boxes, uids = (z[f"p{p}_gt_ptr"], z[f"p{p}_gt_box"], z[f"p{p}_gt_uid"])
    seen: dict = defaultdict(lambda: [0, -1, None, 0.0])
    for f in range(len(ptr) - 1):
        for k in range(ptr[f], ptr[f + 1]):
            b = boxes[k]
            area = (b[2] - b[0]) * (b[3] - b[1])
            if area < min_area * img_area:
                continue
            e = seen[str(uids[k])]
            e[0] += 1
            if area > e[3]:
                e[1], e[2], e[3] = (f, b.tolist(), float(area))
    return {u: (labels.get(u, "?"), n, f, b) for (u, (n, f, b, _)) in seen.items() if n >= min_frames}
