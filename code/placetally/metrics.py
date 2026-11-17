"""Strict MRA and stream-level bootstrap intervals."""

import numpy as np

THRESHOLDS = np.arange(0.5, 0.96, 0.05)


def mra(prediction, truth):
    """Mean over thresholds 0.50, 0.55, ..., 0.95 of |p - g| / g < 1 - threshold."""
    return float(np.mean([abs(prediction - truth) / truth < 1 - th for th in THRESHOLDS]))


def pair_counts(place_of_frame, room_of_frame):
    """Frame pairs in one place, frame pairs in one room, and pairs in both.

    Place F1 over a set of streams is 2 * sum(both) / (sum(in_place) + sum(in_room)).
    """
    _, place = np.unique(place_of_frame, return_inverse=True)
    _, room = np.unique(room_of_frame, return_inverse=True)
    table = np.zeros((place.max() + 1, room.max() + 1), np.int64)
    np.add.at(table, (place, room), 1)

    def pairs(x):
        return int((x * (x - 1) // 2).sum())

    return {"in_place": pairs(table.sum(1)), "in_room": pairs(table.sum(0)), "both": pairs(table)}


def bootstrap_ci(values, stream_ids, draws=5000, seed=0):
    """Mean and 95% interval, resampling whole streams.

    Each draw is the ratio of summed scores to summed questions over the drawn streams.
    """
    streams = sorted(set(stream_ids))
    index = {s: i for i, s in enumerate(streams)}
    total, count = np.zeros(len(streams)), np.zeros(len(streams))
    for v, s in zip(values, stream_ids):
        total[index[s]] += v
        count[index[s]] += 1
    draw = np.random.default_rng(seed).integers(0, len(streams), (draws, len(streams)))
    means = total[draw].sum(1) / count[draw].sum(1)
    return float(total.sum() / count.sum()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
