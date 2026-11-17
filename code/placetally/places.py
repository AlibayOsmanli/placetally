"""Cutting a stream into fragments and linking the fragments into places (the method section of the paper)."""

import numpy as np

from . import settings
from .features import pair_features, unit


class CutDetector:
    """Logistic regression on standardised neighbouring-frame features."""

    def __init__(self, mean, scale, weights, intercept):
        self.mean = np.asarray(mean, np.float64)
        self.scale = np.asarray(scale, np.float64)
        self.weights = np.asarray(weights, np.float32).reshape(1, -1)
        self.intercept = np.asarray(intercept, np.float32).reshape(1)

    def log_odds(self, features):
        # Column-major, as scikit-learn computes it; the layout changes the last float32 bit.
        x = np.array(features, np.float32, order="F")
        x -= self.mean.astype(x.dtype)
        x /= self.scale.astype(x.dtype)
        return (x @ self.weights.T + self.intercept).reshape(-1)


def find_cuts(log_odds, threshold=settings.CUT_THRESHOLD, gap=settings.PEAK_GAP):
    """Peaks above the threshold, taken strongest first and kept at least `gap` pairs apart."""
    candidates = sorted(((s, i) for i, s in enumerate(log_odds) if s > threshold), reverse=True)
    kept = []
    for _, i in candidates:
        if all(abs(i - j) >= gap for j in kept):
            kept.append(i)
    return sorted(kept)


def continuation_probabilities(model, cuts):
    """p_t for every fragment after the first: the probability that the cut before it is false.

    `cuts` holds one dict of statistics per cut. The first fragment gets NaN.
    """
    if not cuts:
        return [np.nan]
    x = np.array([[c[name] for name in model["features"]] for c in cuts])
    z = (x - np.array(model["mean"])) / np.array(model["sd"])
    return [np.nan] + list(1 / (1 + np.exp(-(z @ np.array(model["weights"]) + model["intercept"]))))


def chamfer(a, b):
    """Symmetric chamfer similarity of two sets of unit vectors."""
    s = a @ b.T
    return 0.5 * (float(s.max(1).mean()) + float(s.max(0).mean()))


def link_fragments(p_continue, similarity, level, tau=settings.TAU):
    """The best-link rule. Returns a place label for each fragment.

    A fragment continues the previous place if p_t >= level. Otherwise it joins the place of its most
    similar earlier fragment if that similarity is above tau, and opens a new place if not.
    """
    place = [0]
    for t in range(1, len(p_continue)):
        if p_continue[t] >= level:
            place.append(place[t - 1])
            continue
        j = max(range(t), key=lambda j: similarity[t, j])
        place.append(place[j] if similarity[t, j] > tau else max(place) + 1)
    return place


def find_fragments(thumbnail, histogram, detector_codes, positions, detector):
    """Cut the stream. Returns the fragments (lists of 1-fps frame indices) and the log-odds of each cut.

    thumbnail, histogram, detector_codes: features of the 5-fps frames. positions: index of each 1-fps frame
    among the 5-fps frames. Cuts with no 1-fps frame between them fold into one; the cut keeps the highest log-odds.
    """
    log_odds = detector.log_odds(pair_features(thumbnail, histogram, detector_codes))
    cuts = find_cuts(log_odds)
    by_gap = {}
    for i, pos in enumerate(positions):
        by_gap.setdefault(sum(1 for c in cuts if c < pos), []).append(i)
    gaps = sorted(by_gap)
    fragments = [by_gap[g] for g in gaps]
    cut_log_odds = [max(float(log_odds[cuts[c]]) for c in range(gaps[t - 1], gaps[t])) for t in range(1, len(gaps))]
    return fragments, cut_log_odds


def fragment_similarities(codes, fragments, projection):
    """Raw and learned chamfer similarity between every two fragments.

    Both are kept in float32, as in the paper's runs.
    """
    projected = unit(codes @ projection)
    n = len(fragments)
    raw = np.zeros((n, n), np.float32)
    learned = np.zeros((n, n), np.float32)
    for t in range(1, n):
        for j in range(t):
            raw[t, j] = raw[j, t] = chamfer(codes[fragments[t]], codes[fragments[j]])
            learned[t, j] = learned[j, t] = chamfer(projected[fragments[t]], projected[fragments[j]])
    return raw, learned


def cut_statistics(fragments, cut_log_odds, raw, learned):
    """The six statistics the continuation model reads at each cut."""
    return [
        {
            "cut_log_odds": cut_log_odds[t - 1],
            "learned_similarity_previous": float(learned[t, t - 1]),
            "raw_similarity_previous": float(raw[t, t - 1]),
            "best_learned_similarity_older": float(max([learned[t, j] for j in range(t - 1)], default=0.0)),
            "log_length_new": float(np.log(len(fragments[t]))),
            "log_length_previous": float(np.log(len(fragments[t - 1]))),
        }
        for t in range(1, len(fragments))
    ]


def find_places(codes, thumbnail, histogram, detector_codes, positions, weights):
    """Group the 1-fps frames of a stream into places.

    codes: unit-length SigLIP 2 codes of the 1-fps frames; the other inputs are as in find_fragments.
    Returns a sorted list of 1-fps frame indices for each place.
    """
    fragments, cut_log_odds = find_fragments(thumbnail, histogram, detector_codes, positions, weights.detector)
    raw, learned = fragment_similarities(codes, fragments, weights.projection)
    stats = cut_statistics(fragments, cut_log_odds, raw, learned)
    p_continue = continuation_probabilities(weights.continuation, stats)
    labels = link_fragments(p_continue, raw.astype(np.float64), weights.continuation["level"])

    places = {}
    for fragment, label in zip(fragments, labels):
        places.setdefault(label, []).extend(fragment)
    return [sorted(frames) for _, frames in sorted(places.items())]
