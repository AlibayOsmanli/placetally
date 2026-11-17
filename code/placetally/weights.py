"""The trained parts of PlaceTally: cut detector, continuation model and projection."""

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np

from .places import CutDetector


class Weights(NamedTuple):
    detector: CutDetector
    continuation: dict
    projection: np.ndarray


def load_weights(directory):
    """Load the three trained parts from a folder such as the archive's weights/."""
    directory = Path(directory)
    detector = json.loads((directory / "detector.json").read_text())
    continuation = json.loads((directory / "continuation.json").read_text())
    projection = np.load(directory / "projection.npy")
    return Weights(CutDetector(**detector), continuation, projection)
