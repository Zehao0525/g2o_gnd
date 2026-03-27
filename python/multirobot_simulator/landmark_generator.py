from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Tuple, List


def landmark_generation(
    lm_path: str,
    X: int,
    top_corner: Tuple[float, float, float] = (30, 30, 30),
    bottom_corner: Tuple[float, float, float] = (0, 0, 0),
) -> Dict[str, List[float]]:
    """
    Generate X random 3D landmark coordinates in the axis-aligned cube
    defined by bottom_corner and top_corner, then save them to lm_path.

    Landmark IDs are integer-like strings: "0" .. str(X-1).
    Coordinates are sampled uniformly in each axis from:
      [bottom_corner[i], top_corner[i])
    """
    if X < 0:
        raise ValueError("X must be non-negative")

    b = tuple(float(v) for v in bottom_corner)
    t = tuple(float(v) for v in top_corner)
    if len(b) != 3 or len(t) != 3:
        raise ValueError("top_corner and bottom_corner must both have 3 elements")
    if any(t[i] <= b[i] for i in range(3)):
        raise ValueError("top_corner must be strictly greater than bottom_corner in all dimensions")

    landmarks: Dict[str, List[float]] = {}
    for idx in range(X):
        landmarks[str(idx)] = [
            random.uniform(b[0], t[0]),
            random.uniform(b[1], t[1]),
            random.uniform(b[2], t[2]),
        ]

    out_path = Path(lm_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(landmarks, f, indent=2)

    return landmarks


if __name__ == "__main__":
    landmark_generation(
        lm_path="python/multirobot_simulator/config/landmarks.json",
        X=20,
    )

