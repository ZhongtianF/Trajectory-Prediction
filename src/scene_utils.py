from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np

from .config import DEFAULT_IMAGE_EXTS, get_scene_dir


# ===== ETH / UCY Homography matrices =====
MANUAL_HOMOGRAPHIES = {

    "eth": np.array([
        [2.0091900e-03, 2.8128700e-02, -4.6693600e+00],
        [2.5195500e-02, 8.0625700e-04, -5.0608800e+00],
        [9.2512200e-05, 3.4555400e-04, 4.6255300e-01]
    ], dtype=np.float32),

    "hotel": np.array([
        [6.6958900e-04, 1.1048200e-02, -3.3295300e+00],
        [1.1632400e-02, -1.5966000e-03, -5.3951400e+00],
        [1.3617400e-05, 1.1190700e-04, 5.4276600e-01]
    ], dtype=np.float32),

    "university": np.array([
        [2.12388838e-02, 8.08142657e-05, 1.98877145e-01],
        [4.42541891e-04, -2.37229796e-02, 1.24948357e+01],
        [4.51911949e-05, 5.76091505e-05, 1.00000000e+00]
    ], dtype=np.float32),

    "zara_01": np.array([
        [1.99364401e-02, -7.58906793e-04, 3.19416932e-01],
        [7.60068320e-05, -2.39606910e-02, 1.24160929e+01],
        [4.16500286e-05, -1.85270181e-04, 1.00000000e+00]
    ], dtype=np.float32),

    "zara_02": np.array([
        [2.02309381e-02, -1.68186653e-03, 1.17934728e-01],
        [-2.03265309e-04, -2.41277040e-02, 1.36976107e+01],
        [-5.26400160e-05, -1.29813681e-04, 1.00000000e+00]
    ], dtype=np.float32),
}


def get_homography(scene: str) -> Optional[np.ndarray]:
    return MANUAL_HOMOGRAPHIES.get(scene)


# ---------- scene images ----------

def _all_scene_images(scene: str):
    scene_dir = get_scene_dir(scene)

    candidates = []

    for ext in DEFAULT_IMAGE_EXTS:
        candidates.extend(scene_dir.rglob(f"*{ext}"))

    return sorted(candidates)


def _extract_frame_number(path: Path) -> Optional[int]:

    stem = path.stem

    m = re.search(r"frame0*([0-9]+)", stem, flags=re.IGNORECASE)

    if m:
        return int(m.group(1))

    nums = re.findall(r"\d+", stem)

    if nums:
        return int(nums[-1])

    return None


def find_closest_scene_image(scene: str, frame_num: int) -> Path:

    candidates = _all_scene_images(scene)

    if not candidates:
        raise FileNotFoundError(f"No scene image found for scene '{scene}'")

    best = None
    best_dist = None

    for p in candidates:

        f = _extract_frame_number(p)

        if f is None:
            continue

        dist = abs(f - frame_num)

        if best is None or dist < best_dist:

            best = p
            best_dist = dist

    return best if best is not None else candidates[0]


# ---------- projection ----------

def world_to_image(points_xy: np.ndarray, H: np.ndarray) -> np.ndarray:

    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        raise ValueError("Expected (N,2) points")

    H_inv = np.linalg.inv(H)

    ones = np.ones((points_xy.shape[0], 1), dtype=np.float32)

    pts = np.concatenate([points_xy, ones], axis=1)

    proj = (H_inv @ pts.T).T

    proj = proj / proj[:, 2:3]

    return proj[:, :2]