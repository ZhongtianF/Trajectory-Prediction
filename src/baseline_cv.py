from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import PRED_LEN, SCENES
from .dataset import (
    ConcatSceneDataset,
    TrajectoryDataset,
    build_scene_datasets,
    collate_trajectory_batch,
)


def constant_velocity_predict(obs_abs: torch.Tensor, pred_len: int = PRED_LEN) -> torch.Tensor:
    """
    Constant velocity baseline in absolute coordinates.

    Args:
        obs_abs: (B, obs_len, 2)
        pred_len: future length

    Returns:
        pred_abs: (B, pred_len, 2)
    """
    if obs_abs.ndim != 3 or obs_abs.size(-1) != 2:
        raise ValueError(f"Expected obs_abs shape (B, T, 2), got {tuple(obs_abs.shape)}")

    if obs_abs.size(1) < 2:
        raise ValueError("Need at least 2 observed points for constant velocity prediction.")

    last_pos = obs_abs[:, -1, :]               # (B, 2)
    prev_pos = obs_abs[:, -2, :]               # (B, 2)
    velocity = last_pos - prev_pos             # (B, 2)

    preds = []
    current = last_pos
    for _ in range(pred_len):
        current = current + velocity
        preds.append(current)

    return torch.stack(preds, dim=1)           # (B, pred_len, 2)


def compute_ade_fde(pred_abs: torch.Tensor, fut_abs: torch.Tensor) -> Tuple[float, float]:
    """
    Args:
        pred_abs: (B, pred_len, 2)
        fut_abs:  (B, pred_len, 2)

    Returns:
        ade, fde
    """
    if pred_abs.shape != fut_abs.shape:
        raise ValueError(
            f"Prediction and target shapes must match, got {tuple(pred_abs.shape)} vs {tuple(fut_abs.shape)}"
        )

    dists = torch.norm(pred_abs - fut_abs, dim=-1)   # (B, pred_len)
    ade = dists.mean().item()
    fde = dists[:, -1].mean().item()
    return ade, fde


@torch.no_grad()
def evaluate_loader_cv(loader: DataLoader) -> Dict[str, float]:
    total_samples = 0
    ade_sum = 0.0
    fde_sum = 0.0

    per_scene_ade_sum: Dict[str, float] = {}
    per_scene_fde_sum: Dict[str, float] = {}
    per_scene_count: Dict[str, int] = {}

    for batch in loader:
        obs_abs = batch["obs_abs"]   # (B, 8, 2)
        fut_abs = batch["fut_abs"]   # (B, 12, 2)
        meta = batch["meta"]

        pred_abs = constant_velocity_predict(obs_abs, pred_len=fut_abs.size(1))

        dists = torch.norm(pred_abs - fut_abs, dim=-1)  # (B, pred_len)
        ade_each = dists.mean(dim=1)                    # (B,)
        fde_each = dists[:, -1]                         # (B,)

        bs = obs_abs.size(0)
        total_samples += bs
        ade_sum += ade_each.sum().item()
        fde_sum += fde_each.sum().item()

        for i in range(bs):
            scene = meta[i].scene
            per_scene_ade_sum[scene] = per_scene_ade_sum.get(scene, 0.0) + ade_each[i].item()
            per_scene_fde_sum[scene] = per_scene_fde_sum.get(scene, 0.0) + fde_each[i].item()
            per_scene_count[scene] = per_scene_count.get(scene, 0) + 1

    if total_samples == 0:
        raise ValueError("No samples found in loader during CV evaluation.")

    results: Dict[str, float] = {
        "ade": ade_sum / total_samples,
        "fde": fde_sum / total_samples,
        "num_samples": float(total_samples),
    }

    for scene in sorted(per_scene_count.keys()):
        count = per_scene_count[scene]
        results[f"{scene}_ade"] = per_scene_ade_sum[scene] / count
        results[f"{scene}_fde"] = per_scene_fde_sum[scene] / count
        results[f"{scene}_count"] = float(count)

    return results


def format_results(results: Dict[str, float]) -> str:
    lines: List[str] = []
    lines.append("=== Constant Velocity Baseline ===")
    lines.append(f"Total samples: {int(results['num_samples'])}")
    lines.append(f"Global ADE   : {results['ade']:.4f}")
    lines.append(f"Global FDE   : {results['fde']:.4f}")
    lines.append("")

    for scene in SCENES:
        ade_key = f"{scene}_ade"
        fde_key = f"{scene}_fde"
        cnt_key = f"{scene}_count"
        if ade_key in results:
            lines.append(
                f"{scene:10s} | count={int(results[cnt_key]):5d} | "
                f"ADE={results[ade_key]:.4f} | FDE={results[fde_key]:.4f}"
            )

    return "\n".join(lines)


def build_test_loader_all_scenes(batch_size: int = 128) -> DataLoader:
    train_datasets, norm_map = build_scene_datasets(split="train")
    _ = train_datasets  # only to get train norms

    test_datasets, _ = build_scene_datasets(
        split="test",
        train_norm_by_scene=norm_map,
    )

    concat_ds = ConcatSceneDataset(test_datasets)

    return DataLoader(
        concat_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_trajectory_batch,
    )


def build_test_loader_one_scene(scene: str, batch_size: int = 128) -> DataLoader:
    train_ds = TrajectoryDataset(scene=scene, split="train")
    test_ds = TrajectoryDataset(scene=scene, split="test", norm_stats=train_ds.norm_stats)

    return DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_trajectory_batch,
    )


def main() -> None:
    loader = build_test_loader_all_scenes(batch_size=256)
    results = evaluate_loader_cv(loader)
    print(format_results(results))


if __name__ == "__main__":
    main()