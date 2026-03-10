from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import DataLoader

from .config import SCENES
from .dataset import (
    ConcatSceneDataset,
    build_scene_datasets,
    collate_trajectory_batch,
    denormalize_tensor,
)
from .models import LSTMConfig, TrajectoryLSTM


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path: str = "runs/lstm_best.pt"):
    # 你的 checkpoint 是你自己本地训练出来的，所以这里可以安全地关闭 weights_only
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)

    cfg = ckpt["config"]
    if isinstance(cfg, dict):
        cfg = LSTMConfig(**cfg)

    model = TrajectoryLSTM(cfg).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    norm_map = ckpt["norm"]
    return model, cfg, norm_map


def build_test_loader(norm_map, batch_size: int = 256) -> DataLoader:
    test_datasets, _ = build_scene_datasets(
        split="test",
        train_norm_by_scene=norm_map,
    )
    test_ds = ConcatSceneDataset(test_datasets)

    return DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_trajectory_batch,
    )


@torch.no_grad()
def evaluate_lstm(model: TrajectoryLSTM, loader: DataLoader, norm_map) -> Dict[str, float]:
    total_samples = 0
    ade_sum = 0.0
    fde_sum = 0.0

    per_scene_ade_sum: Dict[str, float] = {}
    per_scene_fde_sum: Dict[str, float] = {}
    per_scene_count: Dict[str, int] = {}

    for batch in loader:
        obs = batch["obs"].to(DEVICE)           # normalized
        fut_abs = batch["fut_abs"].to(DEVICE)   # absolute
        meta = batch["meta"]

        pred_norm = model(obs)

        pred_abs_list = []
        for i in range(pred_norm.size(0)):
            scene = meta[i].scene
            pred_abs_i = denormalize_tensor(pred_norm[i], norm_map[scene])
            pred_abs_list.append(pred_abs_i)

        pred_abs = torch.stack(pred_abs_list, dim=0)

        dists = torch.norm(pred_abs - fut_abs, dim=-1)
        ade_each = dists.mean(dim=1)
        fde_each = dists[:, -1]

        bs = obs.size(0)
        total_samples += bs
        ade_sum += ade_each.sum().item()
        fde_sum += fde_each.sum().item()

        for i in range(bs):
            scene = meta[i].scene
            per_scene_ade_sum[scene] = per_scene_ade_sum.get(scene, 0.0) + ade_each[i].item()
            per_scene_fde_sum[scene] = per_scene_fde_sum.get(scene, 0.0) + fde_each[i].item()
            per_scene_count[scene] = per_scene_count.get(scene, 0) + 1

    if total_samples == 0:
        raise ValueError("No samples found in test loader during LSTM evaluation.")

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
    lines = []
    lines.append("=== LSTM Evaluation ===")
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


def main() -> None:
    model, cfg, norm_map = load_checkpoint("runs/lstm_best.pt")
    _ = cfg

    loader = build_test_loader(norm_map, batch_size=256)
    results = evaluate_lstm(model, loader, norm_map)

    print(format_results(results))


if __name__ == "__main__":
    main()