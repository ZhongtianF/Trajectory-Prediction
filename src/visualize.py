from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .dataset import TrajectoryDataset, denormalize_tensor
from .models import LSTMConfig, TrajectoryLSTM
from .scene_utils import get_homography, find_closest_scene_image, world_to_image


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path: str = "runs/lstm_best.pt"):

    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)

    cfg = ckpt["config"]

    if isinstance(cfg, dict):
        cfg = LSTMConfig(**cfg)

    model = TrajectoryLSTM(cfg).to(DEVICE)

    model.load_state_dict(ckpt["model_state"])

    model.eval()

    return model, ckpt["norm"]


def draw_polyline(ax, pts, color, label):

    ax.plot(
        pts[:, 0],
        pts[:, 1],
        color=color,
        linewidth=2.5,
        marker="o",
        markersize=4,
        label=label
    )


@torch.no_grad()
def visualize_scene(
    scene: str,
    sample_index: int = 0,
    checkpoint_path: str = "runs/lstm_best.pt",
):

    model, norm_map = load_checkpoint(checkpoint_path)

    train_ds = TrajectoryDataset(scene=scene, split="train")

    test_ds = TrajectoryDataset(scene=scene, split="test", norm_stats=train_ds.norm_stats)

    sample = test_ds[sample_index]

    meta = sample["meta"]

    obs = sample["obs"].unsqueeze(0).to(DEVICE)

    obs_abs = sample["obs_abs"].cpu().numpy()

    fut_abs = sample["fut_abs"].cpu().numpy()

    pred_norm = model(obs)[0]

    pred_abs = denormalize_tensor(pred_norm, norm_map[scene]).cpu().numpy()

    img_path = find_closest_scene_image(scene, meta.start_frame)

    img = Image.open(img_path).convert("RGB")

    img_np = np.array(img)

    H = get_homography(scene)

    obs_img = world_to_image(obs_abs, H)

    fut_img = world_to_image(fut_abs, H)

    pred_img = world_to_image(pred_abs, H)

    out_dir = Path("runs/vis")

    out_dir.mkdir(parents=True, exist_ok=True)

    save_file = out_dir / f"{scene}_sample_{sample_index}.png"

    plt.figure(figsize=(10, 8))

    ax = plt.gca()

    ax.imshow(img_np)

    draw_polyline(ax, obs_img, "blue", "Observed")

    draw_polyline(ax, fut_img, "green", "Ground Truth")

    draw_polyline(ax, pred_img, "red", "Predicted")

    ax.set_title(f"{scene}  frame={meta.start_frame}")

    ax.legend()

    ax.axis("off")

    plt.tight_layout()

    plt.savefig(save_file, dpi=200)

    plt.close()

    print("Saved:", save_file)


def main():

    scenes = [
        "eth",
        "hotel",
        "university",
        "zara_01",
        "zara_02",
    ]

    for scene in scenes:

        visualize_scene(scene, sample_index=0)


if __name__ == "__main__":
    main()