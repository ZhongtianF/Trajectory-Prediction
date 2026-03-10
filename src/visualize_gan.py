from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .dataset import TrajectoryDataset, denormalize_tensor
from .gan_models import GANConfig, TrajectoryGenerator, sample_noise
from .scene_utils import get_homography, find_closest_scene_image, world_to_image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path: str = "runs/gan_best.pt"):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)

    cfg = ckpt["config"]
    if isinstance(cfg, dict):
        cfg = GANConfig(**cfg)

    generator = TrajectoryGenerator(cfg).to(DEVICE)
    generator.load_state_dict(ckpt["generator_state"])
    generator.eval()

    norm_map = ckpt["norm"]
    scene_feature_bank = {k: v.to(DEVICE) for k, v in ckpt["scene_feature_bank"].items()}

    return generator, cfg, norm_map, scene_feature_bank


def draw_polyline(ax, pts, color, label=None, alpha=1.0, linewidth=2.0, zorder=3):
    ax.plot(
        pts[:, 0],
        pts[:, 1],
        color=color,
        linewidth=linewidth,
        marker="o",
        markersize=3,
        alpha=alpha,
        label=label,
        zorder=zorder,
    )


@torch.no_grad()
def visualize_gan_scene(
    scene: str,
    sample_index: int = 0,
    k_samples: int = 20,
    checkpoint_path: str = "runs/gan_best.pt",
    out_dir: str = "runs/vis_gan",
):
    generator, cfg, norm_map, feat_bank = load_checkpoint(checkpoint_path)

    train_ds = TrajectoryDataset(scene=scene, split="train")
    test_ds = TrajectoryDataset(scene=scene, split="test", norm_stats=train_ds.norm_stats)

    if sample_index >= len(test_ds):
        raise IndexError(
            f"sample_index={sample_index} out of range for scene '{scene}', len={len(test_ds)}"
        )

    sample = test_ds[sample_index]
    meta = sample["meta"]

    obs = sample["obs"].unsqueeze(0).to(DEVICE)
    obs_abs = sample["obs_abs"].cpu().numpy()
    fut_abs = sample["fut_abs"].cpu().numpy()

    scene_feat = feat_bank[scene].unsqueeze(0).to(DEVICE)

    pred_abs_list = []
    best_pred_abs = None
    best_ade = None

    fut_abs_tensor = sample["fut_abs"].unsqueeze(0).to(DEVICE)

    for _ in range(k_samples):
        noise = sample_noise(1, cfg.noise_dim, DEVICE)
        pred_norm = generator(obs, noise, scene_feat=scene_feat)[0]
        pred_abs = denormalize_tensor(pred_norm, norm_map[scene]).cpu().numpy()
        pred_abs_list.append(pred_abs)

        pred_abs_tensor = torch.tensor(pred_abs, dtype=fut_abs_tensor.dtype, device=DEVICE).unsqueeze(0)
        ade = torch.norm(pred_abs_tensor - fut_abs_tensor, dim=-1).mean().item()

        if best_ade is None or ade < best_ade:
            best_ade = ade
            best_pred_abs = pred_abs

    img_path = find_closest_scene_image(scene, meta.start_frame)
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)

    H = get_homography(scene)
    if H is None:
        raise ValueError(f"No homography available for scene '{scene}'")

    obs_img = world_to_image(obs_abs, H)
    fut_img = world_to_image(fut_abs, H)

    pred_img_list = [world_to_image(pred_abs, H) for pred_abs in pred_abs_list]
    best_pred_img = world_to_image(best_pred_abs, H)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    save_file = out_path / f"{scene}_sample_{sample_index}_k{k_samples}.png"

    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    ax.imshow(img_np)

    # all sampled futures
    for idx, pred_img in enumerate(pred_img_list):
        lbl = "Sampled futures" if idx == 0 else None
        draw_polyline(ax, pred_img, color="red", label=lbl, alpha=0.18, linewidth=1.2, zorder=2)

    # observed and GT
    draw_polyline(ax, obs_img, color="blue", label="Observed", alpha=1.0, linewidth=2.8, zorder=5)
    draw_polyline(ax, fut_img, color="green", label="Ground Truth", alpha=1.0, linewidth=2.8, zorder=5)

    # best sampled future
    draw_polyline(ax, best_pred_img, color="orange", label="Best sample", alpha=1.0, linewidth=2.8, zorder=6)

    ax.set_title(
        f"{scene} | sample={sample_index} | frame={meta.start_frame} | K={k_samples}\n"
        f"GAN multimodal prediction"
    )
    ax.legend()
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_file, dpi=220, bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_file}")


def main():
    scenes = ["eth", "hotel", "university", "zara_01", "zara_02"]

    for scene in scenes:
        visualize_gan_scene(scene=scene, sample_index=0, k_samples=20)


if __name__ == "__main__":
    main()