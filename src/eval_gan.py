from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader

from .config import SCENES
from .dataset import (
    ConcatSceneDataset,
    build_scene_datasets,
    collate_trajectory_batch,
    denormalize_tensor,
)
from .gan_models import GANConfig, TrajectoryGenerator, sample_noise

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


def get_scene_features_for_batch(meta, feat_bank: Dict[str, torch.Tensor]) -> torch.Tensor:
    feats = [feat_bank[m.scene] for m in meta]
    return torch.stack(feats, dim=0).to(DEVICE)


def ade_fde_per_sample(pred_abs: torch.Tensor, fut_abs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    pred_abs, fut_abs: (B, pred_len, 2)
    returns:
        ade_each: (B,)
        fde_each: (B,)
    """
    dists = torch.norm(pred_abs - fut_abs, dim=-1)
    ade_each = dists.mean(dim=1)
    fde_each = dists[:, -1]
    return ade_each, fde_each


@torch.no_grad()
def evaluate_gan(
    generator: TrajectoryGenerator,
    loader: DataLoader,
    norm_map,
    feat_bank,
    k_samples: int = 20,
):
    total_samples = 0

    ade_sum_1 = 0.0
    fde_sum_1 = 0.0

    ade_sum_bestk = 0.0
    fde_sum_bestk = 0.0

    per_scene = {}

    for batch in loader:
        obs = batch["obs"].to(DEVICE)           # normalized
        fut_abs = batch["fut_abs"].to(DEVICE)   # absolute
        meta = batch["meta"]

        bs = obs.size(0)
        scene_feat = get_scene_features_for_batch(meta, feat_bank)

        # ---- single sample metrics ----
        noise = sample_noise(bs, generator.cfg.noise_dim, DEVICE)
        pred_norm_1 = generator(obs, noise, scene_feat=scene_feat)

        pred_abs_1_list = []
        for i in range(bs):
            scene = meta[i].scene
            pred_abs_i = denormalize_tensor(pred_norm_1[i], norm_map[scene])
            pred_abs_1_list.append(pred_abs_i)
        pred_abs_1 = torch.stack(pred_abs_1_list, dim=0)

        ade_1_each, fde_1_each = ade_fde_per_sample(pred_abs_1, fut_abs)

        # ---- Best-of-K metrics ----
        best_ade_each = None
        best_fde_each = None

        for _ in range(k_samples):
            noise_k = sample_noise(bs, generator.cfg.noise_dim, DEVICE)
            pred_norm_k = generator(obs, noise_k, scene_feat=scene_feat)

            pred_abs_k_list = []
            for i in range(bs):
                scene = meta[i].scene
                pred_abs_i = denormalize_tensor(pred_norm_k[i], norm_map[scene])
                pred_abs_k_list.append(pred_abs_i)
            pred_abs_k = torch.stack(pred_abs_k_list, dim=0)

            ade_k_each, fde_k_each = ade_fde_per_sample(pred_abs_k, fut_abs)

            if best_ade_each is None:
                best_ade_each = ade_k_each
                best_fde_each = fde_k_each
            else:
                better = ade_k_each < best_ade_each
                best_ade_each = torch.minimum(best_ade_each, ade_k_each)
                best_fde_each = torch.where(better, fde_k_each, best_fde_each)

        total_samples += bs
        ade_sum_1 += ade_1_each.sum().item()
        fde_sum_1 += fde_1_each.sum().item()
        ade_sum_bestk += best_ade_each.sum().item()
        fde_sum_bestk += best_fde_each.sum().item()

        for i in range(bs):
            scene = meta[i].scene
            if scene not in per_scene:
                per_scene[scene] = {
                    "count": 0,
                    "ade_1_sum": 0.0,
                    "fde_1_sum": 0.0,
                    "ade_bestk_sum": 0.0,
                    "fde_bestk_sum": 0.0,
                }

            per_scene[scene]["count"] += 1
            per_scene[scene]["ade_1_sum"] += ade_1_each[i].item()
            per_scene[scene]["fde_1_sum"] += fde_1_each[i].item()
            per_scene[scene]["ade_bestk_sum"] += best_ade_each[i].item()
            per_scene[scene]["fde_bestk_sum"] += best_fde_each[i].item()

    results = {
        "num_samples": float(total_samples),
        "ade_1": ade_sum_1 / total_samples,
        "fde_1": fde_sum_1 / total_samples,
        "ade_bestk": ade_sum_bestk / total_samples,
        "fde_bestk": fde_sum_bestk / total_samples,
    }

    for scene, vals in per_scene.items():
        c = vals["count"]
        results[f"{scene}_count"] = float(c)
        results[f"{scene}_ade_1"] = vals["ade_1_sum"] / c
        results[f"{scene}_fde_1"] = vals["fde_1_sum"] / c
        results[f"{scene}_ade_bestk"] = vals["ade_bestk_sum"] / c
        results[f"{scene}_fde_bestk"] = vals["fde_bestk_sum"] / c

    return results


def format_results(results, k_samples: int) -> str:
    lines = []
    lines.append("=== GAN Evaluation ===")
    lines.append(f"Total samples : {int(results['num_samples'])}")
    lines.append(f"Single-sample ADE : {results['ade_1']:.4f}")
    lines.append(f"Single-sample FDE : {results['fde_1']:.4f}")
    lines.append(f"Best-of-{k_samples} ADE  : {results['ade_bestk']:.4f}")
    lines.append(f"Best-of-{k_samples} FDE  : {results['fde_bestk']:.4f}")
    lines.append("")

    for scene in SCENES:
        if f"{scene}_count" in results:
            lines.append(
                f"{scene:10s} | count={int(results[f'{scene}_count']):5d} | "
                f"ADE1={results[f'{scene}_ade_1']:.4f} | "
                f"FDE1={results[f'{scene}_fde_1']:.4f} | "
                f"ADE@K={results[f'{scene}_ade_bestk']:.4f} | "
                f"FDE@K={results[f'{scene}_fde_bestk']:.4f}"
            )

    return "\n".join(lines)


def main():
    k_samples = 20

    generator, cfg, norm_map, feat_bank = load_checkpoint("runs/gan_best.pt")
    _ = cfg

    loader = build_test_loader(norm_map, batch_size=256)
    results = evaluate_gan(
        generator,
        loader,
        norm_map,
        feat_bank,
        k_samples=k_samples,
    )

    print(format_results(results, k_samples=k_samples))


if __name__ == "__main__":
    main()