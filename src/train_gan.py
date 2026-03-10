from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import SCENES, ensure_dirs
from .dataset import (
    ConcatSceneDataset,
    build_scene_datasets,
    collate_trajectory_batch,
)
from .gan_models import (
    GANConfig,
    SceneEncoder,
    TrajectoryDiscriminator,
    TrajectoryGenerator,
    load_scene_image_tensor,
    sample_noise,
)
from .scene_utils import find_closest_scene_image


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_loaders(batch_size: int = 128):
    train_datasets, norm_map = build_scene_datasets(split="train")
    val_datasets, _ = build_scene_datasets(
        split="val",
        train_norm_by_scene=norm_map,
    )

    train_ds = ConcatSceneDataset(train_datasets)
    val_ds = ConcatSceneDataset(val_datasets)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_trajectory_batch,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_trajectory_batch,
    )

    return train_loader, val_loader, norm_map


@torch.no_grad()
def build_scene_feature_bank(scene_encoder: SceneEncoder, scene_dim: int) -> Dict[str, torch.Tensor]:
    """
    Build one global scene feature per scene using one representative image.
    """
    scene_encoder.eval()

    feat_bank: Dict[str, torch.Tensor] = {}

    for scene in SCENES:
        img_path = find_closest_scene_image(scene, frame_num=0)
        img_tensor = load_scene_image_tensor(img_path, device=DEVICE)
        feat = scene_encoder(img_tensor)[0]   # (scene_dim,)
        feat_bank[scene] = feat.detach()

    return feat_bank


def get_scene_features_for_batch(meta, feat_bank: Dict[str, torch.Tensor]) -> torch.Tensor:
    feats = [feat_bank[m.scene] for m in meta]
    return torch.stack(feats, dim=0).to(DEVICE)


def discriminator_loss_fn(
    discriminator: TrajectoryDiscriminator,
    obs: torch.Tensor,
    fut_real: torch.Tensor,
    fut_fake: torch.Tensor,
    bce_logits: nn.Module,
):
    real_traj = torch.cat([obs, fut_real], dim=1)
    fake_traj = torch.cat([obs, fut_fake.detach()], dim=1)

    real_logits = discriminator(real_traj)
    fake_logits = discriminator(fake_traj)

    real_targets = torch.ones_like(real_logits) * 0.9
    fake_targets = torch.zeros_like(fake_logits) + 0.1

    loss_real = bce_logits(real_logits, real_targets)
    loss_fake = bce_logits(fake_logits, fake_targets)

    return 0.5 * (loss_real + loss_fake)


def generator_loss_fn(
    discriminator: TrajectoryDiscriminator,
    obs: torch.Tensor,
    fut_real: torch.Tensor,
    fut_fake: torch.Tensor,
    bce_logits: nn.Module,
    l1_weight: float = 10.0,
):
    fake_traj = torch.cat([obs, fut_fake], dim=1)
    fake_logits = discriminator(fake_traj)

    adv_targets = torch.ones_like(fake_logits) * 0.9
    adv_loss = bce_logits(fake_logits, adv_targets)

    recon_loss = torch.mean(torch.abs(fut_fake - fut_real))

    total = adv_loss + l1_weight * recon_loss
    return total, adv_loss.item(), recon_loss.item()


@torch.no_grad()
def evaluate_gan(
    generator: TrajectoryGenerator,
    discriminator: TrajectoryDiscriminator,
    val_loader: DataLoader,
    feat_bank: Dict[str, torch.Tensor],
    cfg: GANConfig,
    bce_logits: nn.Module,
):
    generator.eval()
    discriminator.eval()

    total_g = 0.0
    total_d = 0.0
    total_samples = 0

    for batch in val_loader:
        obs = batch["obs"].to(DEVICE)
        fut = batch["fut"].to(DEVICE)
        meta = batch["meta"]

        bs = obs.size(0)

        noise = sample_noise(bs, cfg.noise_dim, DEVICE)
        scene_feat = get_scene_features_for_batch(meta, feat_bank)

        fut_fake = generator(obs, noise, scene_feat=scene_feat)

        d_loss = discriminator_loss_fn(discriminator, obs, fut, fut_fake, bce_logits)

        g_loss, _, _ = generator_loss_fn(discriminator, obs, fut, fut_fake, bce_logits)

        total_d += d_loss.item() * bs
        total_g += g_loss.item() * bs
        total_samples += bs

    return total_g / total_samples, total_d / total_samples


def train():
    ensure_dirs()

    train_loader, val_loader, norm_map = build_loaders(batch_size=128)

    cfg = GANConfig()

    generator = TrajectoryGenerator(cfg).to(DEVICE)
    discriminator = TrajectoryDiscriminator(cfg).to(DEVICE)
    scene_encoder = SceneEncoder(scene_dim=cfg.scene_dim).to(DEVICE)

    feat_bank = build_scene_feature_bank(scene_encoder, cfg.scene_dim)

    opt_g = torch.optim.Adam(generator.parameters(), lr=1e-3, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=1e-3, betas=(0.5, 0.999))

    bce_logits = nn.BCEWithLogitsLoss()

    epochs = 20
    best_val_g = float("inf")

    for epoch in range(epochs):
        generator.train()
        discriminator.train()

        total_g = 0.0
        total_d = 0.0
        total_samples = 0

        for batch in train_loader:
            obs = batch["obs"].to(DEVICE)
            fut = batch["fut"].to(DEVICE)
            meta = batch["meta"]

            bs = obs.size(0)
            scene_feat = get_scene_features_for_batch(meta, feat_bank)

            # -------------------------
            # 1) Update Discriminator
            # -------------------------
            noise = sample_noise(bs, cfg.noise_dim, DEVICE)
            fut_fake = generator(obs, noise, scene_feat=scene_feat)

            opt_d.zero_grad()
            d_loss = discriminator_loss_fn(discriminator, obs, fut, fut_fake, bce_logits)
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
            opt_d.step()

            # -------------------------
            # 2) Update Generator
            # -------------------------
            noise = sample_noise(bs, cfg.noise_dim, DEVICE)
            fut_fake = generator(obs, noise, scene_feat=scene_feat)

            opt_g.zero_grad()
            g_loss, adv_value, recon_value = generator_loss_fn(
                discriminator,
                obs,
                fut,
                fut_fake,
                bce_logits,
                l1_weight=10.0,
            )
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            opt_g.step()

            total_d += d_loss.item() * bs
            total_g += g_loss.item() * bs
            total_samples += bs

        train_g = total_g / total_samples
        train_d = total_d / total_samples

        val_g, val_d = evaluate_gan(
            generator,
            discriminator,
            val_loader,
            feat_bank,
            cfg,
            bce_logits,
        )

        print(
            f"Epoch {epoch+1:02d}/{epochs} | "
            f"train_G={train_g:.6f} | train_D={train_d:.6f} | "
            f"val_G={val_g:.6f} | val_D={val_d:.6f}"
        )

        if val_g < best_val_g:
            best_val_g = val_g

            torch.save(
                {
                    "generator_state": generator.state_dict(),
                    "discriminator_state": discriminator.state_dict(),
                    "config": vars(cfg),
                    "norm": norm_map,
                    "scene_feature_bank": {k: v.cpu() for k, v in feat_bank.items()},
                },
                "runs/gan_best.pt",
            )
            print("Saved best GAN checkpoint to runs/gan_best.pt")


if __name__ == "__main__":
    train()