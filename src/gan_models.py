from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

from .config import OBS_LEN, PRED_LEN


@dataclass
class GANConfig:
    obs_len: int = OBS_LEN
    pred_len: int = PRED_LEN
    input_dim: int = 2
    embedding_dim: int = 32
    hidden_dim: int = 64
    noise_dim: int = 16
    scene_dim: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    use_scene: bool = True


class SceneEncoder(nn.Module):
    """
    Frozen ResNet18 encoder for scene images.
    Output: (B, scene_dim)
    """

    def __init__(self, scene_dim: int = 64) -> None:
        super().__init__()

        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.proj = nn.Linear(feat_dim, scene_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        feat = self.proj(feat)
        return feat


def build_scene_transform():
    weights = models.ResNet18_Weights.DEFAULT
    return weights.transforms()


def load_scene_image_tensor(image_path: Path, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Load one scene image and convert it into tensor shape (1, C, H, W)
    """
    transform = build_scene_transform()
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0)
    if device is not None:
        x = x.to(device)
    return x


class TrajectoryGenerator(nn.Module):
    """
    Generator:
      obs trajectory -> encoder LSTM
      + noise
      + optional scene feature
      -> autoregressive decoder
    Output:
      future trajectory in normalized coordinates, shape (B, pred_len, 2)
    """

    def __init__(self, cfg: GANConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.input_embed = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.embedding_dim),
            nn.ReLU(),
        )

        self.encoder = nn.LSTM(
            input_size=cfg.embedding_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )

        cond_dim = cfg.hidden_dim + cfg.noise_dim + (cfg.scene_dim if cfg.use_scene else 0)

        self.cond_to_h = nn.Linear(cond_dim, cfg.hidden_dim)
        self.cond_to_c = nn.Linear(cond_dim, cfg.hidden_dim)

        self.decoder_cell = nn.LSTMCell(cfg.embedding_dim, cfg.hidden_dim)
        self.output_head = nn.Linear(cfg.hidden_dim, 2)

    def forward(
        self,
        obs: torch.Tensor,
        noise: torch.Tensor,
        scene_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        obs: (B, obs_len, 2)
        noise: (B, noise_dim)
        scene_feat: (B, scene_dim) or None
        """
        if obs.ndim != 3 or obs.size(-1) != 2:
            raise ValueError(f"Expected obs shape (B,T,2), got {tuple(obs.shape)}")

        if noise.ndim != 2:
            raise ValueError(f"Expected noise shape (B,noise_dim), got {tuple(noise.shape)}")

        emb_obs = self.input_embed(obs)
        _, (h_enc, c_enc) = self.encoder(emb_obs)

        h_last = h_enc[-1]
        cond_parts = [h_last, noise]

        if self.cfg.use_scene:
            if scene_feat is None:
                raise ValueError("scene_feat is required when use_scene=True")
            cond_parts.append(scene_feat)

        cond = torch.cat(cond_parts, dim=1)

        h = self.cond_to_h(cond)
        c = self.cond_to_c(cond)

        decoder_input = obs[:, -1, :]  # (B, 2)
        preds = []

        for _ in range(self.cfg.pred_len):
            dec_emb = self.input_embed(decoder_input)
            h, c = self.decoder_cell(dec_emb, (h, c))
            next_step = self.output_head(h)
            preds.append(next_step.unsqueeze(1))
            decoder_input = next_step

        return torch.cat(preds, dim=1)


class TrajectoryDiscriminator(nn.Module):
    """
    Discriminator:
      full trajectory (obs + fut) -> real/fake score
    """

    def __init__(self, cfg: GANConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.input_embed = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.embedding_dim),
            nn.ReLU(),
        )

        self.encoder = nn.LSTM(
            input_size=cfg.embedding_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, 1),
        )

    def forward(self, traj: torch.Tensor) -> torch.Tensor:
        """
        traj: (B, obs_len + pred_len, 2)
        returns logits: (B, 1)
        """
        if traj.ndim != 3 or traj.size(-1) != 2:
            raise ValueError(f"Expected traj shape (B,T,2), got {tuple(traj.shape)}")

        emb = self.input_embed(traj)
        _, (h, _) = self.encoder(emb)
        h_last = h[-1]
        logits = self.head(h_last)
        return logits


def sample_noise(batch_size: int, noise_dim: int, device: torch.device) -> torch.Tensor:
    return torch.randn(batch_size, noise_dim, device=device)