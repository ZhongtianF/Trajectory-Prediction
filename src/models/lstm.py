from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class LSTMConfig:
    input_dim: int = 2
    embedding_dim: int = 32
    hidden_dim: int = 64
    num_layers: int = 1
    pred_len: int = 12
    dropout: float = 0.0


class TrajectoryLSTM(nn.Module):
    """
    Deterministic encoder-decoder LSTM for trajectory prediction.

    Input:
        obs: (B, obs_len, 2)   normalized coordinates
    Output:
        pred: (B, pred_len, 2) normalized future coordinates
    """

    def __init__(self, cfg: LSTMConfig) -> None:
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

        self.decoder = nn.LSTM(
            input_size=cfg.embedding_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )

        self.output_head = nn.Linear(cfg.hidden_dim, 2)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (B, obs_len, 2)

        Returns:
            pred: (B, pred_len, 2)
        """
        if obs.ndim != 3 or obs.size(-1) != 2:
            raise ValueError(f"Expected obs shape (B, T, 2), got {tuple(obs.shape)}")

        emb_obs = self.input_embed(obs)                # (B, obs_len, emb)
        _, (h, c) = self.encoder(emb_obs)              # h/c: (num_layers, B, hidden)

        decoder_input = obs[:, -1:, :]                 # (B, 1, 2), last observed point
        preds = []

        for _ in range(self.cfg.pred_len):
            dec_emb = self.input_embed(decoder_input)  # (B, 1, emb)
            dec_out, (h, c) = self.decoder(dec_emb, (h, c))
            next_step = self.output_head(dec_out)      # (B, 1, 2)
            preds.append(next_step)
            decoder_input = next_step

        return torch.cat(preds, dim=1)                 # (B, pred_len, 2)