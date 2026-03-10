from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import ensure_dirs
from .dataset import (
    ConcatSceneDataset,
    build_scene_datasets,
    collate_trajectory_batch,
)
from .models import LSTMConfig, TrajectoryLSTM


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


def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            obs = batch["obs"].to(DEVICE)
            fut = batch["fut"].to(DEVICE)

            pred = model(obs)

            loss = criterion(pred, fut)

            bs = obs.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

    return total_loss / total_samples


def train():

    ensure_dirs()

    train_loader, val_loader, norm_map = build_loaders(batch_size=128)

    cfg = LSTMConfig()

    model = TrajectoryLSTM(cfg).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    criterion = nn.MSELoss()

    epochs = 20

    best_val = float("inf")

    for epoch in range(epochs):

        model.train()

        total_loss = 0
        total_samples = 0

        for batch in train_loader:

            obs = batch["obs"].to(DEVICE)
            fut = batch["fut"].to(DEVICE)

            optimizer.zero_grad()

            pred = model(obs)

            loss = criterion(pred, fut)

            loss.backward()

            optimizer.step()

            bs = obs.size(0)

            total_loss += loss.item() * bs
            total_samples += bs

        train_loss = total_loss / total_samples

        val_loss = evaluate(model, val_loader, criterion)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f}"
        )

        if val_loss < best_val:

            best_val = val_loss

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": vars(cfg),
                    "norm": norm_map,
                },
                "runs/lstm_best.pt",
            )

            print("Model saved to runs/lstm_best.pt")


if __name__ == "__main__":
    train()