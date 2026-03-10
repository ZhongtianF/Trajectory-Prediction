from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import OBS_LEN, PRED_LEN, SCENES, get_scene_dir


@dataclass
class SampleMeta:
    scene: str
    ped_id: int
    start_frame: int
    end_frame: int
    data_id: int


@dataclass
class NormStats:
    mean: np.ndarray
    std: np.ndarray


def _find_first_file(scene_dir: Path, patterns: Sequence[str]) -> Optional[Path]:
    for pattern in patterns:
        matches = sorted(scene_dir.rglob(pattern))
        if matches:
            return matches[0]
    return None


def find_db_file(scene: str) -> Path:
    scene_dir = get_scene_dir(scene)
    db_file = _find_first_file(scene_dir, ["*.db", "*.sqlite", "*.sqlite3"])
    if db_file is None:
        raise FileNotFoundError(f"No database file found under scene directory: {scene_dir}")
    return db_file


def list_tables(db_path: Path) -> List[str]:
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        return [r[0] for r in cur.fetchall()]
    finally:
        conn.close()


def infer_main_table(db_path: Path) -> str:
    tables = list_tables(db_path)
    if not tables:
        raise ValueError(f"No tables found in database: {db_path}")

    preferred = [
        "dataset_t_length_20delta_coordinates",
        "dataset_T_length_20delta_coordinates",
        "pos_data",
        "trajectory",
        "trajectories",
        "tracks",
        "data",
    ]
    lower_map = {t.lower(): t for t in tables}
    for p in preferred:
        if p.lower() in lower_map:
            return lower_map[p.lower()]

    return tables[0]


def read_table_as_array(db_path: Path, table_name: str) -> np.ndarray:
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute(f"SELECT * FROM {table_name}")
        rows = cur.fetchall()
        if not rows:
            raise ValueError(f"Table '{table_name}' in {db_path} is empty.")
        return np.asarray(rows)
    finally:
        conn.close()


def get_table_columns(db_path: Path, table_name: str) -> List[str]:
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute(f"PRAGMA table_info({table_name})")
        info = cur.fetchall()
        return [row[1] for row in info]
    finally:
        conn.close()


def _lower_index_map(columns: Sequence[str]) -> Dict[str, int]:
    return {c.lower(): i for i, c in enumerate(columns)}


def build_sequences_from_precomputed_table(
    rows: np.ndarray,
    columns: Sequence[str],
    obs_len: int = OBS_LEN,
    pred_len: int = PRED_LEN,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Expected DB columns include:
      pos_x, pos_y, ped_id, frame_num, data_id

    Each data_id corresponds to one full sequence of length obs_len + pred_len.
    """
    total_len = obs_len + pred_len
    colmap = _lower_index_map(columns)

    required = ["pos_x", "pos_y", "ped_id", "frame_num", "data_id"]
    missing = [c for c in required if c not in colmap]
    if missing:
        raise ValueError(
            f"Precomputed table is missing required columns: {missing}. "
            f"Available columns: {list(columns)}"
        )

    x_idx = colmap["pos_x"]
    y_idx = colmap["pos_y"]
    ped_idx = colmap["ped_id"]
    frame_idx = colmap["frame_num"]
    data_idx = colmap["data_id"]

    data_ids = np.unique(rows[:, data_idx]).astype(int)

    obs_list: List[np.ndarray] = []
    fut_list: List[np.ndarray] = []
    meta_list: List[Dict[str, Any]] = []

    for did in data_ids:
        seq_rows = rows[rows[:, data_idx] == did]

        if len(seq_rows) != total_len:
            continue

        seq_rows = seq_rows[np.argsort(seq_rows[:, frame_idx].astype(np.int64))]

        coords = seq_rows[:, [x_idx, y_idx]].astype(np.float32)
        ped_values = seq_rows[:, ped_idx].astype(np.int64)
        frame_values = seq_rows[:, frame_idx].astype(np.int64)

        obs = coords[:obs_len]
        fut = coords[obs_len:]

        obs_list.append(obs)
        fut_list.append(fut)
        meta_list.append(
            {
                "ped_id": int(ped_values[0]),
                "start_frame": int(frame_values[0]),
                "end_frame": int(frame_values[-1]),
                "data_id": int(did),
            }
        )

    if not obs_list:
        raise ValueError("No valid precomputed trajectory sequences found in DB table.")

    return (
        np.stack(obs_list, axis=0),
        np.stack(fut_list, axis=0),
        meta_list,
    )


def build_sequences_from_raw_rows(
    rows: np.ndarray,
    columns: Sequence[str],
    obs_len: int = OBS_LEN,
    pred_len: int = PRED_LEN,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Fallback for raw table format:
      [frame, ped_id, x, y] or similarly named columns
    """
    total_len = obs_len + pred_len
    colmap = _lower_index_map(columns)

    candidates = [
        ("frame", "ped_id", "x", "y"),
        ("frame_num", "ped_id", "pos_x", "pos_y"),
    ]

    chosen = None
    for c in candidates:
        if all(name in colmap for name in c):
            chosen = c
            break

    if chosen is None:
        if rows.ndim != 2 or rows.shape[1] < 4:
            raise ValueError(
                f"Could not infer raw trajectory format from columns={columns}, shape={rows.shape}"
            )
        arr = rows[:, :4].astype(np.float32)
        frame_idx, ped_idx, x_idx, y_idx = 0, 1, 2, 3
    else:
        frame_idx = colmap[chosen[0]]
        ped_idx = colmap[chosen[1]]
        x_idx = colmap[chosen[2]]
        y_idx = colmap[chosen[3]]
        arr = rows

    ped_ids = np.unique(arr[:, ped_idx]).astype(int)

    obs_list: List[np.ndarray] = []
    fut_list: List[np.ndarray] = []
    meta_list: List[Dict[str, Any]] = []

    for ped_id in ped_ids:
        ped_rows = arr[arr[:, ped_idx] == ped_id]
        ped_rows = ped_rows[np.argsort(ped_rows[:, frame_idx])]

        frames = ped_rows[:, frame_idx].astype(int)
        coords = ped_rows[:, [x_idx, y_idx]].astype(np.float32)

        if len(frames) < total_len:
            continue

        for i in range(0, len(frames) - total_len + 1):
            frame_window = frames[i:i + total_len]
            step = frame_window[1] - frame_window[0] if len(frame_window) > 1 else 1
            expected = np.arange(frame_window[0], frame_window[0] + total_len * step, step)

            if not np.array_equal(frame_window, expected):
                continue

            coord_window = coords[i:i + total_len]
            obs_list.append(coord_window[:obs_len])
            fut_list.append(coord_window[obs_len:])
            meta_list.append(
                {
                    "ped_id": int(ped_id),
                    "start_frame": int(frame_window[0]),
                    "end_frame": int(frame_window[-1]),
                    "data_id": -1,
                }
            )

    if not obs_list:
        raise ValueError("No valid raw trajectory sequences could be built from rows.")

    return (
        np.stack(obs_list, axis=0),
        np.stack(fut_list, axis=0),
        meta_list,
    )


def build_sequences(
    rows: np.ndarray,
    columns: Sequence[str],
    obs_len: int = OBS_LEN,
    pred_len: int = PRED_LEN,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    colmap = _lower_index_map(columns)

    if all(k in colmap for k in ["pos_x", "pos_y", "ped_id", "frame_num", "data_id"]):
        return build_sequences_from_precomputed_table(
            rows, columns, obs_len=obs_len, pred_len=pred_len
        )

    return build_sequences_from_raw_rows(
        rows, columns, obs_len=obs_len, pred_len=pred_len
    )


def split_indices(
    n: int,
    split: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> np.ndarray:
    if split not in {"train", "val", "test", "all"}:
        raise ValueError(f"Invalid split: {split}")

    idx = np.arange(n)
    if split == "all":
        return idx

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    if split == "train":
        return train_idx
    if split == "val":
        return val_idx
    return test_idx


def compute_norm_stats(obs: np.ndarray) -> NormStats:
    flat = obs.reshape(-1, 2).astype(np.float32)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return NormStats(mean=mean, std=std)


def apply_normalization(arr: np.ndarray, norm: NormStats) -> np.ndarray:
    return (arr - norm.mean) / norm.std


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        scene: str,
        split: str = "train",
        obs_len: int = OBS_LEN,
        pred_len: int = PRED_LEN,
        norm_stats: Optional[NormStats] = None,
    ) -> None:
        super().__init__()

        self.scene = scene
        self.split = split
        self.obs_len = obs_len
        self.pred_len = pred_len

        db_path = find_db_file(scene)
        table_name = infer_main_table(db_path)
        columns = get_table_columns(db_path, table_name)
        raw_table = read_table_as_array(db_path, table_name)

        obs_all, fut_all, meta_all = build_sequences(
            raw_table,
            columns,
            obs_len=obs_len,
            pred_len=pred_len,
        )

        idx = split_indices(len(obs_all), split=split)
        self.obs = obs_all[idx].copy()
        self.fut = fut_all[idx].copy()
        self.meta_raw = [meta_all[i] for i in idx.tolist()]

        if norm_stats is None:
            self.norm_stats = compute_norm_stats(self.obs)
        else:
            self.norm_stats = norm_stats

        self.obs_norm = apply_normalization(self.obs, self.norm_stats).astype(np.float32)
        self.fut_norm = apply_normalization(self.fut, self.norm_stats).astype(np.float32)

    def __len__(self) -> int:
        return len(self.obs)

    def __getitem__(self, index: int):
        meta_info = self.meta_raw[index]
        meta = SampleMeta(
            scene=self.scene,
            ped_id=meta_info["ped_id"],
            start_frame=meta_info["start_frame"],
            end_frame=meta_info["end_frame"],
            data_id=meta_info["data_id"],
        )

        return {
            "obs": torch.from_numpy(self.obs_norm[index]).float(),
            "fut": torch.from_numpy(self.fut_norm[index]).float(),
            "obs_abs": torch.from_numpy(self.obs[index]).float(),
            "fut_abs": torch.from_numpy(self.fut[index]).float(),
            "meta": meta,
        }


class ConcatSceneDataset(Dataset):
    def __init__(self, datasets: Sequence[TrajectoryDataset]) -> None:
        self.datasets = list(datasets)
        self.cum_sizes = []
        total = 0
        for ds in self.datasets:
            total += len(ds)
            self.cum_sizes.append(total)

    def __len__(self) -> int:
        return 0 if not self.cum_sizes else self.cum_sizes[-1]

    def __getitem__(self, index: int):
        if index < 0:
            raise IndexError("Negative indices are not supported.")
        for ds_idx, cum_size in enumerate(self.cum_sizes):
            if index < cum_size:
                prev = 0 if ds_idx == 0 else self.cum_sizes[ds_idx - 1]
                return self.datasets[ds_idx][index - prev]
        raise IndexError("Index out of range.")


def collate_trajectory_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "obs": torch.stack([item["obs"] for item in batch], dim=0),
        "fut": torch.stack([item["fut"] for item in batch], dim=0),
        "obs_abs": torch.stack([item["obs_abs"] for item in batch], dim=0),
        "fut_abs": torch.stack([item["fut_abs"] for item in batch], dim=0),
        "meta": [item["meta"] for item in batch],
    }


def build_scene_datasets(
    split: str,
    scenes: Sequence[str] = SCENES,
    train_norm_by_scene: Optional[Dict[str, NormStats]] = None,
) -> Tuple[List[TrajectoryDataset], Dict[str, NormStats]]:
    datasets: List[TrajectoryDataset] = []
    norm_map: Dict[str, NormStats] = {}

    for scene in scenes:
        if split == "train":
            ds = TrajectoryDataset(scene=scene, split="train")
            norm_map[scene] = ds.norm_stats
        else:
            if train_norm_by_scene is None or scene not in train_norm_by_scene:
                raise ValueError(f"Missing train normalization stats for scene '{scene}'.")
            ds = TrajectoryDataset(
                scene=scene,
                split=split,
                norm_stats=train_norm_by_scene[scene],
            )
            norm_map[scene] = train_norm_by_scene[scene]

        datasets.append(ds)

    return datasets, norm_map


def denormalize_tensor(x: torch.Tensor, norm: NormStats) -> torch.Tensor:
    mean = torch.tensor(norm.mean, dtype=x.dtype, device=x.device)
    std = torch.tensor(norm.std, dtype=x.dtype, device=x.device)
    return x * std + mean


def inspect_scene(scene: str) -> None:
    db_path = find_db_file(scene)
    tables = list_tables(db_path)
    main_table = infer_main_table(db_path)
    columns = get_table_columns(db_path, main_table)
    arr = read_table_as_array(db_path, main_table)

    print(f"Scene      : {scene}")
    print(f"DB path    : {db_path}")
    print(f"Tables     : {tables}")
    print(f"Main table : {main_table}")
    print(f"Columns    : {columns}")
    print(f"Raw shape  : {arr.shape}")
    print("First 5 rows:")
    print(arr[:5])

    obs_all, fut_all, _ = build_sequences(arr, columns)
    print("Built obs shape:", obs_all.shape)
    print("Built fut shape:", fut_all.shape)


if __name__ == "__main__":
    for s in SCENES:
        print("=" * 80)
        inspect_scene(s)
        ds = TrajectoryDataset(scene=s, split="train")
        print(f"Train samples: {len(ds)}")
        print(f"Norm mean: {ds.norm_stats.mean}")
        print(f"Norm std : {ds.norm_stats.std}")
        sample = ds[0]
        print("obs shape:", sample["obs"].shape)
        print("fut shape:", sample["fut"].shape)
        print("meta     :", sample["meta"])