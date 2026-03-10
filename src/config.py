from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    project_root: Path
    data_root: Path
    runs_root: Path


# 你已经确认好的数据路径
DATA_ROOT = Path("/Users/hblffzt/Documents/T2IA_TrajPred/data/data_trajpred")

# 自动推断项目根目录：
# /Users/hblffzt/Documents/T2IA_TrajPred/data/data_trajpred
# -> parents[0] = .../data
# -> parents[1] = .../T2IA_TrajPred
PROJECT_ROOT = DATA_ROOT.parents[1]

RUNS_ROOT = PROJECT_ROOT / "runs"

PATHS = Paths(
    project_root=PROJECT_ROOT,
    data_root=DATA_ROOT,
    runs_root=RUNS_ROOT,
)

SCENES = ["eth", "hotel", "university", "zara_01", "zara_02"]

RAW_SCENE = "raw"

OBS_LEN = 8
PRED_LEN = 12

DEFAULT_IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]


def ensure_dirs() -> None:
    PATHS.runs_root.mkdir(parents=True, exist_ok=True)


def get_scene_dir(scene: str) -> Path:
    scene_dir = PATHS.data_root / scene
    if not scene_dir.exists():
        raise FileNotFoundError(f"Scene directory not found: {scene_dir}")
    return scene_dir


def validate_data_layout() -> None:
    if not PATHS.data_root.exists():
        raise FileNotFoundError(f"Data root not found: {PATHS.data_root}")

    missing = []
    for scene in SCENES:
        if not (PATHS.data_root / scene).exists():
            missing.append(scene)

    if missing:
        raise FileNotFoundError(
            f"Missing scene folders under {PATHS.data_root}: {missing}"
        )


if __name__ == "__main__":
    print("PROJECT_ROOT =", PATHS.project_root)
    print("DATA_ROOT    =", PATHS.data_root)
    print("RUNS_ROOT    =", PATHS.runs_root)
    print("SCENES       =", SCENES)

    ensure_dirs()
    validate_data_layout()
    print("Config check passed.")