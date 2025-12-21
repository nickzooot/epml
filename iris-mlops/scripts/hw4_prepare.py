from __future__ import annotations

import json
import pathlib

import hydra
import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split


def _project_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def _validate(cfg: DictConfig) -> None:
    test_size = float(cfg.data.test_size)
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"data.test_size must be in (0, 1), got: {test_size}")


@hydra.main(version_base=None, config_path="../configs/hw4", config_name="config")
def main(cfg: DictConfig) -> None:
    _validate(cfg)
    project_root = _project_root()

    raw_path = project_root / pathlib.Path(str(cfg.paths.raw)) / str(cfg.data.filename)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset is missing: {raw_path} (run get_data first)")

    processed_dir = project_root / pathlib.Path(str(cfg.paths.processed))
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"
    meta_path = processed_dir / "meta.json"

    if train_path.exists() and test_path.exists() and meta_path.exists() and not bool(cfg.force):
        print(f"[prepare] cache hit -> skip: {processed_dir}")
        return

    df = pd.read_csv(raw_path)
    target_col = str(cfg.data.target_col)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {list(df.columns)}")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(cfg.data.test_size),
        random_state=int(cfg.seed),
        stratify=y,
    )

    train_df = X_train.copy()
    train_df[target_col] = y_train
    test_df = X_test.copy()
    test_df[target_col] = y_test

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    meta = {
        "seed": int(cfg.seed),
        "test_size": float(cfg.data.test_size),
        "target_col": target_col,
        "feature_cols": [str(c) for c in X.columns],
        "n_train": int(train_df.shape[0]),
        "n_test": int(test_df.shape[0]),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        "[prepare] wrote: "
        f"{train_path.name} (rows={train_df.shape[0]}), {test_path.name} (rows={test_df.shape[0]})"
    )


if __name__ == "__main__":
    main()
