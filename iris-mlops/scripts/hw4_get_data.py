from __future__ import annotations

import pathlib

import hydra
from omegaconf import DictConfig
from sklearn import datasets


def _project_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


@hydra.main(version_base=None, config_path="../configs/hw4", config_name="config")
def main(cfg: DictConfig) -> None:
    project_root = _project_root()
    raw_dir = project_root / pathlib.Path(str(cfg.paths.raw))
    raw_dir.mkdir(parents=True, exist_ok=True)

    out_path = raw_dir / str(cfg.data.filename)
    if out_path.exists() and not bool(cfg.force):
        print(f"[get_data] cache hit -> skip: {out_path}")
        return

    iris = datasets.load_iris(as_frame=True)
    df = iris.frame
    df.to_csv(out_path, index=False)
    print(f"[get_data] wrote: {out_path} (rows={len(df)}, cols={df.shape[1]})")


if __name__ == "__main__":
    main()
