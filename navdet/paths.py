"""Small path helpers so every script can be run from anywhere."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_path(p: str | Path) -> Path:
    """Relative paths are taken relative to the repo root, absolute ones are left alone."""
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def resolved_data_yaml(data_yaml: str | Path = "configs/data.yaml") -> str:
    """
    Ultralytics treats a relative `path:` as relative to its own global datasets
    folder, not to the yaml. So we write a copy with an absolute path and hand
    that to YOLO instead.
    """
    src = repo_path(data_yaml)
    cfg = yaml.safe_load(src.read_text())
    cfg["path"] = str(repo_path(cfg["path"]).resolve())
    out = REPO_ROOT / "runs" / f"{src.stem}_resolved.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return str(out)
