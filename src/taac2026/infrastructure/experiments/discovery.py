"""Shared experiment package discovery helpers."""

from __future__ import annotations

from pathlib import Path


def discover_experiment_paths(
    config_root: Path,
    *,
    required_files: tuple[str, ...] = ("__init__.py", "model.py", "ns_groups.json"),
) -> list[str]:
    experiment_paths: list[str] = []
    root = config_root.expanduser().resolve()
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("__"):
            continue
        if all((child / name).exists() for name in required_files):
            experiment_paths.append(child.relative_to(root.parent).as_posix())
    return experiment_paths
