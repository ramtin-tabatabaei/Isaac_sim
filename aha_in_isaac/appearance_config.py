"""
appearance_config.py

Load the per-task object-appearance configuration (textures / colours / visibility).

Layout (isolated, one file per task so editing one task never touches another):
    task_data/appearance/_shared.json - shared blocks: _defaults, _scene (floor/table), _README
    task_data/appearance/<task>.json  - one file per task
A single legacy monolith (one JSON keyed by task name plus _defaults/_scene) is
still accepted for backwards compatibility.

Returns the same dict shape SceneBuilder consumes: the shared ``_defaults`` and
``_scene`` blocks plus the requested task's block under its task name.
"""

from __future__ import annotations

import json
from pathlib import Path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def load_appearance_config(path: Path, task_name: str) -> dict:
    """Return {_defaults, _scene, _README?, <task_name>: <task block>} for one task."""
    if path.is_dir():
        shared = _read_json(path / "_shared.json")  # _README, _defaults, _scene
        task = _read_json(path / f"{task_name}.json")
        return {**shared, task_name: task}
    # Legacy single-file monolith keyed by task name.
    return _read_json(path)
