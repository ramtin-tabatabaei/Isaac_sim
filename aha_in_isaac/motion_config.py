"""
motion_config.py

Load the per-task arm-motion configuration.

The scene-context report still defines WHERE the arm goes. This config controls
execution policy around those waypoints, such as release timing and grasp tuning.
Each task's config is merged over the shared ``_default`` block.

NOTE: the planner, the gripper open/close widths + friction, and the per-waypoint
step counts now live in ``task_data/physics/<task>.json`` (so each physics file fully
describes the task). ``run_scene`` overlays those onto the dict returned here at load
time, so downstream readers still see them as motion-config keys.

Layout (isolated, one file per task so editing one task never touches another):
    task_data/motion/_default.json   - the shared default config
    task_data/motion/<task>.json     - one file per task that overrides the default
A single legacy monolith (one JSON keyed by task name with a ``_default`` block)
is still accepted for backwards compatibility.
"""

from __future__ import annotations

import json
from pathlib import Path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def load_motion_config(path: Path, task_name: str) -> dict:
    """Return the task's motion config, merged over the shared default block."""
    if path.is_dir():
        task_config = _read_json(path / f"{task_name}.json")
        config = dict(_read_json(path / "_default.json")) if task_config.get("_inherit_defaults", True) else {}
        config.update(task_config)
        config.pop("_inherit_defaults", None)
        return config
    # Legacy single-file monolith keyed by task name.
    data = _read_json(path)
    if not path.is_file():
        print(f"[WARN]: Motion config {path} not found; using built-in defaults.")
    task_config = data.get(task_name, {})
    config = dict(data.get("_default", {})) if task_config.get("_inherit_defaults", True) else {}
    config.update(task_config)
    config.pop("_inherit_defaults", None)
    return config
