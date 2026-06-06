"""
motion_config.py

Load the per-task arm-motion configuration (``task_motion_config.json``).

The scene-context report still defines WHERE the arm goes. This config controls
execution policy around those waypoints, such as waypoint durations, default
planner, gripper close width, and release timing. Each task block is merged over
the file's ``_default`` block.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_motion_config(path: Path, task_name: str) -> dict:
    """Return the task's motion config, merged over the file's '_default' block."""
    data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    config = dict(data.get("_default", {}))
    config.update(data.get(task_name, {}))
    if not path.is_file():
        print(f"[WARN]: Motion config {path} not found; using built-in defaults.")
    return config
