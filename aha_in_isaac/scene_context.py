"""
scene_context.py

Parse an exported AHA ``*.scene_context.md`` report and derive everything the
rest of the pipeline needs to place the scene: object poses, the dining-table
top height, the task-root transform, and the robot base pose.

This module is pure Python (no Isaac Lab / USD imports), so it can run *before*
the simulator is launched. The Isaac-dependent spawning lives in
``scene_builder.py``; this file only reads and reasons about the report data.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

USD_EXTENSIONS = (".usd", ".usdc", ".usda")


# ----------------------------------------------------------------------
# Report loading + small quaternion / pose helpers (shared with the builder).
# ----------------------------------------------------------------------
def load_report(path: Path) -> dict:
    """Pull the fenced ```json``` scene block out of a scene-context .md file."""
    text = path.read_text(encoding="utf-8")
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if match is None:
        raise RuntimeError(f"No fenced JSON scene data found in {path}")
    return json.loads(match.group(1))


def pose_from_location(location: dict | None):
    """Return ((x, y, z), (w, x, y, z)) from a report location block."""
    if location is None:
        return (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)
    pos = tuple(float(v) for v in location["position_xyz_m"])
    qx, qy, qz, qw = (float(v) for v in location["quaternion_xyzw"])
    return pos, (qw, qx, qy, qz)


def pose_from_world_location(entry: dict):
    return pose_from_location(entry["world_location"])


def _qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _qinv(q):
    w, x, y, z = q
    norm = w * w + x * x + y * y + z * z
    return (w / norm, -x / norm, -y / norm, -z / norm)


def _qapply(q, v):
    return _qmul(_qmul(q, (0.0, *v)), _qinv(q))[1:]


def _subtract_pose(world_pos, world_quat, local_pos, local_quat):
    parent_quat = _qmul(world_quat, _qinv(local_quat))
    rotated_local_pos = _qapply(parent_quat, local_pos)
    parent_pos = tuple(world_pos[i] - rotated_local_pos[i] for i in range(3))
    return parent_pos, parent_quat


# ----------------------------------------------------------------------
# Derived scene facts.
# ----------------------------------------------------------------------
def object_entries(report: dict) -> dict[str, dict]:
    return {entry["name"]: entry for entry in report["objects"]}


def task_name_from(report: dict, scene_context: Path) -> str:
    if report.get("task_name"):
        return str(report["task_name"])
    return scene_context.name.removesuffix(".scene_context.md")


def task_root_object(objects: dict[str, dict], task_name: str) -> dict:
    root_candidates = [
        entry
        for entry in objects.values()
        if entry.get("parent") == task_name or entry["name"].endswith("_root") or "boundary_root" in entry["name"]
    ]
    return root_candidates[0] if root_candidates else next(iter(objects.values()))


def sampled_task_root_pose(objects: dict[str, dict], task_name: str):
    root_entry = task_root_object(objects, task_name)
    root_world_pos, root_world_quat = pose_from_world_location(root_entry)
    root_local_pos, root_local_quat = pose_from_location(root_entry.get("task_root_local_location"))
    return _subtract_pose(root_world_pos, root_world_quat, root_local_pos, root_local_quat)


def table_top_z(objects: dict[str, dict], task_name: str, table_top_object: str) -> float:
    if table_top_object != "auto":
        if table_top_object not in objects:
            raise KeyError(f"Table top object '{table_top_object}' was not found in the scene report.")
        return float(objects[table_top_object]["world_location"]["position_xyz_m"][2])
    return float(sampled_task_root_pose(objects, task_name)[0][2])


def robot_base_pose(report: dict, table_top_z_value: float):
    """World-frame Franka base pose from the report's ``robot_base`` block.

    The report's ``robot_base`` is captured in the RLBench/CoppeliaSim frame,
    whose orientation convention tips an Isaac Franka over. We therefore keep the
    horizontal (x, y) placement from the report but rest the base upright on the
    detected table top so the arm is mounted and can actually reach the objects.
    """
    robot_base = report.get("robot_base")
    if not robot_base:
        return (-0.35, 0.0, table_top_z_value), (1.0, 0.0, 0.0, 0.0)
    base_pos, _ = pose_from_location(robot_base.get("location"))
    return (base_pos[0], base_pos[1], table_top_z_value), (1.0, 0.0, 0.0, 0.0)


# ----------------------------------------------------------------------
# USD file resolution for each scene object.
# ----------------------------------------------------------------------
def _usd_candidates(usd_dir: Path, task_name: str, object_name: str) -> list[Path]:
    stems = (f"{task_name}_{object_name}", object_name)
    candidates = [usd_dir / f"{stem}{extension}" for stem in stems for extension in USD_EXTENSIONS]
    candidates.extend(
        path
        for path in sorted(usd_dir.iterdir())
        if path.is_file()
        and path.suffix in USD_EXTENSIONS
        and path.stem.endswith(f"_{object_name}")
        and path.name != "diningTable.usdc"
    )
    return candidates


def usd_paths_for(usd_dir: Path, task_name: str, objects: dict[str, dict]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    missing: list[str] = []
    for object_name in objects:
        matches = [path for path in _usd_candidates(usd_dir, task_name, object_name) if path.is_file()]
        if matches:
            paths[object_name] = matches[0]
        else:
            missing.append(f"{task_name}_{object_name}.usd")
    if missing:
        raise FileNotFoundError(
            "Missing USD file(s) for scene object(s):\n  " + "\n  ".join(missing) + f"\nSearched in: {usd_dir}"
        )
    return paths


def resolve_table_usd(table_usd: Path | None, usd_dir: Path) -> Path:
    if table_usd is not None:
        return table_usd
    # Prefer a copy inside the task's --usd-dir, then the shared one in the parent
    # task_usds/ folder (e.g. task_usds/basketball_in_hoop_physics/.. == task_usds).
    for candidate in (usd_dir / "diningTable.usdc", usd_dir.parent / "diningTable.usdc"):
        if candidate.is_file():
            return candidate
    return usd_dir.parent / "diningTable.usdc"


# ----------------------------------------------------------------------
# The bundle handed to the scene builder + motion code.
# ----------------------------------------------------------------------
@dataclass
class SceneContext:
    report: dict
    task_name: str
    objects: dict[str, dict]
    usd_paths: dict[str, Path]
    table_usd: Path
    table_top_z: float
    sampled_task_root_pos: tuple[float, float, float]
    sampled_task_root_quat: tuple[float, float, float, float]
    robot_base_pos: tuple[float, float, float]
    robot_base_quat: tuple[float, float, float, float]

    @property
    def waypoints(self) -> list[dict]:
        return self.report.get("waypoints") or []

    @classmethod
    def load(cls, args) -> "SceneContext":
        report = load_report(args.scene_context)
        task_name = task_name_from(report, args.scene_context)
        objects = object_entries(report)
        usd_paths = usd_paths_for(args.usd_dir, task_name, objects)
        table_usd = resolve_table_usd(args.table_usd, args.usd_dir)
        top_z = table_top_z(objects, task_name, args.table_top_object)
        root_pos, root_quat = sampled_task_root_pose(objects, task_name)
        base_pos, base_quat = robot_base_pose(report, top_z)
        return cls(
            report=report,
            task_name=task_name,
            objects=objects,
            usd_paths=usd_paths,
            table_usd=table_usd,
            table_top_z=top_z,
            sampled_task_root_pos=root_pos,
            sampled_task_root_quat=root_quat,
            robot_base_pos=base_pos,
            robot_base_quat=base_quat,
        )
