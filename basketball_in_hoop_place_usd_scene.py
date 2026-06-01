"""
place_usd_scene_from_context.py

Launch Isaac Sim and place exported AHA task USD objects in the design/world
layout stored in:
    /home/ramtin/AHA/portable_scene_reports/basketball_in_hoop.scene_context.md

This is a scene inspection script only: it does not spawn a robot or run a task.

Run with:
    cd ~/IsaacLab
    ./isaaclab.sh -p scripts/basketball_in_hoop_place_usd_scene.py
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

ISAACLAB_ROOT = Path(__file__).resolve().parents[1]
for package_dir in ("isaaclab", "isaaclab_assets", "isaaclab_tasks"):
    source_path = ISAACLAB_ROOT / "source" / package_dir
    if source_path.is_dir() and str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

from isaaclab.app import AppLauncher


DEFAULT_SCENE_CONTEXT = Path("/home/ramtin/AHA/portable_scene_reports/basketball_in_hoop.scene_context.md")
DEFAULT_USD_DIR = Path(
    "/home/ramtin/Downloads/basketball_in_hoop_usd-20260529T125644Z-3-001/basketball_in_hoop_usd"
)
DINING_TABLE_LOCAL_TOP_Z = 0.750022
USD_EXTENSIONS = (".usd", ".usdc", ".usda")

WAYPOINT_COLORS = (
    (0.1, 0.35, 1.0),
    (0.0, 0.75, 0.35),
    (1.0, 0.72, 0.05),
    (1.0, 0.15, 0.1),
)


def _load_scene_context(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if match is None:
        raise RuntimeError(f"No fenced JSON scene data found in {path}")
    return json.loads(match.group(1))


def _pose_from_world_location(entry: dict) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    return _pose_from_location(entry["world_location"])


def _pose_from_location(location: dict | None) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    if location is None:
        return (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)
    pos = tuple(float(v) for v in location["position_xyz_m"])
    qx, qy, qz, qw = (float(v) for v in location["quaternion_xyzw"])
    return pos, (qw, qx, qy, qz)


def _qmul(a, b) -> tuple[float, float, float, float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _qinv(q) -> tuple[float, float, float, float]:
    w, x, y, z = q
    norm = w * w + x * x + y * y + z * z
    return (w / norm, -x / norm, -y / norm, -z / norm)


def _qapply(q, v) -> tuple[float, float, float]:
    return _qmul(_qmul(q, (0.0, *v)), _qinv(q))[1:]


def _subtract_pose(world_pos, world_quat, local_pos, local_quat):
    parent_quat = _qmul(world_quat, _qinv(local_quat))
    rotated_local_pos = _qapply(parent_quat, local_pos)
    parent_pos = tuple(world_pos[i] - rotated_local_pos[i] for i in range(3))
    return parent_pos, parent_quat


def _object_entries(scene_data: dict) -> dict[str, dict]:
    return {entry["name"]: entry for entry in scene_data["objects"]}


def _task_name(scene_data: dict, scene_context: Path) -> str:
    if scene_data.get("task_name"):
        return str(scene_data["task_name"])
    return scene_context.name.removesuffix(".scene_context.md")


def _usd_candidates(usd_dir: Path, task_name: str, object_name: str) -> list[Path]:
    stems = (
        f"{task_name}_{object_name}",
        object_name,
    )
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


def _usd_paths(usd_dir: Path, task_name: str, objects: dict[str, dict]) -> dict[str, Path]:
    paths = {}
    missing = []
    for object_name in objects:
        matches = [path for path in _usd_candidates(usd_dir, task_name, object_name) if path.is_file()]
        if matches:
            paths[object_name] = matches[0]
        else:
            missing.append(f"{task_name}_{object_name}.usd")
    if missing:
        raise FileNotFoundError(
            "Missing USD file(s) for scene object(s):\n  "
            + "\n  ".join(missing)
            + f"\nSearched in: {usd_dir}"
        )
    return paths


def _task_root_object(objects: dict[str, dict], task_name: str) -> dict:
    root_candidates = [
        entry
        for entry in objects.values()
        if entry.get("parent") == task_name or entry["name"].endswith("_root") or "boundary_root" in entry["name"]
    ]
    return root_candidates[0] if root_candidates else next(iter(objects.values()))


def _sampled_task_root_pose(objects: dict[str, dict], task_name: str):
    root_entry = _task_root_object(objects, task_name)
    root_world_pos, root_world_quat = _pose_from_world_location(root_entry)
    root_local_pos, root_local_quat = _pose_from_location(root_entry.get("task_root_local_location"))
    return _subtract_pose(root_world_pos, root_world_quat, root_local_pos, root_local_quat)


def _resolve_table_usd(table_usd: Path | None, usd_dir: Path) -> Path:
    if table_usd is not None:
        return table_usd
    local_table = usd_dir / "diningTable.usdc"
    if local_table.is_file():
        return local_table
    return Path("/home/ramtin/Downloads/diningTable.usdc")


def _table_top_z(objects: dict[str, dict], task_name: str, table_top_object: str) -> float:
    if table_top_object != "auto":
        if table_top_object not in objects:
            raise KeyError(f"Table top object '{table_top_object}' was not found in the scene report.")
        return float(objects[table_top_object]["world_location"]["position_xyz_m"][2])

    return float(_sampled_task_root_pose(objects, task_name)[0][2])


def _prim_name(name: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", name)
    return "".join(word[:1].upper() + word[1:] for word in words) or "Object"


parser = argparse.ArgumentParser(description="Place exported AHA task USD objects from a scene-context report.")
parser.add_argument("--scene-context", type=Path, default=DEFAULT_SCENE_CONTEXT)
parser.add_argument("--usd-dir", type=Path, default=DEFAULT_USD_DIR)
parser.add_argument("--table-usd", type=Path, default=None)
parser.add_argument(
    "--table-top-object",
    default="auto",
    help="Scene object whose z position defines the dining-table top. Use 'auto' to infer it.",
)
parser.add_argument("--no-table", action="store_true", help="Do not spawn diningTable.usdc.")
parser.add_argument("--no-waypoints", action="store_true", help="Do not spawn waypoint marker spheres.")
parser.add_argument("--hide-root", action="store_true", help="Hide inferred task-root objects.")
parser.add_argument("--hide-object", action="append", default=[], help="Hide a named scene object. Can be repeated.")
parser.add_argument(
    "--object-pose-mode",
    choices=("task-root", "baked", "scene-context"),
    default="task-root",
    help=(
        "Use 'task-root' for exported AHA task USDs, 'baked' for files already at final world poses, "
        "or 'scene-context' for USDs exported around local object origins."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

SCENE_DATA = _load_scene_context(args_cli.scene_context)
TASK_NAME = _task_name(SCENE_DATA, args_cli.scene_context)
OBJECTS = _object_entries(SCENE_DATA)
USD_PATHS = _usd_paths(args_cli.usd_dir, TASK_NAME, OBJECTS)
TABLE_USD = _resolve_table_usd(args_cli.table_usd, args_cli.usd_dir)
TABLE_TOP_Z = _table_top_z(OBJECTS, TASK_NAME, args_cli.table_top_object)
SAMPLED_TASK_ROOT_POS, SAMPLED_TASK_ROOT_QUAT = _sampled_task_root_pose(OBJECTS, TASK_NAME)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from pxr import Usd, UsdGeom


def _ensure_xform(prim_path: str):
    stage = sim_utils.get_current_stage()
    if not stage.GetPrimAtPath(prim_path).IsValid():
        sim_utils.create_prim(prim_path, "Xform")


def _usd_bbox_center(usd_path: Path) -> tuple[float, float, float]:
    stage = Usd.Stage.Open(str(usd_path))
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], useExtentsHint=True
    )
    box = bbox_cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedBox()
    center = (box.GetMin() + box.GetMax()) * 0.5
    return tuple(float(v) for v in center)


def _canonical_task_root_pos() -> tuple[float, float, float]:
    root_entry = _task_root_object(OBJECTS, TASK_NAME)
    root_center = _usd_bbox_center(USD_PATHS[root_entry["name"]])
    root_local_pos, _ = _pose_from_location(root_entry.get("task_root_local_location"))
    return tuple(root_center[i] - root_local_pos[i] for i in range(3))


def _spawn_floor_and_table():
    floor_cfg = sim_utils.CuboidCfg(
        size=(20.0, 20.0, 0.02),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=False),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.08, 0.08, 0.08), roughness=0.9),
    )
    floor_cfg.func("/World/Floor", floor_cfg, translation=(0.0, 0.0, -0.01))

    if args_cli.no_table:
        return

    if not TABLE_USD.is_file():
        raise FileNotFoundError(f"Dining table USD file not found: {TABLE_USD}")

    table_cfg = sim_utils.UsdFileCfg(
        usd_path=str(TABLE_USD),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=False),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )
    table_translation = (0.0, 0.0, TABLE_TOP_Z - DINING_TABLE_LOCAL_TOP_Z)
    table_cfg.func("/World/DesignScene/DiningTable", table_cfg, translation=table_translation)
    print(f"[INFO]: Placed dining table at translation={tuple(round(v, 6) for v in table_translation)}")


def _spawn_lights():
    dome_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9))
    dome_cfg.func("/World/Light", dome_cfg)

    distant_cfg = sim_utils.DistantLightCfg(intensity=1800.0, color=(0.95, 0.92, 0.86))
    distant_cfg.func("/World/KeyLight", distant_cfg, translation=(1.5, -1.2, 4.0))


def _spawn_usd_objects():
    _ensure_xform("/World/DesignScene")
    hidden_objects = set(args_cli.hide_object)
    parent_path = "/World/DesignScene"
    task_root_child_translation = (0.0, 0.0, 0.0)
    task_root_child_orientation = (1.0, 0.0, 0.0, 0.0)

    if args_cli.object_pose_mode == "task-root":
        parent_path = "/World/DesignScene/TaskRoot"
        canonical_root_pos = _canonical_task_root_pos()
        sim_utils.create_prim(
            parent_path,
            "Xform",
            translation=SAMPLED_TASK_ROOT_POS,
            orientation=SAMPLED_TASK_ROOT_QUAT,
        )
        task_root_child_translation = tuple(-v for v in canonical_root_pos)
        print(
            "[INFO]: Task-root placement "
            f"sampled_pos={tuple(round(v, 6) for v in SAMPLED_TASK_ROOT_POS)} "
            f"sampled_quat_wxyz={tuple(round(v, 6) for v in SAMPLED_TASK_ROOT_QUAT)} "
            f"canonical_pos={tuple(round(v, 6) for v in canonical_root_pos)}"
        )

    for object_name, entry in OBJECTS.items():
        if object_name not in USD_PATHS:
            print(f"[WARN]: No USD mapping for object '{object_name}', skipping.")
            continue

        scene_pos, scene_quat = _pose_from_world_location(entry)
        if args_cli.object_pose_mode == "task-root":
            # The exported mesh vertices are in a canonical task-world frame. Move the whole task root once.
            pos = task_root_child_translation
            quat = task_root_child_orientation
        elif args_cli.object_pose_mode == "baked":
            # These exported USDs already include the design/world transform in the mesh hierarchy.
            # Applying the scene-context pose again double-translates them away from the waypoints.
            pos = (0.0, 0.0, 0.0)
            quat = (1.0, 0.0, 0.0, 0.0)
        else:
            pos = scene_pos
            quat = scene_quat

        cfg = sim_utils.UsdFileCfg(
            usd_path=str(USD_PATHS[object_name]),
            visible=not (
                object_name in hidden_objects
                or (args_cli.hide_root and (object_name.endswith("_root") or "boundary_root" in object_name))
            ),
        )
        prim_name = _prim_name(object_name)
        cfg.func(f"{parent_path}/{prim_name}", cfg, translation=pos, orientation=quat)
        print(
            f"[INFO]: Placed {object_name} using {args_cli.object_pose_mode} USD pose "
            f"(scene pos={tuple(round(v, 6) for v in scene_pos)}, scene quat_wxyz={scene_quat})."
        )


def _path_sample_positions(waypoint: dict) -> list[tuple[float, float, float]]:
    samples = waypoint.get("cartesian_path_samples") or []
    positions = []
    for sample in samples:
        position = sample.get("position_xyz_m")
        if position is not None:
            positions.append(tuple(float(v) for v in position))
    return positions


def _quat_from_z_axis_to_vector(vector) -> tuple[float, float, float, float]:
    length = sum(component * component for component in vector) ** 0.5
    if length < 1.0e-9:
        return (1.0, 0.0, 0.0, 0.0)

    bx, by, bz = (component / length for component in vector)
    dot = bz
    if dot < -0.999999:
        return (0.0, 1.0, 0.0, 0.0)

    # Cross product from local +Z to target direction.
    q = (1.0 + dot, -by, bx, 0.0)
    norm = math.sqrt(sum(component * component for component in q))
    return tuple(component / norm for component in q)


def _spawn_path_segment(parent_path: str, name: str, start, end, color):
    mid = tuple((start[i] + end[i]) * 0.5 for i in range(3))
    delta = tuple(end[i] - start[i] for i in range(3))
    length = sum(component * component for component in delta) ** 0.5
    if length < 1.0e-6:
        return

    cfg = sim_utils.CapsuleCfg(
        radius=0.004,
        height=length,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.45),
    )
    cfg.func(
        f"{parent_path}/{name}",
        cfg,
        translation=mid,
        orientation=_quat_from_z_axis_to_vector(delta),
    )


def _spawn_waypoint_marker(parent_path: str, name: str, pos, quat, color, radius: float = 0.012):
    marker_cfg = sim_utils.SphereCfg(
        radius=radius,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.6),
    )
    marker_cfg.func(f"{parent_path}/{name}", marker_cfg, translation=pos, orientation=quat)


def _spawn_waypoints():
    if args_cli.no_waypoints:
        return

    _ensure_xform("/World/Waypoints")
    for index, waypoint in enumerate(SCENE_DATA["waypoints"]):
        pos, quat = _pose_from_world_location(waypoint)
        color = WAYPOINT_COLORS[index % len(WAYPOINT_COLORS)]
        waypoint_path = f"/World/Waypoints/{waypoint['name']}"
        _ensure_xform(waypoint_path)

        _spawn_waypoint_marker(waypoint_path, "Pose", pos, quat, color)
        path_positions = _path_sample_positions(waypoint)
        if path_positions:
            for sample_index, sample_pos in enumerate(path_positions):
                _spawn_waypoint_marker(
                    waypoint_path,
                    f"PathSample{sample_index:02d}",
                    sample_pos,
                    (1.0, 0.0, 0.0, 0.0),
                    color,
                    radius=0.009,
                )
            for segment_index, (start, end) in enumerate(zip(path_positions, path_positions[1:])):
                _spawn_path_segment(waypoint_path, f"PathSegment{segment_index:02d}", start, end, color)
            print(
                f"[INFO]: Marked {waypoint['name']} path with {len(path_positions)} samples "
                f"at start={tuple(round(v, 6) for v in path_positions[0])} "
                f"end={tuple(round(v, 6) for v in path_positions[-1])}"
            )
        else:
            print(f"[INFO]: Marked {waypoint['name']} at pos={tuple(round(v, 6) for v in pos)}")


def design_scene():
    _ensure_xform("/World/DesignScene")
    _spawn_lights()
    _spawn_floor_and_table()
    _spawn_usd_objects()
    _spawn_waypoints()


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(1.8, 1.4, 1.65), target=(0.30, 0.03, 0.92))

    design_scene()
    sim.reset()
    print(f"[INFO]: {TASK_NAME} design scene is placed.")

    while simulation_app.is_running():
        sim.step()


if __name__ == "__main__":
    main()
    simulation_app.close()
