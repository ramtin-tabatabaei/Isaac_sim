"""
run_scene.py

Entry point: place an exported AHA task scene in Isaac Sim and drive a Franka
arm through the task's waypoints.

    cd ~/IsaacLab
    ./isaaclab.sh -p scripts/aha_in_isaac/run_scene.py \
        --scene-context /home/ramtin/AHA/portable_scene_reports/wipe_desk.scene_context.md \
        --usd-dir .../task_usds/wipe_desk --hide-root

Pipeline (one module per responsibility):
    cli.py            - command-line arguments
    scene_context.py  - parse the report; derive poses / table / robot base
    motion_config.py  - load per-task step counts (task_motion_config.json)
    scene_builder.py  - spawn floor / table / objects / waypoints / robot
    arm_motion.py     - build the waypoint list the robot follows
    robot_arm.py      - spawn the Franka articulation
    robot_controller.py - differential-IK waypoint controller
    add_physics_to_usds.py - (offline) bake rigid-body physics onto object USDs

Ordering matters: argparse + report parsing run first (pure Python), then the
simulator is launched, and only afterwards do we import the Isaac-dependent
modules (scene_builder / arm_motion / robot_*), which import Isaac Lab at load.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make sibling modules importable, and put the Isaac Lab source packages on the
# path (this file lives at <root>/scripts/aha_in_isaac/run_scene.py).
THIS_DIR = Path(__file__).resolve().parent
ISAACLAB_ROOT = Path(__file__).resolve().parents[2]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
for _package_dir in ("isaaclab", "isaaclab_assets", "isaaclab_tasks"):
    _source_path = ISAACLAB_ROOT / "source" / _package_dir
    if _source_path.is_dir() and str(_source_path) not in sys.path:
        sys.path.insert(0, str(_source_path))

from isaaclab.app import AppLauncher  # noqa: E402

from cli import build_parser  # noqa: E402
from motion_config import load_motion_config  # noqa: E402
from scene_context import SceneContext  # noqa: E402

# 1. Parse args and the report (pure Python, before the simulator starts).
parser = build_parser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

CONTEXT = SceneContext.load(args_cli)
MOTION_CONFIG = load_motion_config(args_cli.motion_config, CONTEXT.task_name)
APPEARANCE_CONFIG = (
    json.loads(args_cli.appearance_config.read_text(encoding="utf-8"))
    if args_cli.appearance_config.is_file()
    else {}
)

# 2. Launch the simulator.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Optional PhysX collider visualization (debug): draw every collision shape so the
# actual collider geometry (e.g. whether the ring keeps its hole) is visible.
if args_cli.show_colliders:
    import carb  # noqa: E402

    carb.settings.get_settings().set_int("/persistent/physics/visualizationDisplayColliders", 2)
    print("[INFO]: Collider visualization ON (PhysX visualizationDisplayColliders=2).")

# 3. Now that Isaac Lab is live, import the modules that depend on it.
import isaaclab.sim as sim_utils  # noqa: E402

from arm_motion import _grasp_index, build_arm_motion  # noqa: E402
from robot_arm import GRIPPER_CLOSED, GRIPPER_OPEN  # noqa: E402
from robot_controller import FrankaWaypointController  # noqa: E402
from scene_builder import SceneBuilder, _render_base  # noqa: E402

# Metres to lift the grasped wand straight up so the gripper/fingers clear the
# buzz-wire apex (~1.05 m world) during the over-the-top carry. 0.20 m puts the
# fingers well above the wire and keeps every carry point within the arm's reach.
BEAT_THE_BUZZ_AUTO_CARRY_LIFT = 0.20
# Physics steps to let the wand drop from its authored placement and settle into a
# stable resting contact on the rod (under real gravity, no pin) before the arm moves.
# The settle probe shows it converges by ~60 steps; 150 leaves a comfortable margin.
PREGRASP_SETTLE_STEPS = 150
# Friction grasp: how far BELOW the handle half-thickness the fingers close, so they pinch
# the handle (the drive force grips it) instead of just resting against it. The wand rides
# the rod (rod bears the weight), so the grip only needs to guide/slide it, not hold it up.
FRICTION_GRASP_SQUEEZE = 0.004


def _grasped_body_name(body_names) -> str | None:
    """The body nearest the lowest waypoint (where the gripper closes); it must NOT
    be an obstacle, or the arm could never reach it to grasp."""
    from scene_context import pose_from_world_location

    waypoints = CONTEXT.waypoints
    if not waypoints or not body_names:
        return None
    lowest = min(waypoints, key=lambda w: float(w["world_location"]["position_xyz_m"][2]))
    gx, gy, gz = (float(v) for v in lowest["world_location"]["position_xyz_m"])
    objects = {o["name"]: o for o in CONTEXT.report.get("objects", [])}
    best_name, best_d = None, None
    for name in body_names:
        if name not in objects:
            continue
        (px, py, pz), _ = pose_from_world_location(objects[name])
        d = (px - gx) ** 2 + (py - gy) ** 2 + (pz - gz) ** 2
        if best_d is None or d < best_d:
            best_name, best_d = name, d
    return best_name


def _graspable_object_name():
    """The graspable object = the one the report gives a placement range (the object
    the arm picks up; for beat_the_buzz that is `wand`). Used to pick the grasp
    waypoint (the one sitting on this object) instead of merely the lowest one."""
    for obj in CONTEXT.report.get("objects", []):
        if obj.get("placement_range"):
            return obj.get("name")
    return None


def _grasp_close_width(builder, graspable) -> float | None:
    """Per-finger close target DERIVED from the grasped object's handle thickness at
    the grasp point, instead of a hand-tuned constant. The fingers then rest on the
    handle rather than crushing through it - a deep squeeze stores PhysX contact energy
    that ejects the object when the gripper re-opens. Returns metres, or None if the
    geometry can't be measured."""
    if not graspable:
        return None
    path = builder.body_prim_paths.get(graspable)
    pts = _mesh_world_points(path, max_pts=4000) if path else None
    if not pts:
        return None
    waypoints = CONTEXT.waypoints
    positions = [tuple(float(v) for v in w["world_location"]["position_xyz_m"]) for w in waypoints]
    gx, gy, gz = positions[_grasp_index(waypoints, positions, graspable)]
    # Only the geometry the fingers can actually reach: within the gripper's open
    # half-span horizontally and a finger-length band vertically of the grasp point
    # (this isolates the handle from e.g. the ring ~0.12 m above it).
    near = [p for p in pts
            if ((p[0] - gx) ** 2 + (p[1] - gy) ** 2) ** 0.5 <= GRIPPER_OPEN and abs(p[2] - gz) <= GRIPPER_OPEN]
    if len(near) < 3:
        near = pts
    xs = [p[0] for p in near]
    ys = [p[1] for p in near]
    thickness = min(max(xs) - min(xs), max(ys) - min(ys))
    return float(min(max(0.5 * thickness, 0.0), GRIPPER_OPEN))


def _has_collision_filter(a: str, b: str) -> bool:
    target = {a.lower(), b.lower()}
    for first, second in getattr(args_cli, "filter_collision", []) or []:
        if {str(first).lower(), str(second).lower()} == target:
            return True
    return False


def _effective_carry_lift() -> float:
    """Resolve the requested carry lift.

    ``None`` means the user omitted --carry-lift, so we may pick a task-specific
    safe default. A numeric value, including 0, is an explicit user override.
    """
    if args_cli.carry_lift is not None:
        return float(args_cli.carry_lift)
    # The beat_the_buzz lift-over is a diff-IK sweep whose VIA POINTS trace an
    # up-and-over arc; it only applies to the planner-free diff-IK path. Planners that
    # plan to a goal pose (curobo) or do their own linear-first/RRT routing (rrt) handle
    # obstacle clearance themselves and should follow the recorded waypoints (lift 0).
    if CONTEXT.task_name == "beat_the_buzz" and args_cli.planner == "diffik":
        print(
            "[INFO]: Auto-enabling beat_the_buzz lift-over path: "
            f"--carry-lift {BEAT_THE_BUZZ_AUTO_CARRY_LIFT:.2f} "
            "(pass --carry-lift 0 to follow the recorded path)."
        )
        return BEAT_THE_BUZZ_AUTO_CARRY_LIFT
    return 0.0


def _grasped_protruding_skin_paths(builder, grasped, min_dist: float = 0.06):
    """Prim paths of the grasped object's PROTRUDING render skins (e.g. the wand's
    ring, which sticks up away from the handle), so the planner avoids driving the arm
    through them on the way in. Skins sitting AT the grasp (the handle) are excluded so
    they don't block the grasp itself.

    A skin qualifies when it belongs to the grasped body (render base == ``grasped``)
    and its world position is more than ``min_dist`` from the grasp waypoint. For a
    normally-grasped object (skin at the grasp) nothing qualifies, so other tasks are
    unaffected; for beat_the_buzz the ring (wand_visual, ~0.12 m above the grasp) does."""
    from scene_context import pose_from_world_location

    skins = getattr(builder, "skin_prim_paths", {}) or {}
    if not skins or not grasped:
        return []
    waypoints = CONTEXT.waypoints
    objects = {o["name"]: o for o in CONTEXT.report.get("objects", [])}
    positions = [tuple(float(v) for v in w["world_location"]["position_xyz_m"]) for w in waypoints]
    grasp_pos = positions[_grasp_index(waypoints, positions, _graspable_object_name())]
    extra = []
    for skin_name, skin_path in skins.items():
        if _render_base(skin_name) != grasped or skin_name not in objects:
            continue
        (px, py, pz), _ = pose_from_world_location(objects[skin_name])
        dist = ((px - grasp_pos[0]) ** 2 + (py - grasp_pos[1]) ** 2 + (pz - grasp_pos[2]) ** 2) ** 0.5
        if dist > min_dist:
            extra.append(skin_path)
            print(f"[INFO]: Treating grasped-object part '{skin_name}' as an obstacle "
                  f"(d={dist:.3f} m from the grasp).")
    return extra


def _mesh_world_points(prim_path, max_pts: int = 400):
    """World-space vertices of every Mesh under ``prim_path`` (subsampled to ``max_pts``).
    Used by the collision-watch probe to measure geometric clearance."""
    from pxr import Gf, Usd, UsdGeom

    stage = sim_utils.get_current_stage()
    root = stage.GetPrimAtPath(prim_path)
    pts: list = []
    if not root or not root.IsValid():
        return pts
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if not points:
            continue
        xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        for p in points:
            wp = xf.Transform(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
            pts.append((wp[0], wp[1], wp[2]))
    if len(pts) > max_pts:
        pts = pts[:: max(len(pts) // max_pts, 1)]
    return pts


def _mesh_world_surface_points(prim_path, target_pts: int = 5000):
    """Dense world-space point cloud sampled ON the triangle FACES of every Mesh under
    ``prim_path`` (not just its vertices), so a thin tube like the buzz-wire is well
    represented - a vertex-only cloud leaves cm-scale gaps the arm can slip through. Samples
    per triangle proportional to area, plus the raw vertices for coverage."""
    import math
    import random

    from pxr import Gf, Usd, UsdGeom

    stage = sim_utils.get_current_stage()
    root = stage.GetPrimAtPath(prim_path)
    if not root or not root.IsValid():
        return []
    tris: list = []
    verts_out: list = []
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        if not points or not counts or not indices:
            continue
        xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        wpts = [xf.Transform(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2]))) for p in points]
        verts_out += [(w[0], w[1], w[2]) for w in wpts]
        offset = 0
        for count in counts:
            face = [int(indices[offset + k]) for k in range(count)]
            for k in range(1, count - 1):  # fan-triangulate
                tris.append((wpts[face[0]], wpts[face[k]], wpts[face[k + 1]]))
            offset += count
    if not tris:
        return verts_out

    def _area(tri):
        a, b, c = tri
        ab = [b[i] - a[i] for i in range(3)]
        ac = [c[i] - a[i] for i in range(3)]
        cx = (ab[1] * ac[2] - ab[2] * ac[1], ab[2] * ac[0] - ab[0] * ac[2], ab[0] * ac[1] - ab[1] * ac[0])
        return 0.5 * math.sqrt(sum(v * v for v in cx))

    areas = [_area(t) for t in tris]
    total = sum(areas) or 1.0
    pts = list(verts_out)
    for tri, ar in zip(tris, areas):
        a, b, c = tri
        for _ in range(max(1, int(round(target_pts * ar / total)))):
            r1, r2 = random.random(), random.random()
            if r1 + r2 > 1.0:
                r1, r2 = 1.0 - r1, 1.0 - r2
            pts.append(tuple(a[i] + r1 * (b[i] - a[i]) + r2 * (c[i] - a[i]) for i in range(3)))
    return pts


def _quat_to_mat(q):
    """Rotation matrix (3,3) from a wxyz quaternion tensor."""
    import torch

    w, x, y, z = q
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)]),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)]),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]),
    ])


def _robot_mesh_local_points(robot, max_pts_per_body: int = 80):
    """Sample robot mesh vertices and express them in each body frame.

    During playback we transform these local points by the live body poses, giving
    a much better robot-vs-obstacle clearance estimate than body origins alone.
    """
    from pxr import Gf, Usd, UsdGeom

    stage = sim_utils.get_current_stage()
    root = stage.GetPrimAtPath("/World/DesignScene/Robot")
    if not root or not root.IsValid():
        return []

    body_prims = {}
    for prim in Usd.PrimRange(root):
        name = prim.GetName()
        if name in robot.body_names and name not in body_prims:
            body_prims[name] = prim

    samples = []
    for body_id, body_name in enumerate(robot.body_names):
        body_prim = body_prims.get(body_name)
        if body_prim is None:
            continue
        body_xf = UsdGeom.Xformable(body_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        body_inv = body_xf.GetInverse()
        pts = []
        for prim in Usd.PrimRange(body_prim):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
            if not points:
                continue
            mesh_xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            for p in points:
                world = mesh_xf.Transform(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
                local = body_inv.Transform(world)
                pts.append((float(local[0]), float(local[1]), float(local[2])))
        if len(pts) > max_pts_per_body:
            pts = pts[:: max(len(pts) // max_pts_per_body, 1)][:max_pts_per_body]
        if pts:
            samples.append((body_id, pts))
    print(f"[COLLISION-WATCH] sampled robot mesh points from {len(samples)} body/bodies.")
    return samples


def _wand_rigid_view(grasped_prim_path):
    """A physics-tensor view on the grasped object's rigid body so we can read its TRUE
    world pose each step (to detect grasp slip/droop). Uses the same GPU-safe physics
    view Isaac Lab uses internally (read-only ``get_transforms``); the higher-level
    RigidPrim wrapper is NOT usable here (it calls setGlobalPose, illegal under the
    direct-GPU-API pipeline). The RigidBodyAPI usually lives on a child of the spawned
    prim, so we search for it. Returns the view or None if unavailable."""
    try:
        from isaacsim.core.simulation_manager import SimulationManager
    except Exception as exc:  # pragma: no cover - depends on Isaac build
        print(f"[COLLISION-WATCH] physics sim view unavailable ({exc}); using rigid-grasp approximation.")
        return None
    from pxr import Usd, UsdPhysics

    stage = sim_utils.get_current_stage()
    root = stage.GetPrimAtPath(grasped_prim_path)
    body_path = None
    if root and root.IsValid():
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                body_path = str(prim.GetPath())
                break
    body_path = body_path or grasped_prim_path
    try:
        view = SimulationManager.get_physics_sim_view().create_rigid_body_view(body_path)
        transforms = view.get_transforms()
        if transforms is None or transforms.shape[0] == 0:
            raise RuntimeError("rigid body view is empty")
        print(f"[COLLISION-WATCH] reading TRUE wand pose from '{body_path}' (bodies={transforms.shape[0]}).")
        return view
    except Exception as exc:
        print(f"[COLLISION-WATCH] could not view '{body_path}' ({exc}); using rigid-grasp approximation.")
        return None


def _load_franka_collision_spheres(robot, device):
    """The Franka's own collision spheres (from the Lula robot descriptor) mapped to
    live body ids. This is the authoritative robot collision geometry - the same set a
    planner uses - so the probe can measure the TRUE arm-surface->obstacle clearance
    instead of a body-origin proxy. Returns (body_ids, local_centers, radii) tensors."""
    import os

    import torch

    mg = os.environ.get(
        "AHA_LULA_MG",
        "/home/ramtin/miniconda3/envs/env_isaacsim51/lib/python3.11/site-packages/"
        "isaacsim/exts/isaacsim.robot_motion.motion_generation",
    )
    path = os.path.join(mg, "motion_policy_configs/franka/rmpflow/robot_descriptor.yaml")
    try:
        import yaml

        entries = yaml.safe_load(open(path, encoding="utf-8")).get("collision_spheres", [])
    except Exception as exc:  # pragma: no cover - depends on Isaac install
        print(f"[COLLISION-WATCH] could not load collision spheres ({exc}); arm-clearance check off.")
        return None
    name_to_id = {name: i for i, name in enumerate(robot.body_names)}
    bids, centers, radii = [], [], []
    for entry in entries:
        for link, sphere_list in entry.items():
            bid = name_to_id.get(link)
            if bid is None:
                continue
            for sphere in sphere_list:
                bids.append(bid)
                centers.append([float(v) for v in sphere["center"]])
                radii.append(float(sphere["radius"]))
    if not bids:
        return None
    print(f"[COLLISION-WATCH] loaded {len(bids)} robot collision spheres "
          f"across {len(set(bids))} link(s).")
    return (
        torch.tensor(bids, dtype=torch.long, device=device),
        torch.tensor(centers, dtype=torch.float32, device=device),
        torch.tensor(radii, dtype=torch.float32, device=device),
    )


def _soften_rod_friction_for_slide(builder):
    """For the slide-along-rod motion, lower the wand+rod friction (on their baked
    ``<root>/PhysicsMaterial``) so the captive ring SLIDES along the rod. The baked 5.0 is for a
    stable resting grasp and makes the ring grab the rod; a real beat-the-buzz ring slides nearly
    free. Wand stays high enough for the gripper to hold the handle; the rod is dropped low."""
    from pxr import UsdPhysics

    stage = sim_utils.get_current_stage()
    for name, mu in (("wand", 1.5), ("Cuboid", 0.2)):
        root = builder.body_prim_paths.get(name)
        if not root:
            continue
        mat = stage.GetPrimAtPath(root + "/PhysicsMaterial")
        if not mat or not mat.IsValid():
            print(f"[SLIDE]: no PhysicsMaterial under {root}; friction unchanged for {name}.")
            continue
        api = UsdPhysics.MaterialAPI.Apply(mat)
        api.CreateStaticFrictionAttr().Set(float(mu))
        api.CreateDynamicFrictionAttr().Set(float(mu))
        print(f"[SLIDE]: {name} friction -> {mu} so the ring slides along the rod.")


def _install_screenshot_capture(controller, sim, out_dir, interval_s):
    """Wrap the controller's physics step so a viewport screenshot is saved every
    ``interval_s`` seconds of WALL-CLOCK time while the arm runs - a running visual record
    of the scene. Captures are asynchronous (the renderer writes the PNG on its next frame),
    which is fine at a multi-second cadence. Needs the GUI/renderer (not --headless)."""
    import time

    # Save under the IsaacLab folder: a relative dir is resolved against the repo root, so the
    # frames land in <IsaacLab>/<dir> no matter the current working directory.
    out = Path(out_dir)
    if not out.is_absolute():
        out = ISAACLAB_ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    try:
        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
    except Exception as exc:
        print(f"[SCREENSHOT]: viewport utility unavailable ({exc}); screenshots disabled.")
        return
    viewport = get_active_viewport()
    if viewport is None:
        print("[SCREENSHOT]: no active viewport (are you running --headless?); screenshots disabled.")
        return

    state = {"last": 0.0, "shots": 0, "started": False}
    orig_step = controller._step

    def stepped():
        orig_step()
        now = time.monotonic()
        if not state["started"] or (now - state["last"]) >= interval_s:
            path = str(out / f"frame_{state['shots']:04d}.png")
            try:
                capture_viewport_to_file(viewport, path)
                state["shots"] += 1
                state["last"] = now
                state["started"] = True
            except Exception as exc:
                if not state["started"]:
                    print(f"[SCREENSHOT]: capture failed ({exc}); screenshots disabled.")
                    controller._step = orig_step  # stop trying

    controller._step = stepped
    print(f"[SCREENSHOT]: saving a viewport frame every {interval_s:.1f}s (wall clock) -> {out}/frame_*.png")


def _install_collision_watch(builder, robot, controller, csv_path):
    """Wrap the controller's physics step to record, each step, the closest distance
    between (a) the grasped object's visible geometry (the ring) and the obstacles, and
    (b) sampled robot mesh vertices / wrist-hand origins and the obstacles.

    The ring is tracked by transforming its spawn geometry with the grasped body's
    motion. We read the body's TRUE pose from a physics view when available (so grasp
    slip/droop is captured); otherwise we fall back to assuming it is rigid to the hand.
    Returns a state dict consumed by ``_report_collision_watch``."""
    import torch

    device = robot.device
    grasped = _grasped_body_name(list(builder.body_prim_paths)) or "wand"

    obs_pts: list = []
    for name, path in builder.body_prim_paths.items():
        if name != grasped:
            obs_pts += _mesh_world_surface_points(path, target_pts=5000)  # sample faces: see the thin wire
    ring_pts: list = []
    for skin_name, skin_path in (getattr(builder, "skin_prim_paths", {}) or {}).items():
        if _render_base(skin_name) == grasped:
            ring_pts += _mesh_world_points(skin_path)
    if grasped in builder.body_prim_paths:
        ring_pts += _mesh_world_points(builder.body_prim_paths[grasped])

    # The ring LOOP only (top of the wand_visual skin), for the hole-frame PCA. Mixing the
    # handle in (ring_pts above) skews the centroid/normal, so keep this set separate.
    ring_loop_pts: list = []
    loop_path = (getattr(builder, "skin_prim_paths", {}) or {}).get("wand_visual")
    if loop_path:
        import numpy as _np
        lp = _mesh_world_points(loop_path, max_pts=600)
        if lp:
            a = _np.asarray(lp)
            a = a[a[:, 2] > a[:, 2].max() - 0.09]  # keep the top loop, drop the handle
            ring_loop_pts = a.tolist()

    obs = torch.tensor(obs_pts, dtype=torch.float32, device=device) if obs_pts else None
    ring0 = torch.tensor(ring_pts, dtype=torch.float32, device=device) if ring_pts else None
    ring_loop0 = torch.tensor(ring_loop_pts, dtype=torch.float32, device=device) if ring_loop_pts else None
    print(f"[COLLISION-WATCH] grasped='{grasped}' obstacle_pts={0 if obs is None else obs.shape[0]} "
          f"ring_pts={0 if ring0 is None else ring0.shape[0]}")

    hand_id = robot.find_bodies("panda_hand")[0][0]
    wrist_ids = robot.find_bodies("panda_(hand|link5|link6|link7|leftfinger|rightfinger)")[0]
    body_names = list(robot.body_names)
    robot_spheres = _load_franka_collision_spheres(robot, device)  # (bids, centers, radii) or None
    robot_mesh_samples = [
        (body_id, torch.tensor(local_pts, dtype=torch.float32, device=device))
        for body_id, local_pts in _robot_mesh_local_points(robot)
    ]
    # Track the ring from the grasped body's TRUE pose (read via a low-level physics-
    # tensor view, which is GPU-pipeline-safe; the high-level RigidPrim wrapper is not).
    # If the view is unavailable the probe falls back to a rigid-to-hand approximation;
    # the true-pose view is preferred because the wand now has gravity enabled.
    wand_view = _wand_rigid_view(builder.body_prim_paths.get(grasped)) if ring0 is not None else None

    def _view_pose():
        transforms = wand_view.get_transforms()  # (N,7): pos(3) + quat xyzw(4)
        row = transforms[0].to(device).float()
        p = row[0:3]
        qx, qy, qz, qw = row[3], row[4], row[5], row[6]
        q = torch.stack([qw, qx, qy, qz])  # -> wxyz for _quat_to_mat
        return p, q

    state = {
        "rows": [],
        "min_ring": (1.0e9, -1),
        "min_arm": (1.0e9, -1),
        "min_robot_mesh": (1.0e9, -1),
        "min_sphere": (1.0e9, -1, -1),  # (clearance, step, body_id) - true arm-surface clearance
        "ring_local": None,
        "wand0": None,
        "use_view": wand_view is not None,
        "error": None,
        "wand_start_z": None,
        "wand_min_z": None,
        "wand_end_z": None,
        # Captive-ring tracking: how many obstacle (rod) points pierce the ring's hole each
        # step, and the wand's speed. If the count drops from >0 to 0 the ring left the rod -
        # which, for this topologically-captive ring, can ONLY be a tunnel-through.
        "hole": None,            # (C_local, N_local, r_in) in the grasped body frame
        "ring_loop0": ring_loop0,  # world ring-loop pts at spawn (for the hole-frame PCA)
        "prev_p": None,
        "rod_in_hole_start": None,
        "rod_in_hole_min": 1_000_000,
        "tunnel_step": None,     # first step rod-in-hole hit 0 after being threaded
        "tunnel_speed": None,
    }
    orig_step = controller._step

    # Gripper collision points in the panda_hand frame (same model the linear-first
    # controller uses): the fingers/tip that are most likely to clip a thin rod. The
    # Franka USD is instanceable so the mesh-vertex sampler finds nothing; this gives a
    # reliable gripper->obstacle clearance instead.
    gripper_local = torch.tensor(
        [(0.0, 0.0, 0.0), (0.0, 0.0, 0.058), (0.0, 0.0, 0.10),
         (0.0, 0.04, 0.10), (0.0, -0.04, 0.10), (0.0, 0.0, 0.11)],
        dtype=torch.float32, device=device,
    )

    def _measure(step):
        bp = robot.data.body_pose_w[0]
        d_arm = torch.cdist(bp[wrist_ids, 0:3], obs).min().item()
        # TRUE arm clearance: place every Franka collision sphere by its link's live pose
        # (batched quaternion rotate) and take the closest sphere-SURFACE distance to any
        # obstacle point. Negative => the arm penetrates the obstacle.
        d_sphere = float("nan")
        d_sphere_link = -1
        if robot_spheres is not None:
            bids, local, radii = robot_spheres
            q = bp[bids, 3:7]  # (S,4) wxyz
            qv, qw = q[:, 1:], q[:, 0:1]
            cross1 = 2.0 * torch.cross(qv, local, dim=1)
            world_c = bp[bids, 0:3] + local + qw * cross1 + torch.cross(qv, cross1, dim=1)
            per_sphere = torch.cdist(world_c, obs).min(dim=1).values - radii  # (S,)
            dmn, idx = torch.min(per_sphere, dim=0)
            d_sphere, d_sphere_link = float(dmn), int(bids[idx])
            if d_sphere < state["min_sphere"][0]:
                state["min_sphere"] = (d_sphere, step, body_names[d_sphere_link])
        hand_p, hand_q = bp[hand_id, 0:3], bp[hand_id, 3:7]
        # Gripper-model clearance: transform the local finger/tip points by the live
        # hand pose and measure the closest distance to any obstacle point.
        grip_w = (_quat_to_mat(hand_q) @ gripper_local.T).T + hand_p
        d_robot_mesh = torch.cdist(grip_w, obs).min().item()
        if robot_mesh_samples:
            live_pts = []
            for body_id, local_pts in robot_mesh_samples:
                body_p = bp[body_id, 0:3]
                body_q = bp[body_id, 3:7]
                rot = _quat_to_mat(body_q)
                live_pts.append((rot @ local_pts.T).T + body_p)
            robot_mesh = torch.cat(live_pts, dim=0)
            d_robot_mesh = min(d_robot_mesh, torch.cdist(robot_mesh, obs).min().item())
        grip = controller._current_grip
        d_ring = float("nan")
        wand_z = ""
        row_rod_in_hole = ""
        row_speed = ""
        if ring0 is not None and state["use_view"]:
            # TRUE pose: ring_world(t) = T(t) * T(0)^-1 * ring0  (captures slip/droop).
            wand_p, wand_q = _view_pose()
            wand_z_value = float(wand_p[2])
            wand_z = round(wand_z_value, 4)
            if state["wand_start_z"] is None:
                state["wand_start_z"] = wand_z_value
            state["wand_min_z"] = wand_z_value if state["wand_min_z"] is None else min(state["wand_min_z"], wand_z_value)
            state["wand_end_z"] = wand_z_value
            if state["wand0"] is None:
                r0 = _quat_to_mat(wand_q)
                state["wand0"] = (r0.T @ (ring0 - wand_p).T).T  # ring in body frame at spawn
            rot = _quat_to_mat(wand_q)
            ring_w = (rot @ state["wand0"].T).T + wand_p
            d_ring = torch.cdist(ring_w, obs).min().item()
            # Captive-ring metric: build the ring-hole frame once (PCA on the ring in body
            # frame: thinnest axis = hole normal, centroid = hole center, 5th-pct in-plane
            # radius = inner radius), then count rod points piercing the hole disk each step.
            if state["hole"] is None and obs is not None and state["ring_loop0"] is not None:
                # Ring loop in the body frame at spawn, then PCA: thinnest axis = hole normal,
                # centroid = hole center, 5th-pct in-plane radius = inner radius.
                rl = (rot.T @ (state["ring_loop0"] - wand_p).T).T
                c_local = rl.mean(0)
                centered = rl - c_local
                _, _, Vt = torch.linalg.svd(centered, full_matrices=False)
                n_local = Vt[2]
                inplane = centered - torch.outer(centered @ n_local, n_local)
                r_in = float(torch.quantile(torch.linalg.norm(inplane, dim=1), 0.05))
                state["hole"] = (c_local, n_local, r_in)
            if state["hole"] is not None:
                c_local, n_local, r_in = state["hole"]
                cw = rot @ c_local + wand_p
                nw = rot @ n_local
                rel = obs - cw
                along = rel @ nw
                perp = torch.linalg.norm(rel - torch.outer(along, nw), dim=1)
                pierce = int(((perp < r_in) & (along.abs() < 0.015)).sum())
                row_rod_in_hole = pierce
                if state["rod_in_hole_start"] is None:
                    state["rod_in_hole_start"] = pierce
                state["rod_in_hole_min"] = min(state["rod_in_hole_min"], pierce)
                if (state["tunnel_step"] is None and pierce == 0
                        and (state["rod_in_hole_start"] or 0) > 0):
                    state["tunnel_step"] = step  # tunnel_speed filled in once speed is known below
            # wand speed (m/s) from the true pose, ~120 Hz physics step.
            if state["prev_p"] is not None:
                row_speed = float(torch.linalg.norm(wand_p - state["prev_p"]) * 120.0)
            else:
                row_speed = 0.0
            state["prev_p"] = wand_p.clone()
            if state.get("tunnel_step") == step and state.get("tunnel_speed") is None:
                state["tunnel_speed"] = row_speed
        elif ring0 is not None:
            # Fallback: assume the ring is rigid to the hand from the grasp moment.
            rot = _quat_to_mat(hand_q)
            if state["ring_local"] is None and grip <= controller.gripper_closed + 1.0e-3:
                state["ring_local"] = (rot.T @ (ring0 - hand_p).T).T
            if state["ring_local"] is not None:
                ring_w = (rot @ state["ring_local"].T).T + hand_p
                d_ring = torch.cdist(ring_w, obs).min().item()
        if d_arm < state["min_arm"][0]:
            state["min_arm"] = (d_arm, step)
        if d_robot_mesh == d_robot_mesh and d_robot_mesh < state["min_robot_mesh"][0]:
            state["min_robot_mesh"] = (d_robot_mesh, step)
        if d_ring == d_ring and d_ring < state["min_ring"][0]:
            state["min_ring"] = (d_ring, step)
        state["rows"].append(
            (step, getattr(controller, "_active_label", ""), round(grip, 4),
             round(float(hand_p[0]), 4), round(float(hand_p[1]), 4),
             round(float(hand_p[2]), 4), round(d_arm, 4),
             "" if d_robot_mesh != d_robot_mesh else round(d_robot_mesh, 4),
             "" if d_ring != d_ring else round(d_ring, 4), wand_z,
             "" if d_sphere != d_sphere else round(d_sphere, 4),
             "" if d_sphere_link < 0 else body_names[d_sphere_link],
             row_rod_in_hole, "" if row_speed == "" else round(row_speed, 4))
        )

    def watched():
        orig_step()
        if obs is None or state["error"] == "fatal":
            return
        step = len(state["rows"])
        try:
            _measure(step)
        except Exception as exc:  # never let the probe abort the motion
            import traceback
            if state["error"] is None:
                state["error"] = f"{exc!r}"
                with open(str(csv_path) + ".debug.txt", "a", encoding="utf-8") as handle:
                    handle.write(f"[step {step}] probe error: {exc!r}\n{traceback.format_exc()}\n")
                if state["use_view"]:
                    # The true-pose view is the likely culprit; fall back to rigid-grasp.
                    state["use_view"] = False
                    state["wand0"] = None
                    handle = open(str(csv_path) + ".debug.txt", "a", encoding="utf-8")
                    handle.write("[probe] disabling true-pose view; falling back to rigid-grasp.\n")
                    handle.close()
                else:
                    state["error"] = "fatal"

    # Persist setup info immediately (app.close() can drop buffered stdout).
    with open(str(csv_path) + ".debug.txt", "w", encoding="utf-8") as handle:
        handle.write(f"grasped={grasped} obstacle_pts={0 if obs is None else obs.shape[0]} "
                     f"ring_pts={0 if ring0 is None else ring0.shape[0]} use_view={state['use_view']}\n")

    controller._step = watched
    return state


def _report_collision_watch(state, csv_path):
    import csv

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "step", "active_target", "grip", "hand_x", "hand_y", "hand_z",
            "hand_wrist_min_dist_m", "robot_mesh_min_dist_m", "ring_min_dist_m", "wand_z",
            "arm_sphere_clear_m", "closest_link", "rod_in_hole_pts", "wand_speed_mps"
        ])
        writer.writerows(state["rows"])
    arm_d, arm_s = state["min_arm"]
    robot_mesh_d, robot_mesh_s = state["min_robot_mesh"]
    ring_d, ring_s = state["min_ring"]
    sphere_d, sphere_s, sphere_link = state["min_sphere"]
    threshold_m = 0.005
    arm_hit = arm_s >= 0 and arm_d <= threshold_m
    robot_mesh_hit = robot_mesh_s >= 0 and robot_mesh_d <= threshold_m
    ring_hit = ring_s >= 0 and ring_d <= threshold_m
    # The authoritative arm collision: a Franka collision SPHERE within 5 mm of (or inside)
    # the obstacle. This is the "robot hit the Cuboid" check the user actually cares about.
    sphere_hit = sphere_s >= 0 and sphere_d <= threshold_m
    drop_line = "[COLLISION-WATCH] wand z unavailable (true rigid-body view was not available)"
    drop_hit = False
    if state.get("wand_start_z") is not None and state.get("wand_min_z") is not None:
        drop = state["wand_start_z"] - state["wand_min_z"]
        drop_hit = drop > 0.02
        drop_line = (
            f"[COLLISION-WATCH] wand z start/min/end = "
            f"{state['wand_start_z']:.4f}/{state['wand_min_z']:.4f}/{state['wand_end_z']:.4f} m "
            f"(drop {drop * 1000:.1f} mm)"
        )
    arm_collision = arm_hit or robot_mesh_hit or sphere_hit
    # Captive-ring / tunnel report: did the rod leave the ring's hole (=tunnel-through)?
    rih_start = state.get("rod_in_hole_start")
    rih_min = state.get("rod_in_hole_min")
    tunnel_step = state.get("tunnel_step")
    tunnel_speed = state.get("tunnel_speed")
    if rih_start is None:
        captive_line = "[COLLISION-WATCH] captive-ring tracking unavailable (no true wand view)."
    elif tunnel_step is not None:
        captive_line = (
            f"[COLLISION-WATCH] RING TUNNELED OUT OF THE ROD at step {tunnel_step} "
            f"(rod-in-hole {rih_start}->0; wand speed there "
            f"{('%.3f m/s' % tunnel_speed) if tunnel_speed is not None else 'n/a'}). "
            f"The ring is topologically captive, so this is a collision tunnel-through, not a valid removal."
        )
    else:
        captive_line = (
            f"[COLLISION-WATCH] ring stayed threaded on the rod the whole run "
            f"(rod-in-hole start={rih_start}, min={rih_min}) - NO tunneling."
        )
    verdict = "COLLISION/FAIL" if (arm_collision or ring_hit or drop_hit or tunnel_step is not None) else "CLEAR/PASS"
    sphere_line = (
        f"[COLLISION-WATCH] closest ARM collision-SPHERE->obstacle    = {sphere_d * 1000:.1f} mm "
        f"(step {sphere_s}, link {sphere_link})  <-- TRUE arm-vs-Cuboid"
        if sphere_s >= 0 else
        "[COLLISION-WATCH] ARM collision-sphere clearance unavailable"
    )
    summary = (
        f"[COLLISION-WATCH] closest HAND/WRIST body-origin->obstacle = {arm_d * 1000:.1f} mm (step {arm_s})\n"
        f"[COLLISION-WATCH] closest GRIPPER-model->obstacle          = {robot_mesh_d * 1000:.1f} mm (step {robot_mesh_s})\n"
        f"{sphere_line}\n"
        f"[COLLISION-WATCH] closest RING(grasped)->obstacle          = {ring_d * 1000:.1f} mm (step {ring_s})\n"
        f"{drop_line}\n"
        f"{captive_line}\n"
        f"[COLLISION-WATCH] route verdict: {verdict} (threshold <= {threshold_m * 1000:.1f} mm; drop > 20 mm)\n"
        f"[COLLISION-WATCH] (<~5 mm == touching/penetrating)  trace -> {csv_path}"
    )
    print(summary)
    # app.close() can drop stdout, so persist the summary next to the CSV too.
    with open(str(csv_path) + ".summary.txt", "w", encoding="utf-8") as handle:
        handle.write(summary + "\n")


def _build_planner(builder):
    # Obstacles = static scene bodies, EXCEPT the grasped one (so the arm can reach
    # it). The table is left out: its bbox reaches the work surface and would block
    # every waypoint (they sit just above it).
    grasped = _grasped_body_name(list(builder.body_prim_paths))
    obstacle_names = [name for name in builder.body_prim_paths if name != grasped]
    obstacle_paths = [builder.body_prim_paths[name] for name in obstacle_names]
    print(f"[INFO]: Planner grasped body (not an obstacle): {grasped}")
    print(f"[INFO]: Planner obstacle bodies: {obstacle_names}")
    # NOTE: the wand's ring is NOT a planner obstacle. The approach + grasp is executed
    # with a deterministic side-approach (differential IK following the recorded
    # wp0->wp1 vector) that clears the ring geometrically; cuRobo then plans only the
    # CARRY (where the ring is held), avoiding the cuboid. See
    # FrankaWaypointController._follow_with_batch_planner.

    if args_cli.planner == "rrt":
        from lula_planner import LulaRrtPlanner

        # RRT plans per-segment on demand; obstacles are handled by the controller's
        # straight-line collision check (it passes the colliding rod points to the
        # planner as spheres), so nothing is registered here.
        return LulaRrtPlanner(
            CONTEXT.robot_base_pos, CONTEXT.robot_base_quat, args_cli.device,
            max_iterations=args_cli.rrt_max_iter,
        )

    if args_cli.planner == "curobo":
        from curobo_planner import CuroboPlanner

        # cuRobo takes obstacles at construction (scene_model); no separate add call.
        return CuroboPlanner(
            CONTEXT.robot_base_pos, CONTEXT.robot_base_quat, args_cli.device, obstacle_paths,
            obstacle_mode=args_cli.curobo_obstacles,
            safety_margin=args_cli.curobo_safety_margin,
            use_graph=(args_cli.curobo_graph == "on"),
        )

    from rmpflow_planner import RmpFlowPlanner

    planner = RmpFlowPlanner("/World/DesignScene/Robot", args_cli.device)
    planner.set_base_pose(CONTEXT.robot_base_pos, CONTEXT.robot_base_quat)
    planner.add_box_obstacles(obstacle_paths)
    return planner


def _run_settle_probe(builder, n_steps, sim, csv_path):
    """Step the sim ``n_steps`` under pure gravity (no pin, no grasp) and log the wand's
    TRUE rigid-body pose plus the min distance from its (moving) collider to the static
    Cuboid collider each step. Tells us whether the ring settles and HANGS on the rod
    (z drops a little then holds, ring->rod distance -> ~0 and stable) or FALLS off (z
    keeps dropping toward the table). Writes a CSV + .summary.txt and returns."""
    import csv

    import numpy as np

    wand_root = builder.body_prim_paths.get("wand")
    cuboid_root = builder.body_prim_paths.get("Cuboid")
    view = _wand_rigid_view(wand_root) if wand_root else None
    cub_pts = np.array(_mesh_world_points(cuboid_root, max_pts=4000)) if cuboid_root else np.empty((0, 3))
    if view is None or cub_pts.size == 0:
        print(f"[SETTLE-PROBE] cannot run (view={view is not None}, cuboid_pts={cub_pts.shape}).")
        return

    def _pose():
        t = view.get_transforms()[0]
        t = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
        p = t[0:3].astype(float)
        qx, qy, qz, qw = (float(v) for v in t[3:7])  # view quat is xyzw
        # rotation matrix from xyzw
        R = np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
        ])
        return p, R

    # Body-local collider points from the spawn pose, so we can re-place them each step.
    p0, R0 = _pose()
    wand_world0 = np.array(_mesh_world_points(wand_root, max_pts=400))
    wand_local = (R0.T @ (wand_world0 - p0).T).T

    # Separate the two contacts that matter: the RING vs the WIRE, and the HANDLE vs the BASE.
    # Ring = the wand_visual render skin; wire = upper Cuboid pts (z>0.95); base box = lower
    # Cuboid pts (z<0.88) bounding box. Track ring->wire distance and how far the wand collider
    # PENETRATES the base box (points sunk inside it = the "wand inside the base" the user saw).
    ring_root = (getattr(builder, "skin_prim_paths", {}) or {}).get("wand_visual")
    ring_w0 = np.array(_mesh_world_points(ring_root, max_pts=300)) if ring_root else wand_world0
    ring_local = (R0.T @ (ring_w0 - p0).T).T
    wire = cub_pts[cub_pts[:, 2] > 0.95]
    base = cub_pts[cub_pts[:, 2] < 0.88]
    base_lo, base_hi = (base.min(0), base.max(0)) if len(base) else (np.zeros(3), np.zeros(3))

    def _metrics(p, R):
        w = (R @ wand_local.T).T + p
        ring = (R @ ring_local.T).T + p
        d_all = float(np.sqrt(((w[:, None, :] - cub_pts[None, :, :]) ** 2).sum(-1)).min())
        d_ring_wire = float(np.sqrt(((ring[:, None, :] - wire[None, :, :]) ** 2).sum(-1)).min()) if len(wire) else -1.0
        inside = ((w >= base_lo + 0.002) & (w <= base_hi - 0.002)).all(1)  # 2mm inside all 3 axes
        pen = int(inside.sum())
        pen_depth = float((w[inside, 2].min() and base_hi[2] - w[inside, 2].min())) if pen else 0.0
        return d_all, d_ring_wire, pen, pen_depth

    rows = []
    z0 = float(p0[2])
    d_start, dw_start, pen0, _ = _metrics(p0, R0)
    print(f"[SETTLE-PROBE] start: ring->wire={dw_start*1000:.1f}mm, base-penetration pts={pen0}. Stepping {n_steps}...")
    for i in range(n_steps):
        sim.step()
        p, R = _pose()
        d_all, dw, pen, pend = _metrics(p, R)
        rows.append((i, float(p[0]), float(p[1]), float(p[2]), d_all, dw, pen, pend))

    csv_path = Path(csv_path)
    with open(csv_path, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["step", "wand_x", "wand_y", "wand_z", "wand_cuboid_min_m",
                       "ring_wire_min_m", "base_penetration_pts", "base_penetration_depth_m"])
        for r in rows:
            wcsv.writerow([r[0], round(r[1], 5), round(r[2], 5), round(r[3], 5),
                           round(r[4], 5), round(r[5], 5), r[6], round(r[7], 5)])

    z_end = rows[-1][3]
    d_end = rows[-1][4]
    dw_end = rows[-1][5]
    pen_end = rows[-1][6]
    pen_max = max(r[6] for r in rows)
    pend_max = max(r[7] for r in rows)
    z_min = min(r[3] for r in rows)
    # Judge by the ring<->rod relationship, NOT the body-origin z (the wand geometry is
    # baked ~1 m above its rigid-body origin, so origin z is meaningless as a height).
    # Settled-and-hanging = the ring->rod distance ends SMALL (resting on the rod) and the
    # last ~50 steps barely move (xy + dist stable). Fell off / ejected = the distance ends
    # large (ring left the rod) or the body is still flying (last steps not converged).
    tail = rows[-min(50, len(rows)):]
    pos_span = max(
        max(r[1] for r in tail) - min(r[1] for r in tail),
        max(r[2] for r in tail) - min(r[2] for r in tail),
        max(r[3] for r in tail) - min(r[3] for r in tail),
    )
    ring_on_wire = dw_end < 0.020   # ring still riding the wire (<20 mm)
    penetrating = pen_end > 0       # wand collider sunk into the base box
    stable = pos_span < 0.003       # body not drifting/flying in the last ~50 steps
    if penetrating:
        verdict = f"WAND PENETRATING THE BASE ({pen_end} pts inside, up to {pend_max*1000:.0f} mm deep) - contact failed"
    elif ring_on_wire and stable:
        verdict = "RING RIDES THE WIRE (settled, stable, no penetration)"
    elif dw_end > 0.10:
        verdict = "RING LEFT THE WIRE (slid/fell off)"
    elif not stable:
        verdict = "NOT SETTLED (still moving at end)"
    else:
        verdict = f"UNCERTAIN (ring->wire={dw_end*1000:.1f} mm, tail span={pos_span*1000:.1f} mm)"
    summary = (
        f"[SETTLE-PROBE] steps={n_steps}\n"
        f"  ring->WIRE dist: start={dw_start*1000:.1f} mm end={dw_end*1000:.1f} mm\n"
        f"  base PENETRATION: end={pen_end} pts, max over run={pen_max} pts (max depth {pend_max*1000:.1f} mm)\n"
        f"  wand->cuboid min: end={d_end*1000:.1f} mm ; tail({len(tail)}-step) body span={pos_span*1000:.2f} mm\n"
        f"  VERDICT: {verdict}\n"
    )
    with open(str(csv_path) + ".summary.txt", "w") as f:
        f.write(summary)
    print(summary)


def _run_pull_test(builder, speed, sim, csv_path):
    """Isolated COLLISION test (no robot). Settle the bare wand on the rod under gravity,
    then drive the wand's rigid body along the RECORDED carry direction (grasp waypoint ->
    last waypoint) at ``speed`` m/s and log, each step, whether the rod is still threaded
    through the ring's hole and the closest ring-tube<->rod surface distance.

    Interpretation: the recorded carry is ~0.2 m almost entirely PERPENDICULAR to the rod
    axis, so with real collision the rod cannot leave the ring's hole (it would have to pass
    through the ring's solid tube) -> the wand should STALL, rod stays threaded. If instead
    the wand sails the full distance and the rod leaves the hole, the ring TUNNELED through
    the rod = a collider defect."""
    import csv

    import numpy as np
    import torch

    wand_root = builder.body_prim_paths.get("wand")
    cuboid_root = builder.body_prim_paths.get("Cuboid")
    view = _wand_rigid_view(wand_root) if wand_root else None
    cub_pts = np.array(_mesh_world_points(cuboid_root, max_pts=8000)) if cuboid_root else np.empty((0, 3))
    if view is None or cub_pts.size == 0:
        print(f"[PULL-TEST] cannot run (view={view is not None}, cuboid_pts={cub_pts.shape}).")
        return

    def _pose():
        t = view.get_transforms()[0]
        t = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
        p = t[0:3].astype(float)
        qx, qy, qz, qw = (float(v) for v in t[3:7])
        R = np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
        ])
        return p, R

    # Recorded carry direction: grasp waypoint -> last waypoint (world).
    wps = CONTEXT.waypoints
    pos = [np.array([float(v) for v in w["world_location"]["position_xyz_m"]]) for w in wps]
    gi = min(1, len(pos) - 1)
    direction = pos[-1] - pos[gi]
    dist_total = float(np.linalg.norm(direction))
    direction = direction / (dist_total or 1.0)
    print(f"[PULL-TEST] carry dir (wp{gi}->wp{len(pos)-1}) = "
          f"[{direction[0]:.3f},{direction[1]:.3f},{direction[2]:.3f}], length {dist_total*1000:.0f} mm")

    # Settle the ring onto the rod first (pure gravity), then snapshot the ring-hole frame.
    for _ in range(150):
        sim.step()
    p0, R0 = _pose()

    # Ring geometry (the wand_visual render skin = the loop). Build the hole frame by PCA:
    # the ring is a flat loop, so its thinnest principal axis is the hole normal, the centroid
    # is the hole center, and the min in-plane radius is the hole's inner radius.
    ring_root = (getattr(builder, "skin_prim_paths", {}) or {}).get("wand_visual")
    ring_w = np.array(_mesh_world_points(ring_root, max_pts=400)) if ring_root else np.array(_mesh_world_points(wand_root, max_pts=400))
    ring_local = (R0.T @ (ring_w - p0).T).T
    c_local = ring_local.mean(0)
    centered = ring_local - c_local
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    normal_local = Vt[2]  # smallest-variance axis = hole normal
    inplane = centered - np.outer(centered @ normal_local, normal_local)
    radii = np.linalg.norm(inplane, axis=1)
    in_radius = float(np.percentile(radii, 5))  # robust inner radius
    print(f"[PULL-TEST] ring hole: inner radius ~{in_radius*1000:.1f} mm, "
          f"normal(local)=[{normal_local[0]:.2f},{normal_local[1]:.2f},{normal_local[2]:.2f}]")

    # Wand collider points (for the closest ring-tube <-> rod distance) in body-local frame.
    wand_w0 = np.array(_mesh_world_points(wand_root, max_pts=600))
    wand_local = (R0.T @ (wand_w0 - p0).T).T

    def _rod_in_hole(p, R):
        """How many static Cuboid points currently pierce the ring's hole disk (projection
        within the inner radius AND within +/-15 mm of the ring plane)."""
        c = R @ c_local + p
        n = R @ normal_local
        rel = cub_pts - c
        along = rel @ n
        proj = rel - np.outer(along, n)
        rad = np.linalg.norm(proj, axis=1)
        pierce = (rad < in_radius) & (np.abs(along) < 0.015)
        return int(pierce.sum())

    def _ring_rod_min(p, R):
        w = (R @ wand_local.T).T + p
        return float(np.sqrt(((w[:, None, :] - cub_pts[None, :, :]) ** 2).sum(-1)).min())

    pierce0 = _rod_in_hole(p0, R0)
    d0 = _ring_rod_min(p0, R0)
    print(f"[PULL-TEST] after settle: rod-in-hole pts={pierce0}, ring->rod min={d0*1000:.1f} mm. "
          f"Pulling at {speed} m/s...")

    n_steps = int((dist_total + 0.05) / max(speed, 1e-3) * 120) + 1  # cover full dist + margin @120Hz
    n_steps = min(n_steps, 2000)
    vel = torch.tensor([[direction[0] * speed, direction[1] * speed, direction[2] * speed,
                         0.0, 0.0, 0.0]], dtype=torch.float32)
    idx = torch.tensor([0], dtype=torch.int32)
    rows = []
    for i in range(n_steps):
        try:
            view.set_velocities(vel, idx)
        except Exception:
            view.set_velocities(vel)
        sim.step()
        p, R = _pose()
        moved = float(np.linalg.norm((p - p0) * np.array([1, 1, 1])))
        rows.append((i, float(p[0]), float(p[1]), float(p[2]), moved,
                     _rod_in_hole(p, R), _ring_rod_min(p, R)))

    csv_path = Path(csv_path)
    with open(csv_path, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["step", "wand_x", "wand_y", "wand_z", "moved_m", "rod_in_hole_pts", "ring_rod_min_m"])
        for r in rows:
            wcsv.writerow([r[0], round(r[1], 5), round(r[2], 5), round(r[3], 5),
                           round(r[4], 5), r[5], round(r[6], 5)])

    moved_end = rows[-1][4]
    pierce_end = rows[-1][5]
    pierce_min = min(r[5] for r in rows)
    # When did the rod leave the hole (pierce -> 0), and how far had the wand moved by then?
    left_at = next((r for r in rows if r[5] == 0), None)
    frac = moved_end / (dist_total or 1.0)
    if pierce_end > 0 and frac < 0.4:
        verdict = (f"BLOCKED by the rod (real collision OK): rod still threaded "
                   f"({pierce_end} pts), wand stalled at {moved_end*1000:.0f} mm of {dist_total*1000:.0f} mm.")
    elif pierce_end == 0 and left_at is not None:
        verdict = (f"RING TUNNELED THROUGH THE ROD (collider defect): rod left the hole after only "
                   f"{left_at[4]*1000:.0f} mm of lateral pull; wand reached {moved_end*1000:.0f} mm "
                   f"({frac*100:.0f}% of the {dist_total*1000:.0f} mm carry). The lateral move cannot "
                   f"free the ring without the rod crossing the ring's solid tube.")
    else:
        verdict = (f"UNCERTAIN: rod-in-hole end={pierce_end} (min {pierce_min}), "
                   f"moved {moved_end*1000:.0f}/{dist_total*1000:.0f} mm.")
    summary = (
        f"[PULL-TEST] speed={speed} m/s, steps={n_steps}\n"
        f"  carry dir=[{direction[0]:.3f},{direction[1]:.3f},{direction[2]:.3f}], full carry={dist_total*1000:.0f} mm\n"
        f"  rod-in-hole pts: start={pierce0} end={pierce_end} (min over run={pierce_min})\n"
        f"  wand displacement: {moved_end*1000:.0f} mm ({frac*100:.0f}% of carry)\n"
        f"  VERDICT: {verdict}\n"
    )
    with open(str(csv_path) + ".summary.txt", "w") as f:
        f.write(summary)
    print(summary)


def main():
    # A finer step + PhysX tuning are needed for a stable friction grasp; fall
    # back to the lighter inspect-only step when no robot/grasping is involved.
    if args_cli.no_robot:
        sim_cfg = sim_utils.SimulationCfg(
            dt=1.0 / 60.0,
            device=args_cli.device,
        )
    else:
        sim_cfg = sim_utils.SimulationCfg(
            dt=1.0 / 120.0,
            device=args_cli.device,
            physx=sim_utils.PhysxCfg(
                enable_external_forces_every_iteration=True,
                min_velocity_iteration_count=2,
            ),
        )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(1.8, 1.4, 1.65), target=(0.30, 0.03, 0.92))

    builder = SceneBuilder(args_cli, CONTEXT, APPEARANCE_CONFIG)
    robot = builder.design_scene()
    # beat_the_buzz: the ring is topologically CAPTIVE on the rod, so the recorded sideways carry
    # (waypoint3) is impossible - it tunnels/ejects the ring (the wand flies off). Default to
    # sliding the ring ALONG the rod instead, which keeps it threaded. Pass --slide-along-rod 0
    # to force the (broken) recorded carry.
    beat = CONTEXT.task_name == "beat_the_buzz"
    slide_raw = getattr(args_cli, "slide_along_rod", None)
    # Default OFF: the robot follows the RECORDED waypoints. Pass --slide-along-rod DIST to use the
    # along-rod slide instead (the only motion that keeps the captive ring threaded).
    slide_along_rod = 0.0 if slide_raw is None else float(slide_raw)
    carry_lift = 0.0 if slide_along_rod > 0.0 else _effective_carry_lift()
    if slide_along_rod > 0.0:
        print(f"[INFO]: beat_the_buzz: sliding the captive ring {slide_along_rod * 1000:.0f} mm ALONG "
              "the rod (the recorded sideways carry would eject the captive ring).")
    # The linear-first/RRT planner needs STRAIGHT segments so its straight-line collision
    # check matches what executes (a curved Catmull-Rom path wouldn't), so force it off.
    curvy = (not args_cli.straight_path) and args_cli.planner != "rrt"
    arm_waypoints = build_arm_motion(
        CONTEXT.waypoints, MOTION_CONFIG, force_down=args_cli.ee_down,
        curvy=curvy, carry_lift=carry_lift,
        graspable_name=_graspable_object_name(),
        slide_along_rod=slide_along_rod,
    )
    # The baked 5.0 friction makes the ring GRAB the rod, so the forced slide jerks it off. Lower
    # it so the ring slides freely (now safe: CCD + the depenetration cap are baked). ONLY when the
    # robot actually runs the slide - a --no-robot settle test keeps the high baked friction.
    if slide_along_rod > 0.0 and robot is not None:
        _soften_rod_friction_for_slide(builder)
    settle_probe = int(getattr(args_cli, "settle_probe", 0) or 0)
    beat = CONTEXT.task_name == "beat_the_buzz"
    friction_grasp = beat and robot is not None and settle_probe == 0
    on_grasp, on_release = None, None  # contact-only: no pin, no joint
    sim.reset()
    print(f"[INFO]: {CONTEXT.task_name} design scene is placed.")

    if settle_probe > 0:
        _run_settle_probe(builder, settle_probe, sim,
                          args_cli.collision_watch or Path("/tmp/settle_probe.csv"))
        return

    if robot is None:
        while simulation_app.is_running():
            sim.step()
        return

    # Optional collision-aware planner (RMPFlow / cuRobo) with the scene's static
    # objects as obstacles. Default 'diffik' uses no planner.
    planner = _build_planner(builder) if args_cli.planner != "diffik" else None

    # For the RRT planner, give the controller a DENSE obstacle point cloud (every
    # body except the grasped one). It downsamples that into the sphere set it routes
    # every segment around (AHA-style collision-free planning); the dense sampling is
    # so a thin tube like the buzz-wire is not missed.
    obstacle_points = None
    if args_cli.planner == "rrt":
        grasped = _grasped_body_name(list(builder.body_prim_paths))
        obstacle_points = []
        for name, path in builder.body_prim_paths.items():
            if name != grasped:
                obstacle_points += _mesh_world_surface_points(path)  # dense: see the thin wire
        print(f"[INFO]: RRT obstacle points: {len(obstacle_points)} (dense, from bodies != {grasped}).")

    # Close below the handle half-thickness so the fingers physically grip the wand.
    gripper_closed = MOTION_CONFIG.get("gripper_closed", GRIPPER_CLOSED)
    derived = _grasp_close_width(builder, _graspable_object_name()) if friction_grasp else None
    if friction_grasp and derived is not None:
        gripper_closed = max(derived - FRICTION_GRASP_SQUEEZE, 0.0)
        print(f"[INFO]: Friction grasp close width {gripper_closed * 1000.0:.1f} mm (handle half "
              f"~{derived * 1000.0:.1f} mm minus a {FRICTION_GRASP_SQUEEZE * 1000.0:.0f} mm squeeze; fingers grip).")
    controller = FrankaWaypointController(
        robot,
        sim,
        simulation_app,
        gripper_open=GRIPPER_OPEN,
        gripper_closed=gripper_closed,
        planner=planner,
        on_grasp_complete=on_grasp,
        on_release=on_release,
        obstacle_points=obstacle_points,
    )
    controller.apply_gripper_friction()
    watch_state = (
        _install_collision_watch(builder, robot, controller, args_cli.collision_watch)
        if args_cli.collision_watch else None
    )
    if getattr(args_cli, "screenshot_dir", None) is not None:
        _install_screenshot_capture(controller, sim, args_cli.screenshot_dir, args_cli.screenshot_interval)
    controller.reset_to_home()

    # The arm ALWAYS follows the RECORDED waypoints (the dots) - no wand tracking / no
    # re-targeting to wherever the wand settles. (settle_view/authored_wand_pose are kept only
    # for diagnostics; the grasp pose is the recorded waypoint regardless of wand placement.)
    controller.follow(arm_waypoints)
    print("[INFO]: Arm motion finished. Holding final pose.")
    if watch_state is not None:
        # After the gripper has released, step a little longer (arm parked) so the trace
        # captures the wand resting on the wire (it should stay threaded, kept off the base by
        # the wand<->Cuboid collision, not clip through it or fall off).
        if beat:
            print(f"[INFO]: Post-release settle for {PREGRASP_SETTLE_STEPS} steps "
                  "(confirming the wand stays on the wire, doesn't clip the base)...")
            controller.settle(PREGRASP_SETTLE_STEPS)
        # Diagnostic run: write the trace and exit (don't hold forever) so the
        # measurement can be read back; hold() never returns under --headless.
        _report_collision_watch(watch_state, args_cli.collision_watch)
        return
    controller.hold()


if __name__ == "__main__":
    main()
    simulation_app.close()
