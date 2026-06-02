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


def _quat_to_mat(q):
    """Rotation matrix (3,3) from a wxyz quaternion tensor."""
    import torch

    w, x, y, z = q
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)]),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)]),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]),
    ])


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


def _install_collision_watch(builder, robot, controller, csv_path):
    """Wrap the controller's physics step to record, each step, the closest distance
    between (a) the grasped object's visible geometry (the ring) and the obstacles, and
    (b) the wrist/hand/finger links and the obstacles.

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
            obs_pts += _mesh_world_points(path)
    ring_pts: list = []
    for skin_name, skin_path in (getattr(builder, "skin_prim_paths", {}) or {}).items():
        if _render_base(skin_name) == grasped:
            ring_pts += _mesh_world_points(skin_path)
    if grasped in builder.body_prim_paths:
        ring_pts += _mesh_world_points(builder.body_prim_paths[grasped])

    obs = torch.tensor(obs_pts, dtype=torch.float32, device=device) if obs_pts else None
    ring0 = torch.tensor(ring_pts, dtype=torch.float32, device=device) if ring_pts else None
    print(f"[COLLISION-WATCH] grasped='{grasped}' obstacle_pts={0 if obs is None else obs.shape[0]} "
          f"ring_pts={0 if ring0 is None else ring0.shape[0]}")

    hand_id = robot.find_bodies("panda_hand")[0][0]
    wrist_ids = robot.find_bodies("panda_(hand|link5|link6|link7|leftfinger|rightfinger)")[0]
    # Reading the grasped body's live pose via a physics-tensor view segfaults this
    # standalone, direct-GPU-API pipeline, so we track the ring rigidly relative to the
    # hand instead. That is accurate here: the wand has gravity disabled and is held by
    # fully-closed, high-friction fingers, so it does not droop or slip appreciably.
    wand_view = None

    def _view_pose():
        transforms = wand_view.get_transforms()  # (N,7): pos(3) + quat xyzw(4)
        row = transforms[0].to(device).float()
        p = row[0:3]
        qx, qy, qz, qw = row[3], row[4], row[5], row[6]
        q = torch.stack([qw, qx, qy, qz])  # -> wxyz for _quat_to_mat
        return p, q

    state = {"rows": [], "min_ring": (1.0e9, -1), "min_arm": (1.0e9, -1),
             "ring_local": None, "wand0": None, "use_view": wand_view is not None, "error": None}
    orig_step = controller._step

    def _measure(step):
        bp = robot.data.body_pose_w[0]
        d_arm = torch.cdist(bp[wrist_ids, 0:3], obs).min().item()
        hand_p, hand_q = bp[hand_id, 0:3], bp[hand_id, 3:7]
        grip = controller._current_grip
        d_ring = float("nan")
        wand_z = ""
        if ring0 is not None and state["use_view"]:
            # TRUE pose: ring_world(t) = T(t) * T(0)^-1 * ring0  (captures slip/droop).
            wand_p, wand_q = _view_pose()
            wand_z = round(float(wand_p[2]), 4)
            if state["wand0"] is None:
                r0 = _quat_to_mat(wand_q)
                state["wand0"] = (r0.T @ (ring0 - wand_p).T).T  # ring in body frame at spawn
            rot = _quat_to_mat(wand_q)
            ring_w = (rot @ state["wand0"].T).T + wand_p
            d_ring = torch.cdist(ring_w, obs).min().item()
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
        if d_ring == d_ring and d_ring < state["min_ring"][0]:
            state["min_ring"] = (d_ring, step)
        state["rows"].append(
            (step, round(grip, 4), round(float(hand_p[0]), 4), round(float(hand_p[1]), 4),
             round(float(hand_p[2]), 4), round(d_arm, 4), "" if d_ring != d_ring else round(d_ring, 4), wand_z)
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
        writer.writerow(["step", "grip", "hand_x", "hand_y", "hand_z", "arm_min_dist_m", "ring_min_dist_m", "wand_z"])
        writer.writerows(state["rows"])
    arm_d, arm_s = state["min_arm"]
    ring_d, ring_s = state["min_ring"]
    summary = (
        f"[COLLISION-WATCH] closest ARM(wrist/hand/fingers)->obstacle = {arm_d * 1000:.1f} mm (step {arm_s})\n"
        f"[COLLISION-WATCH] closest RING(grasped)->obstacle        = {ring_d * 1000:.1f} mm (step {ring_s})\n"
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


def main():
    # A finer step + PhysX tuning are needed for a stable friction grasp; fall
    # back to the lighter inspect-only step when no robot/grasping is involved.
    if args_cli.no_robot:
        sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, device=args_cli.device)
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
    sim.reset()
    print(f"[INFO]: {CONTEXT.task_name} design scene is placed.")

    if robot is None:
        while simulation_app.is_running():
            sim.step()
        return

    # Optional collision-aware planner (RMPFlow / cuRobo) with the scene's static
    # objects as obstacles. Default 'diffik' uses no planner.
    planner = _build_planner(builder) if args_cli.planner != "diffik" else None

    # The close width is per-task (object sizes differ); fall back to the sponge
    # default. Fully-open is fixed at the Franka finger limit, so it is not tuned.
    controller = FrankaWaypointController(
        robot,
        sim,
        simulation_app,
        gripper_open=GRIPPER_OPEN,
        gripper_closed=MOTION_CONFIG.get("gripper_closed", GRIPPER_CLOSED),
        planner=planner,
    )
    controller.apply_gripper_friction()
    controller.reset_to_home()

    watch_state = (
        _install_collision_watch(builder, robot, controller, args_cli.collision_watch)
        if args_cli.collision_watch else None
    )

    controller.follow(
        build_arm_motion(
            CONTEXT.waypoints, MOTION_CONFIG, force_down=args_cli.ee_down,
            curvy=not args_cli.straight_path, carry_lift=args_cli.carry_lift,
            graspable_name=_graspable_object_name(),
        )
    )
    print("[INFO]: Arm motion finished. Holding final pose.")
    if watch_state is not None:
        # Diagnostic run: write the trace and exit (don't hold forever) so the
        # measurement can be read back; hold() never returns under --headless.
        _report_collision_watch(watch_state, args_cli.collision_watch)
        return
    controller.hold()


if __name__ == "__main__":
    main()
    simulation_app.close()
