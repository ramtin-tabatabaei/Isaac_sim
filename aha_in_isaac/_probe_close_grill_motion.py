"""Run the REAL close_grill robot motion (reusing run_scene's machinery) and verify:
  * the arm motion's lid-close waypoint sweeps a cartesian path AND carries per-sample
    orientations (gradual EE rotation = follow_path_orientation),
  * running the motion rotates the lid on its hinge toward closed (the robot closes it),
  * the lid holds (doesn't fall back open afterwards).
Writes a JSON verdict and exits (no infinite hold)."""
import sys, os, json, math
from pathlib import Path

TASK = os.environ.get("AHA_PROBE_TASK", "close_grill")
LIDNAME = os.environ.get("AHA_LID_NAME", "lid")
CTX = f"/home/ramtin/AHA/portable_scene_reports/{TASK}.scene_context.md"
USD = f"task_usds/{TASK}_physics"
OUT = Path(f"/tmp/{TASK}_motion{os.environ.get('AHA_TAG','')}.json")

sys.argv = ["run_scene.py", "--scene-context", CTX, "--usd-dir", USD,
            "--hide-root", "--headless", "--device", "cpu"]

import run_scene  # noqa: E402  -> launches app, builds CONTEXT/MOTION_CONFIG
import isaaclab.sim as sim_utils  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

result = {"task": TASK, "error": None}


from omni.isaac.dynamic_control import _dynamic_control  # noqa: E402
_DC = _dynamic_control.acquire_dynamic_control_interface()


def lid_quat(stage, path):
    # Read the LIVE physics pose via dynamic_control (Isaac uses Fabric for physics, so the
    # USD prim transform does NOT reflect a stepped rigid body - UsdGeom.XformCache returns
    # the spawn pose and would read 0 rotation even when the lid actually swings).
    body = _DC.get_rigid_body(path)
    if body == 0:
        return (1.0, 0.0, 0.0, 0.0)
    pose = _DC.get_rigid_body_pose(body)
    r = pose.r  # quaternion xyzw
    return (float(r[3]), float(r[0]), float(r[1]), float(r[2]))


def quat_angle_deg(a, b):
    d = abs(sum(a[i] * b[i] for i in range(4)))
    d = min(1.0, max(-1.0, d))
    return math.degrees(2.0 * math.acos(d))


try:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device="cpu"))
    builder = run_scene.SceneBuilder(run_scene.args_cli, run_scene.CONTEXT, run_scene.APPEARANCE_CONFIG)
    robot = builder.design_scene()
    _gripper_physics = run_scene.MOTION_CONFIG["gripper_physics"]
    run_scene.configure_franka_gripper_friction(
        "/World/DesignScene/Robot",
        static_friction=float(_gripper_physics["static_friction"]),
        dynamic_friction=float(_gripper_physics["dynamic_friction"]),
    )
    n_joints = run_scene._add_articulation_joints(builder)

    # Optional damping override (AHA_DAMPING) on the lid hinge so the close can be swept
    # without re-editing the JSON. Overrides the angular drive's damping attr before reset.
    _damp = os.environ.get("AHA_DAMPING")
    if _damp is not None:
        from pxr import UsdPhysics as _UsdPhysics
        _stage = sim_utils.get_current_stage()
        _jname = run_scene._prim_name(LIDNAME) + "Joint"
        for _prim in _stage.Traverse():
            if _prim.GetName() == _jname:
                _drv = _UsdPhysics.DriveAPI.Apply(_prim, "angular")
                _drv.CreateTypeAttr("force")
                _drv.CreateStiffnessAttr(0.0)
                _drv.CreateDampingAttr(float(_damp))
                _drv.CreateTargetVelocityAttr(0.0)
                print(f"[PROBE]: overrode lid hinge damping -> {_damp}")
                break

    arm_waypoints = run_scene.build_arm_motion(
        run_scene.CONTEXT.waypoints, run_scene.MOTION_CONFIG,
        force_down=False, curvy=True, graspable_name=None, graspable_names=set(),
    )
    # Inspect the built motion: which waypoints sweep a path, and do they carry via_quats?
    motion_summary = []
    for w in arm_waypoints:
        vp = getattr(w, "via_points_w", None)
        vq = getattr(w, "via_quats_w", None)
        motion_summary.append({
            "name": getattr(w, "name", "?"), "gripper": getattr(w, "gripper", "?"),
            "via_points": len(vp) if vp else 0,
            "via_quats": len(vq) if vq else 0,
        })

    sim.reset()
    stage = sim_utils.get_current_stage()
    lid_path = builder.body_prim_paths.get(LIDNAME)
    q_start = lid_quat(stage, lid_path)

    gripper_closed = run_scene.MOTION_CONFIG.get("gripper_closed", run_scene.GRIPPER_CLOSED)
    controller = run_scene.FrankaWaypointController(
        robot, sim, run_scene.simulation_app,
        gripper_open=run_scene.GRIPPER_OPEN, gripper_closed=gripper_closed, planner=None,
    )
    run_scene._install_driven_close(controller, builder)  # arm the driven lid-close at its waypoint
    controller.reset_to_home()
    q_home = lid_quat(stage, lid_path)  # lid pose after the arm homes (pre-motion)
    controller.follow(arm_waypoints)
    q_after = lid_quat(stage, lid_path)
    # Use the controller's MANAGED settle (its wrapped _step) so the driven-close keeps re-applying
    # the lid drive after the sweep - a raw sim.step() loop would not run that logic.
    controller.settle(240)
    q_settled = lid_quat(stage, lid_path)

    result.update({
        "n_joints": n_joints,
        "motion_waypoints": motion_summary,
        "lid_rotation_during_motion_deg": round(quat_angle_deg(q_home, q_after), 2),
        "lid_rotation_total_deg": round(quat_angle_deg(q_home, q_settled), 2),
        "lid_change_during_settle_deg": round(quat_angle_deg(q_after, q_settled), 2),
        "sweep_has_via_quats": any(m["via_quats"] > 0 for m in motion_summary),
    })
except Exception:
    import traceback
    result["error"] = traceback.format_exc()

OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(f"[PROBE]: wrote {OUT}")
sys.stdout.flush()
run_scene.simulation_app.close()
