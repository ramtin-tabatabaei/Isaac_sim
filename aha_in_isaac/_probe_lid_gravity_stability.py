"""Gravity-stability probe for the close_grill lid hinge.

Question: with NO robot force, does the lid stay put on its hinge, or does its own
weight droop it open? We override the hinge JointFriction to AHA_HINGE_FRICTION,
settle the scene WITHOUT moving the arm, and report how many degrees the lid drifts
from its spawn pose. A stable lid drifts ~0 deg; a drooping lid swings toward its
lower/upper limit. Run for several friction values to find the smallest that holds.

  AHA_HINGE_FRICTION : float to write into PhysxJointAPI.jointFriction before reset
  AHA_PROBE_TASK     : task (default close_grill)
  AHA_LID_NAME       : dynamic lid body name (default lid)
  AHA_SETTLE_STEPS   : settle steps (default 600)
"""
import sys, os, json, math
from pathlib import Path

TASK = os.environ.get("AHA_PROBE_TASK", "close_grill")
LIDNAME = os.environ.get("AHA_LID_NAME", "lid")
FRICTION = float(os.environ.get("AHA_HINGE_FRICTION", "1.0"))
SETTLE = int(os.environ.get("AHA_SETTLE_STEPS", "600"))
CTX = f"/home/ramtin/AHA/portable_scene_reports/{TASK}.scene_context.md"
USD = f"task_usds/{TASK}_physics"
TAG = os.environ.get("AHA_TAG", f"f{FRICTION}")
OUT = Path(f"/tmp/{TASK}_gravity_{TAG}.json")

sys.argv = ["run_scene.py", "--scene-context", CTX, "--usd-dir", USD,
            "--hide-root", "--headless", "--device", "cpu"]

import run_scene  # noqa: E402  -> launches app, builds CONTEXT/MOTION_CONFIG
import isaaclab.sim as sim_utils  # noqa: E402
from pxr import PhysxSchema, UsdPhysics  # noqa: E402

DAMPING = os.environ.get("AHA_DAMPING")  # if set, apply a pure angular damper (velocity drive)

result = {"task": TASK, "hinge_friction": FRICTION, "settle_steps": SETTLE, "error": None}

from omni.isaac.dynamic_control import _dynamic_control  # noqa: E402
_DC = _dynamic_control.acquire_dynamic_control_interface()


def lid_quat(path):
    body = _DC.get_rigid_body(path)
    if body == 0:
        return (1.0, 0.0, 0.0, 0.0)
    pose = _DC.get_rigid_body_pose(body)
    r = pose.r  # xyzw
    return (float(r[3]), float(r[0]), float(r[1]), float(r[2]))


def quat_angle_deg(a, b):
    d = abs(sum(a[i] * b[i] for i in range(4)))
    d = min(1.0, max(-1.0, d))
    return math.degrees(2.0 * math.acos(d))


try:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device="cpu"))
    builder = run_scene.SceneBuilder(run_scene.args_cli, run_scene.CONTEXT, run_scene.APPEARANCE_CONFIG)
    robot = builder.design_scene()
    n_joints = run_scene._add_articulation_joints(builder)

    # Optional override of the hinge friction on the USD joint prim BEFORE reset so PhysX parses it.
    # run_scene names the joint prim "<_prim_name(name)>Joint" (e.g. lid -> "lidJoint").
    # If AHA_HINGE_FRICTION is unset we test the joint EXACTLY as the JSON built it (drive incl.).
    stage = sim_utils.get_current_stage()
    jname = run_scene._prim_name(LIDNAME) + "Joint"
    lid_joint_path = None
    for prim in stage.Traverse():
        if prim.GetName() == jname:
            lid_joint_path = prim.GetPath().pathString
            if "AHA_HINGE_FRICTION" in os.environ:
                PhysxSchema.PhysxJointAPI.Apply(prim).CreateJointFrictionAttr(FRICTION)
            if DAMPING is not None:
                # Pure angular damper: stiffness 0, targetVel 0, damping D -> resists any
                # angular velocity (viscous joint friction). No spring-back (stiffness 0).
                drv = UsdPhysics.DriveAPI.Apply(prim, "angular")
                drv.CreateTypeAttr("force")
                drv.CreateStiffnessAttr(0.0)
                drv.CreateDampingAttr(float(DAMPING))
                drv.CreateTargetVelocityAttr(0.0)
            break
    result["lid_joint_path"] = lid_joint_path
    result["override_friction"] = "AHA_HINGE_FRICTION" in os.environ
    result["override_damping"] = DAMPING

    sim.reset()
    lid_path = builder.body_prim_paths.get(LIDNAME)
    q_start = lid_quat(lid_path)

    # Settle WITHOUT any robot motion - pure gravity on the lid hinge.
    samples = []
    for i in range(SETTLE):
        sim.step(render=False)
        if i % 60 == 0 or i == SETTLE - 1:
            samples.append((i, round(quat_angle_deg(q_start, lid_quat(lid_path)), 2)))
    q_end = lid_quat(lid_path)

    result.update({
        "n_joints": n_joints,
        "lid_droop_deg": round(quat_angle_deg(q_start, q_end), 2),
        "droop_timeline": samples,
    })
except Exception:
    import traceback
    result["error"] = traceback.format_exc()

OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(f"[GRAVITY-PROBE]: friction={FRICTION} -> wrote {OUT}")
sys.stdout.flush()
run_scene.simulation_app.close()
