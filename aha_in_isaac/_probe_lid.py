"""Headless probe v2: characterise the close_box lid hinge under gravity.

Reports the SIGNED angle about the hinge axis (so we know which way gravity pulls
and which way "closed" is), prints the authored joint limits, and lets us sweep
JOINT FRICTION to see whether a stiff hinge holds the lid open without a pin/servo.

Env:
  FRIC   - joint friction coefficient (default 0 = free hinge)
  LOWER  - lower limit deg  (default from physics json: -180)
  UPPER  - upper limit deg  (default from physics json: 2)
Results -> /tmp/lid_probe2_<tag>.txt  (tag = label from env TAG, default "run")
"""
import os
import sys
from pathlib import Path

TAG = os.environ.get("TAG", "run")
FRIC = float(os.environ.get("FRIC", "0"))
LOWER = os.environ.get("LOWER")
UPPER = os.environ.get("UPPER")
OUT = Path(f"/tmp/lid_probe2_{TAG}.txt")

sys.argv = [
    "run_scene.py",
    "--scene-context", "/home/ramtin/AHA/portable_scene_reports/close_box.scene_context.md",
    "--usd-dir", "/home/ramtin/IsaacLab/task_usds/close_box_physics",
    "--hide-root", "--no-robot", "--device", "cpu", "--headless",
]

import run_scene as R
import numpy as np
import isaaclab.sim as sim_utils

lines = [f"tag={TAG} FRIC={FRIC} LOWER={LOWER} UPPER={UPPER}"]

# Normal physics: gravity ON (no pin), NO servo. Friction/limits from env.
R.CONTEXT.report.setdefault("physics", {}).setdefault("shapes", {}).setdefault("box_lid", {})["disable_gravity"] = False
jcfg = R.MOTION_CONFIG.setdefault("joints", {}).setdefault("box_lid", {})
for k in ("close_at_waypoint", "close_speed", "close_damping"):
    jcfg.pop(k, None)
if FRIC:
    jcfg["friction"] = FRIC
if LOWER is not None:
    jcfg["lower"] = float(LOWER)
if UPPER is not None:
    jcfg["upper"] = float(UPPER)

sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device="cpu")
sim = sim_utils.SimulationContext(sim_cfg)
builder = R.SceneBuilder(R.args_cli, R.CONTEXT, R.APPEARANCE_CONFIG)
builder.design_scene()
R._add_articulation_joints(builder)

# Hinge axis in world: joint local +Z mapped to world. Recompute from the joint prim.
from pxr import UsdPhysics, UsdGeom, Usd, Gf
stage = sim_utils.get_current_stage()
lid_path = builder.body_prim_paths.get("box_lid")
base_path = builder.body_prim_paths.get("box_base")
joint_prim = None
for p in Usd.PrimRange(stage.GetPrimAtPath(base_path)):
    if p.GetName().endswith("Joint"):
        joint_prim = p
        break
if joint_prim is not None:
    rj = UsdPhysics.RevoluteJoint(joint_prim)
    lo = rj.GetLowerLimitAttr().Get()
    hi = rj.GetUpperLimitAttr().Get()
    lines.append(f"authored joint limits: lower={lo} upper={hi}")
    # axis in base-local +Z -> world
    m_base = UsdGeom.Xformable(stage.GetPrimAtPath(base_path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    rot0 = rj.GetLocalRot0Attr().Get()
    q0 = np.array([rot0.GetReal(), *rot0.GetImaginary()])
else:
    lines.append("NO joint prim found")
    q0 = None

view = None


def lid_quat():
    tr = _view.get_transforms()[0]
    tr = tr.detach().cpu().numpy() if hasattr(tr, "detach") else np.asarray(tr)
    qx, qy, qz, qw = (float(v) for v in tr[3:7])
    return np.array([qw, qx, qy, qz])


def qmul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def signed_angle(q_spawn, q_now, axis_w):
    qc = np.array([q_spawn[0], -q_spawn[1], -q_spawn[2], -q_spawn[3]])
    rel = qmul(qc, q_now)  # rotation in spawn frame
    w = max(-1.0, min(1.0, rel[0]))
    ang = 2.0 * np.degrees(np.arccos(abs(w)))
    v = rel[1:]
    n = np.linalg.norm(v)
    if n < 1e-9:
        return 0.0
    sign = np.sign(np.dot(v / n, axis_w))
    return float(sign * ang)


sim.reset()
_view = R._wand_rigid_view(lid_path)
lines.append(f"view={'ok' if _view is not None else 'NONE'}")

# hinge axis in WORLD = lid-body-frame rotation of base-local +Z. Use spawn lid pose.
q_spawn = lid_quat()
# world axis: rotate base-local +Z by base world rotation
if q0 is not None:
    m_base = UsdGeom.Xformable(stage.GetPrimAtPath(base_path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    zb = Gf.Vec3d(0, 0, 1)
    # base-local joint frame +Z -> base local -> world
    jz_base = Gf.Quatd(float(q0[0]), float(q0[1]), float(q0[2]), float(q0[3])).Transform(zb)
    axis_w = np.array(m_base.TransformDir(jz_base))
    axis_w = axis_w / (np.linalg.norm(axis_w) + 1e-12)
else:
    axis_w = np.array([1.0, 0, 0])
lines.append(f"hinge_axis_world={tuple(round(float(v),3) for v in axis_w)}")

samples = []
N = 300
for i in range(N):
    sim.step()
    if i in (0, 9, 29, 59, 119, 199, 299):
        samples.append((i, round(signed_angle(q_spawn, lid_quat(), axis_w), 2)))

lines.append("step : SIGNED angle_from_spawn_deg (open=0; sign shows gravity/close direction)")
for i, a in samples:
    lines.append(f"  {i:4d} : {a:8.2f}")
lines.append(f"FINAL signed angle = {signed_angle(q_spawn, lid_quat(), axis_w):.2f} deg")

OUT.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"[PROBE2] wrote {OUT}")
os._exit(0)
