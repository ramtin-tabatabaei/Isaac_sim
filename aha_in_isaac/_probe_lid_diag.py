"""Direction-aware diagnostic for the close_box lid hinge.

Goal: explain why the lid moves "opposite to gravity" and find where it should REST.
Gravity ONLY (no robot, no drive). Logs per step:
  - signed hinge angle from spawn (deg)            -> which way it rotates
  - lid CoM WORLD z (m)                            -> RISES = gaining PE = impossible/kick
  - lid free-edge (farthest-from-hinge vert) WORLD z
  - |angular velocity| proxy (deg/step)            -> a big spike at step 0-2 = spawn depenetration KICK
  - min lid<->box gap (mm) + verts inside box AABB -> spawn interpenetration
Also prints the authored USD joint limits actually in effect.

Env COLLISION = on|off  (override the joint's collisionEnabled to isolate the cause).
Results -> /tmp/lid_diag_<COLLISION>.txt
"""
import os, sys
from pathlib import Path
import numpy as np

COLL = os.environ.get("COLLISION", "on").lower()
OUT = Path(f"/tmp/lid_diag_{COLL}.txt")
sys.argv = [
    "run_scene.py",
    "--scene-context", "/home/ramtin/AHA/portable_scene_reports/close_box.scene_context.md",
    "--usd-dir", "/home/ramtin/IsaacLab/task_usds/close_box_physics",
    "--hide-root", "--no-robot", "--device", "cpu", "--headless",
]
import run_scene as R
import isaaclab.sim as sim_utils
from pxr import Usd, UsdGeom, UsdPhysics, Gf

R.CONTEXT.report.setdefault("physics", {}).setdefault("shapes", {}).setdefault("box_lid", {})["disable_gravity"] = False
# Force the joint-collision flag for this run so we can compare on vs off.
R.CONTEXT.report["physics"].setdefault("joints", {}).setdefault("box_joint", {})["collision"] = (COLL == "on")

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device="cpu"))
builder = R.SceneBuilder(R.args_cli, R.CONTEXT, R.APPEARANCE_CONFIG)
builder.design_scene()
R._add_articulation_joints(builder)
sim.reset()

lid_path = builder.body_prim_paths.get("box_lid")
box_path = builder.body_prim_paths.get("box_base")
view = R._wand_rigid_view(lid_path)
stage = sim_utils.get_current_stage()

# locate joint + read authored limits and hinge axis
joint_prim = None
for p in Usd.PrimRange(stage.GetPrimAtPath(box_path)):
    if p.GetName().endswith("Joint"):
        joint_prim = p
        break
rj = UsdPhysics.RevoluteJoint(joint_prim)
lo = float(rj.GetLowerLimitAttr().Get())
hi = float(rj.GetUpperLimitAttr().Get())
ce = rj.GetCollisionEnabledAttr()
ce_val = bool(ce.Get()) if ce and ce.HasAuthoredValue() else None
rot0 = rj.GetLocalRot0Attr().Get()
q0 = np.array([rot0.GetReal(), *rot0.GetImaginary()])
m_base = UsdGeom.Xformable(stage.GetPrimAtPath(box_path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
jz_base = Gf.Quatd(float(q0[0]), float(q0[1]), float(q0[2]), float(q0[3])).Transform(Gf.Vec3d(0, 0, 1))
axis_w = np.array(m_base.TransformDir(jz_base)); axis_w /= (np.linalg.norm(axis_w) + 1e-12)

box = np.array(R._mesh_world_points(box_path, max_pts=3000))
box_lo, box_hi = box.min(0), box.max(0)


def pose():
    t = view.get_transforms()[0]
    t = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
    p = t[0:3].astype(float)
    qx, qy, qz, qw = (float(v) for v in t[3:7])
    Rm = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)]])
    return p, Rm, np.array([qw, qx, qy, qz])


def qmul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])


p0, Rm0, q_spawn = pose()
lid_w0 = np.array(R._mesh_world_points(lid_path, max_pts=800))
lid_local = (Rm0.T @ (lid_w0 - p0).T).T
# hinge point (world) ~ lowest lid vert at spawn; free edge = vert farthest from it
hinge_w0 = lid_w0[np.argmin(lid_w0[:, 2])]
hinge_local = Rm0.T @ (hinge_w0 - p0)
edge_idx = int(np.argmax(np.linalg.norm(lid_local - hinge_local, axis=1)))


def signed_angle(qn):
    qc = np.array([q_spawn[0], -q_spawn[1], -q_spawn[2], -q_spawn[3]])
    rel = qmul(qc, qn)
    ang = 2.0 * np.degrees(np.arccos(max(-1.0, min(1.0, abs(rel[0])))))
    v = rel[1:]; n = np.linalg.norm(v)
    return 0.0 if n < 1e-9 else float(np.sign(np.dot(v / n, axis_w)) * ang)


def metrics():
    p, Rm, qn = pose()
    lw = (Rm @ lid_local.T).T + p
    com_z = float(lw[:, 2].mean())
    edge_z = float(lw[edge_idx, 2])
    d = float(np.sqrt(((lw[:, None, :] - box[None, :, :]) ** 2).sum(-1)).min())
    inside = int(((lw >= box_lo + 0.002) & (lw <= box_hi - 0.002)).all(1).sum())
    return signed_angle(qn), com_z, edge_z, d, inside


lines = [
    f"COLLISION={COLL}  joint collisionEnabled read-back={ce_val}",
    f"authored USD joint limits: lower={lo:.2f} upper={hi:.2f} deg   hinge_axis_world={tuple(round(float(v),3) for v in axis_w)}",
    f"box AABB z=[{box_lo[2]:.3f},{box_hi[2]:.3f}]",
    "step :  signed_ang   CoM_z    edge_z   gap_mm  inside   d(ang)/step",
]
prev = None
a0, c0, e0, d0, in0 = metrics()
lines.append(f"spawn : {a0:8.2f}  {c0:7.3f}  {e0:7.3f}  {d0*1000:6.1f}  {in0:5d}      -")
prev = a0
for i in range(400):
    sim.step()
    if i in (0, 1, 2, 4, 9, 19, 39, 79, 159, 299, 399):
        a, c, e, d, ins = metrics()
        lines.append(f"{i:5d} : {a:8.2f}  {c:7.3f}  {e:7.3f}  {d*1000:6.1f}  {ins:5d}   {a-prev:8.3f}")
        prev = a
af, cf, ef, df, insf = metrics()
lines.append("--- READS ---")
lines.append(f"net CoM_z change spawn->final = {cf - c0:+.4f} m  ({'ROSE (gains PE = impossible/kick)' if cf-c0>0.003 else 'fell/flat (gravity-consistent)'})")
lines.append(f"final resting signed angle = {af:.2f} deg   (limits lower={lo:.1f} upper={hi:.1f})")
near = "AT-LOWER-LIMIT" if abs(af - lo) < 2 else ("AT-UPPER-LIMIT" if abs(af - hi) < 2 else "NOT at a limit (free equilibrium / box contact)")
lines.append(f"resting against: {near}")
OUT.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"[DIAG] wrote {OUT}")
os._exit(0)
