"""Verify the real fix: keep collision ON and the open rest at the recorded pose
(USD-relative upper limit = 0). The lid spawns ~0.7mm overlapping the box at the
hinge; with collision on PhysX depenetrates it, kicking the lid NEGATIVE (the closing
direction) toward the gravitational apex. Two knobs to stop that:

  DAMPING     - joint angular damping (absorb the kick so gravity returns lid to open)
  REST_OFFSET - lid collider rest offset in m, negative tolerates the hinge overlap
                without an impulse (e.g. -0.004)

Phase A: gravity settle 250 steps -> does the lid REST at the open pose (tilt ~29.5,
         no CoM rise)?  Phase B: drive CLOSED (negative sweep) -> seats on box, stable?
Results -> /tmp/lid_fix_d<DAMPING>_r<REST_OFFSET>.txt
"""
import os, sys
from pathlib import Path
import numpy as np

DAMPING = float(os.environ.get("DAMPING", "5"))
REST_OFFSET = os.environ.get("REST_OFFSET", "")  # "" = leave as-baked
APPROX = os.environ.get("APPROX", "")            # "" = as-baked, else convexDecomposition/convexHull
TAG = f"d{DAMPING}".replace(".", "p") + (f"_r{REST_OFFSET}".replace(".", "p").replace("-", "m") if REST_OFFSET else "") + (f"_{APPROX}" if APPROX else "")
OUT = Path(f"/tmp/lid_fix_{TAG}.txt")
sys.argv = [
    "run_scene.py",
    "--scene-context", "/home/ramtin/AHA/portable_scene_reports/close_box.scene_context.md",
    "--usd-dir", "/home/ramtin/IsaacLab/task_usds/close_box_physics",
    "--hide-root", "--no-robot", "--device", "cpu", "--headless",
]
import run_scene as R
import isaaclab.sim as sim_utils
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema

R.CONTEXT.report.setdefault("physics", {}).setdefault("shapes", {}).setdefault("box_lid", {})["disable_gravity"] = False
j = R.CONTEXT.report["physics"].setdefault("joints", {}).setdefault("box_joint", {})
j["collision"] = True
j["lower"] = -180.0
j["upper"] = -27.5      # USD-relative upper = 0 -> rest at the recorded OPEN pose
j["position"] = -27.5
j["damping"] = DAMPING

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device="cpu"))
builder = R.SceneBuilder(R.args_cli, R.CONTEXT, R.APPEARANCE_CONFIG)
builder.design_scene()
R._add_articulation_joints(builder)

lid_path = builder.body_prim_paths.get("box_lid")
box_path = builder.body_prim_paths.get("box_base")
stage = sim_utils.get_current_stage()

# Optional: override the lid collider rest offset (tolerate the hinge overlap).
applied_ro = "as-baked"
if REST_OFFSET != "":
    ro = float(REST_OFFSET)
    for p in Usd.PrimRange(stage.GetPrimAtPath(lid_path)):
        if p.HasAPI(UsdPhysics.CollisionAPI):
            pc = PhysxSchema.PhysxCollisionAPI.Apply(p)
            pc.CreateRestOffsetAttr(ro)
            if not pc.GetContactOffsetAttr().HasAuthoredValue():
                pc.CreateContactOffsetAttr(max(ro + 0.005, 0.005))
            applied_ro = f"{ro}"

sim.reset()
view = R._wand_rigid_view(lid_path)
joint_prim = None
for p in Usd.PrimRange(stage.GetPrimAtPath(box_path)):
    if p.GetName().endswith("Joint"):
        joint_prim = p
        break
rj = UsdPhysics.RevoluteJoint(joint_prim)
lo = float(rj.GetLowerLimitAttr().Get()); hi = float(rj.GetUpperLimitAttr().Get())

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
    return p, Rm


p0, Rm0 = pose()
lid_w0 = np.array(R._mesh_world_points(lid_path, max_pts=800))
lid_local = (Rm0.T @ (lid_w0 - p0).T).T
cov = np.cov((lid_local - lid_local.mean(0)).T)
_, evecs = np.linalg.eigh(cov)
normal_local = evecs[:, 0]


def metrics():
    p, Rm = pose()
    lw = (Rm @ lid_local.T).T + p
    n_world = Rm @ normal_local
    tilt = float(np.degrees(np.arccos(min(1.0, abs(n_world[2]) / (np.linalg.norm(n_world) + 1e-12)))))
    com_z = float(lw[:, 2].mean())
    d = float(np.sqrt(((lw[:, None, :] - box[None, :, :]) ** 2).sum(-1)).min())
    inside = int(((lw >= box_lo + 0.002) & (lw <= box_hi - 0.002)).all(1).sum())
    return tilt, com_z, d, inside


lines = [f"DAMPING={DAMPING}  REST_OFFSET={applied_ro}  USD limits lower={lo:.2f} upper={hi:.2f}"]
t0 = metrics()
lines.append(f"spawn      : tilt={t0[0]:5.1f}  CoM_z={t0[1]:.3f}  gap={t0[2]*1000:6.1f}mm  inside={t0[3]}")
peak = t0[1]
for i in range(250):
    sim.step()
    if i in (9, 49, 124, 249):
        m = metrics(); peak = max(peak, m[1])
        lines.append(f"  grav s{i:4d}: tilt={m[0]:5.1f}  CoM_z={m[1]:.3f}  gap={m[2]*1000:6.1f}mm  inside={m[3]}")
ta = metrics()
kick = ta[1] - t0[1]; peak_rise = peak - t0[1]
rest_clean = abs(kick) < 0.005 and peak_rise < 0.010
lines.append(f"after-grav : tilt={ta[0]:5.1f}  CoM_z={ta[1]:.3f}  net_CoM_rise={kick*1000:+.1f}mm  peak_rise={peak_rise*1000:.1f}mm")
lines.append(f"  -> REST CLEAN (lid stays ~open, no fling): {rest_clean}")
# PHASE B: drive CLOSED (negative sweep)
drive = UsdPhysics.DriveAPI.Get(joint_prim, "angular") or UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
drive.CreateDampingAttr(max(DAMPING, 8.0)); drive.CreateMaxForceAttr(120.0)
(drive.GetTargetVelocityAttr() or drive.CreateTargetVelocityAttr(0.0)).Set(-30.0)
for i in range(460):
    sim.step()
tb = metrics()
seated = tb[3] <= 1 and tb[2] * 1000 < 6.0 and tb[0] < 15
lines.append(f"after-close: tilt={tb[0]:5.1f}  CoM_z={tb[1]:.3f}  gap={tb[2]*1000:6.1f}mm  inside={tb[3]}")
lines.append("--- VERDICT ---")
lines.append(f"REST CLEAN (no spontaneous opposite-to-gravity motion): {rest_clean}  (peak CoM rise {peak_rise*1000:.1f}mm)")
lines.append(f"CLOSED SEATED ON BOX: {seated}  (tilt {tb[0]:.1f}, gap {tb[2]*1000:.1f}mm, inside {tb[3]})")
OUT.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"[FIX] wrote {OUT}")
os._exit(0)
