"""Probe: with base<->lid joint collision ENABLED (close_box.json "collision": true),
drive the lid CLOSED and check it (a) seats on the box rim instead of sinking through,
and (b) stays stable (no hinge jitter / explosion).

Phases:
  OPEN-SETTLE  : 150 steps gravity only -> is the spawn pose stable with collision on?
  CLOSE        : add a velocity drive toward the lower (closing) limit, 300 steps ->
                 does the lid stop ON the box (tilt->~0, penetration bounded) and hold?

Metrics each sample: panel tilt from horizontal, min lid<->box vertex distance (mm),
# lid verts inside the box AABB, lid AABB z-range, and lid centroid (explosion guard).
Results -> /tmp/lid_collision_probe.txt
"""
import os, sys
from pathlib import Path
import numpy as np

OUT = Path("/tmp/lid_collision_probe.txt")
sys.argv = [
    "run_scene.py",
    "--scene-context", "/home/ramtin/AHA/portable_scene_reports/close_box.scene_context.md",
    "--usd-dir", "/home/ramtin/IsaacLab/task_usds/close_box_physics",
    "--hide-root", "--no-robot", "--device", "cpu", "--headless",
]
import run_scene as R
import isaaclab.sim as sim_utils
from pxr import Usd, UsdPhysics

# Real physics: gravity on, no pin/servo. Joint collision comes from the JSON (now true).
R.CONTEXT.report.setdefault("physics", {}).setdefault("shapes", {}).setdefault("box_lid", {})["disable_gravity"] = False

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device="cpu"))
builder = R.SceneBuilder(R.args_cli, R.CONTEXT, R.APPEARANCE_CONFIG)
builder.design_scene()
R._add_articulation_joints(builder)
sim.reset()

lid_path = builder.body_prim_paths.get("box_lid")
box_path = builder.body_prim_paths.get("box_base")
view = R._wand_rigid_view(lid_path)
stage = sim_utils.get_current_stage()

# Confirm the joint's collisionEnabled actually came through as True.
joint_prim = None
for p in Usd.PrimRange(stage.GetPrimAtPath(box_path)):
    if p.GetName().endswith("Joint"):
        joint_prim = p
        break
coll_enabled = None
if joint_prim is not None:
    rj = UsdPhysics.RevoluteJoint(joint_prim)
    ce = rj.GetCollisionEnabledAttr()
    coll_enabled = bool(ce.Get()) if ce and ce.HasAuthoredValue() else None

box = np.array(R._mesh_world_points(box_path, max_pts=3000))
box_lo, box_hi = box.min(0), box.max(0)


def lid_pose():
    t = view.get_transforms()[0]
    t = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
    p = t[0:3].astype(float)
    qx, qy, qz, qw = (float(v) for v in t[3:7])
    Rm = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)]])
    return p, Rm


p0, Rm0 = lid_pose()
lid_w0 = np.array(R._mesh_world_points(lid_path, max_pts=800))
lid_local = (Rm0.T @ (lid_w0 - p0).T).T
cov = np.cov((lid_local - lid_local.mean(0)).T)
_, evecs = np.linalg.eigh(cov)
normal_local = evecs[:, 0]


def metrics():
    p, Rm = lid_pose()
    lw = (Rm @ lid_local.T).T + p
    n_world = Rm @ normal_local
    tilt = float(np.degrees(np.arccos(min(1.0, abs(n_world[2]) / (np.linalg.norm(n_world) + 1e-12)))))
    d = float(np.sqrt(((lw[:, None, :] - box[None, :, :]) ** 2).sum(-1)).min())
    inside = int(((lw >= box_lo + 0.002) & (lw <= box_hi - 0.002)).all(1).sum())
    return tilt, d, inside, float(lw[:, 2].min()), float(lw[:, 2].max()), p


lines = [
    f"joint collisionEnabled (read back from USD): {coll_enabled}",
    f"box AABB z=[{box_lo[2]:.3f},{box_hi[2]:.3f}] x=[{box_lo[0]:.3f},{box_hi[0]:.3f}] y=[{box_lo[1]:.3f},{box_hi[1]:.3f}]",
]


def log(tag, m):
    nan = "  *** NaN/EXPLODED ***" if not np.isfinite(m[5]).all() or abs(m[5][2]) > 5 else ""
    lines.append(f"{tag}: tilt={m[0]:5.1f}deg  min_lid_box_dist={m[1]*1000:6.1f}mm  "
                 f"verts_inside={m[2]:2d}  lid_z=[{m[3]:.3f},{m[4]:.3f}]  centroid_z={m[5][2]:.3f}{nan}")


lines.append("--- PHASE 1: OPEN-SETTLE (gravity only, collision on) ---")
log("spawn  ", metrics())
for i in range(150):
    sim.step()
    if i in (29, 79, 149):
        log(f"step{i:4d}", metrics())

# --- PHASE 2: drive the lid CLOSED (toward the lower / closing limit) ---
# The lid spawns at the upper limit (open); closing rotates it toward the lower limit.
# A velocity drive presses it down; with collision ON it must stall on the box rim.
drive = UsdPhysics.DriveAPI.Get(joint_prim, "angular")
if not drive:
    drive = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
drive.CreateDampingAttr(8.0)
drive.CreateMaxForceAttr(80.0)
drive.GetTargetVelocityAttr().Set(-25.0) if drive.GetTargetVelocityAttr() else drive.CreateTargetVelocityAttr(-25.0)
lines.append("--- PHASE 2: CLOSE (velocity drive -25 deg/s toward closing limit) ---")
for i in range(360):
    sim.step()
    if i in (29, 89, 179, 269, 359):
        log(f"step{i:4d}", metrics())

mlast = metrics()
seated = mlast[2] <= 1 and mlast[1] * 1000 < 5.0
stable = np.isfinite(mlast[5]).all() and abs(mlast[5][2]) < 5
lines.append("--- VERDICT ---")
lines.append(f"final tilt={mlast[0]:.1f}deg (closed≈0)  penetration_verts_inside={mlast[2]}  min_dist={mlast[1]*1000:.1f}mm")
lines.append(f"SEATED ON RIM (no sink-through): {seated}    STABLE (no jitter/explosion): {stable}")
OUT.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"[PROBE-COLLISION] wrote {OUT}")
os._exit(0)
