"""Prototype the hinge-edge collider TRIM. The lid is an 8-vertex box; its 2 hinge-edge
verts dip 8.3mm below the box top, inside the box footprint -> the spawn overlap that
PhysX depenetrates (the fling). Trim = pull those hinge-edge verts back in -y (away from
the box's -y wall) so the open lid clears the box, while the slab still covers the box top
when closed.

Env TRIM = metres to move the hinge-edge verts in -y (e.g. 0.05, 0.07).
collision ON, upper limit = 0 (rest at recorded open pose).
Phase A gravity settle (kick?), Phase B drive closed (seats on box?). -> /tmp/lid_trim_<TRIM>.txt
"""
import os, sys
from pathlib import Path
import numpy as np

TRIM = float(os.environ.get("TRIM", "0.06"))
OUT = Path(f"/tmp/lid_trim_{str(TRIM).replace('.', 'p')}.txt")
sys.argv = [
    "run_scene.py",
    "--scene-context", "/home/ramtin/AHA/portable_scene_reports/close_box.scene_context.md",
    "--usd-dir", "/home/ramtin/IsaacLab/task_usds/close_box_physics",
    "--hide-root", "--no-robot", "--device", "cpu", "--headless",
]
import run_scene as R
import isaaclab.sim as sim_utils
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Vt

R.CONTEXT.report.setdefault("physics", {}).setdefault("shapes", {}).setdefault("box_lid", {})["disable_gravity"] = False
j = R.CONTEXT.report["physics"].setdefault("joints", {}).setdefault("box_joint", {})
j["collision"] = True
j["lower"] = -180.0; j["upper"] = -27.5; j["position"] = -27.5  # USD-relative upper=0 (open rest)

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device="cpu"))
builder = R.SceneBuilder(R.args_cli, R.CONTEXT, R.APPEARANCE_CONFIG)
builder.design_scene()
lid_path = builder.body_prim_paths.get("box_lid")
box_path = builder.body_prim_paths.get("box_base")
stage = sim_utils.get_current_stage()

# box top, to identify lid hinge-edge verts (those below it = the overlap).
box = np.array(R._mesh_world_points(box_path, max_pts=4000))
box_top = box[:, 2].max()

# Find the lid mesh prim and trim its hinge-edge verts in -y (LOCAL == world here; identity xform).
lines = [f"TRIM={TRIM}m in -y on hinge-edge verts (verts below box_top={box_top:.4f})"]
trimmed = 0
for prim in Usd.PrimRange(stage.GetPrimAtPath(lid_path)):
    if not prim.IsA(UsdGeom.Mesh):
        continue
    mesh = UsdGeom.Mesh(prim)
    pts = mesh.GetPointsAttr().Get()
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    M = np.array([[m[i][j] for j in range(4)] for i in range(4)])
    P = np.array([[p[0], p[1], p[2]] for p in pts])
    W = (np.c_[P, np.ones(len(P))] @ M)[:, :3]
    newpts = list(pts)
    for k in range(len(P)):
        if W[k, 2] < box_top - 0.0005:   # a hinge-edge vert (dips below box top)
            newpts[k] = Gf.Vec3f(pts[k][0], pts[k][1] - TRIM, pts[k][2])
            trimmed += 1
    mesh.GetPointsAttr().Set(Vt.Vec3fArray(newpts))
lines.append(f"trimmed {trimmed} hinge-edge vert(s)")

R._add_articulation_joints(builder)
sim.reset()
view = R._wand_rigid_view(lid_path)
joint_prim = None
for p in Usd.PrimRange(stage.GetPrimAtPath(box_path)):
    if p.GetName().endswith("Joint"):
        joint_prim = p; break

# Re-sample lid points AFTER trim for metrics (uses the trimmed visual mesh; fine for physics check).
p0t = view.get_transforms()[0]
p0 = (p0t.detach().cpu().numpy() if hasattr(p0t, "detach") else np.asarray(p0t))[0:3].astype(float)
qx, qy, qz, qw = (float(v) for v in (p0t.detach().cpu().numpy() if hasattr(p0t, "detach") else np.asarray(p0t))[3:7])
Rm0 = np.array([[1-2*(qy*qy+qz*qz),2*(qx*qy-qw*qz),2*(qx*qz+qw*qy)],
                [2*(qx*qy+qw*qz),1-2*(qx*qx+qz*qz),2*(qy*qz-qw*qx)],
                [2*(qx*qz-qw*qy),2*(qy*qz+qw*qx),1-2*(qx*qx+qy*qy)]])
lid_w0 = np.array(R._mesh_world_points(lid_path, max_pts=800))
lid_local = (Rm0.T @ (lid_w0 - p0).T).T
cov = np.cov((lid_local - lid_local.mean(0)).T); _, evecs = np.linalg.eigh(cov); normal_local = evecs[:, 0]
box_lo, box_hi = box.min(0), box.max(0)


def pose():
    t = view.get_transforms()[0]
    t = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
    p = t[0:3].astype(float); qx, qy, qz, qw = (float(v) for v in t[3:7])
    Rm = np.array([[1-2*(qy*qy+qz*qz),2*(qx*qy-qw*qz),2*(qx*qz+qw*qy)],
                   [2*(qx*qy+qw*qz),1-2*(qx*qx+qz*qz),2*(qy*qz-qw*qx)],
                   [2*(qx*qz-qw*qy),2*(qy*qz+qw*qx),1-2*(qx*qx+qy*qy)]])
    return p, Rm


def metrics():
    p, Rm = pose(); lw = (Rm @ lid_local.T).T + p
    n = Rm @ normal_local
    tilt = float(np.degrees(np.arccos(min(1.0, abs(n[2]) / (np.linalg.norm(n)+1e-12)))))
    com = float(lw[:, 2].mean())
    d = float(np.sqrt(((lw[:, None, :]-box[None, :, :])**2).sum(-1)).min())
    inside = int(((lw >= box_lo+0.002) & (lw <= box_hi-0.002)).all(1).sum())
    return tilt, com, d, inside


t0 = metrics(); peak = t0[1]
lines.append(f"spawn      : tilt={t0[0]:5.1f}  CoM_z={t0[1]:.3f}  gap={t0[2]*1000:6.1f}mm  inside={t0[3]}")
for i in range(250):
    sim.step()
    if i in (49, 124, 249):
        m = metrics(); peak = max(peak, m[1])
ta = metrics(); peak_rise = peak - t0[1]
rest_clean = abs(ta[1]-t0[1]) < 0.005 and peak_rise < 0.010
lines.append(f"after-grav : tilt={ta[0]:5.1f}  CoM_z={ta[1]:.3f}  peak_CoM_rise={peak_rise*1000:.1f}mm  -> REST CLEAN: {rest_clean}")
# Strong POSITION drive to the closed angle: forces the lid over the gravitational
# apex (what the real arm does by following the recorded path), so we can verify the
# trimmed collider still SEATS the closed lid on the box top via real contact.
drive = UsdPhysics.DriveAPI.Get(joint_prim, "angular") or UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
drive.CreateStiffnessAttr(2000.0); drive.CreateDampingAttr(200.0); drive.CreateMaxForceAttr(100000.0)
(drive.GetTargetPositionAttr() or drive.CreateTargetPositionAttr(0.0)).Set(-150.0)  # USD-relative closed
for i in range(600):
    sim.step()
tb = metrics()
seated = tb[3] <= 1 and tb[2]*1000 < 6.0 and tb[0] < 15
lines.append(f"after-close: tilt={tb[0]:5.1f}  CoM_z={tb[1]:.3f}  gap={tb[2]*1000:6.1f}mm  inside={tb[3]}  -> SEATED ON BOX: {seated}")
lines.append("--- VERDICT ---")
lines.append(f"REST CLEAN (no fling): {rest_clean}   CLOSED SEATED ON BOX: {seated}")
OUT.write_text("\n".join(lines) + "\n")
print("\n".join(lines)); print(f"[TRIM] wrote {OUT}")
os._exit(0)
