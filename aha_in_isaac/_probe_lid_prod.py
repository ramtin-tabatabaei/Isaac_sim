"""Verify the PRODUCTION fix path: run_scene's own _add_articulation_joints +
_trim_lid_hinge_collider (driven by close_box.json: collision=true,
hinge_collision_trim_m=0.09). Confirms:
  (1) the visual lid mesh is UNCHANGED (still has hinge verts below box top),
  (2) a hidden HingeTrimCollider exists and the visual collider is disabled,
  (3) lid RESTS at open (no fling), and (4) closes & SEATS on the box.
Results -> /tmp/lid_prod.txt
"""
import os, sys
from pathlib import Path

OUT = Path("/tmp/lid_prod.txt")
sys.argv = [
    "run_scene.py",
    "--scene-context", "/home/ramtin/AHA/portable_scene_reports/close_box.scene_context.md",
    "--usd-dir", "/home/ramtin/IsaacLab/task_usds/close_box_physics",
    "--hide-root", "--no-robot", "--device", "cpu", "--headless",
]
import run_scene as R
import isaaclab.sim as sim_utils
from pxr import Usd, UsdGeom, UsdPhysics
import math

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device="cpu"))
builder = R.SceneBuilder(R.args_cli, R.CONTEXT, R.APPEARANCE_CONFIG)
builder.design_scene()
R._add_articulation_joints(builder)
R._trim_lid_hinge_collider(builder)        # <-- the real production function

lid_path = builder.body_prim_paths.get("box_lid")
box_path = builder.body_prim_paths.get("box_base")
stage = sim_utils.get_current_stage()
box_pts = R._mesh_world_points(box_path, max_pts=4000)
box_top = max(p[2] for p in box_pts)

# --- Baked collider approximations (confirm the re-bake) ---
lines = []
lines.append("=== BAKED COLLIDERS ===")
for label, path in (("box_base", box_path), ("box_lid", lid_path)):
    for prim in Usd.PrimRange(stage.GetPrimAtPath(path)):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        approx = None
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            approx = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
        co = None
        from pxr import PhysxSchema as _Px
        if prim.HasAPI(_Px.PhysxCollisionAPI):
            a = _Px.PhysxCollisionAPI(prim).GetContactOffsetAttr()
            co = round(float(a.Get()), 4) if a.HasAuthoredValue() else None
        ce = None
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            cea = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr()
            ce = bool(cea.Get()) if cea.HasAuthoredValue() else True
        lines.append(f"  {label}/{prim.GetName():22} approx={approx}  contact_offset={co}  collisionEnabled={ce}")

# --- Structural checks (visual intact, trimmed collider present) ---
visual_below = trim_present = visual_coll_disabled = 0
visual_mesh_pts = None
for prim in Usd.PrimRange(stage.GetPrimAtPath(lid_path)):
    if not prim.IsA(UsdGeom.Mesh):
        continue
    nm = prim.GetName()
    if nm == "HingeTrimCollider":
        trim_present = 1
        continue
    # this is the visual lid mesh
    xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    pts = prim.GetAttribute("points").Get()
    from pxr import Gf
    wz = [xf.Transform(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))[2] for p in pts]
    visual_below = sum(1 for z in wz if z < box_top - 0.0005)
    if prim.HasAPI(UsdPhysics.CollisionAPI):
        ce = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr()
        visual_coll_disabled = 1 if (ce.HasAuthoredValue() and ce.Get() is False) else 0
lines.append("=== STRUCTURE ===")
lines.append(f"  visual lid mesh hinge verts still below box top (intact): {visual_below} (expect 2)")
lines.append(f"  hidden HingeTrimCollider present: {bool(trim_present)}")
lines.append(f"  visual mesh collider disabled: {bool(visual_coll_disabled)}")

sim.reset()
import numpy as np
view = R._wand_rigid_view(lid_path)
joint_prim = None
for p in Usd.PrimRange(stage.GetPrimAtPath(box_path)):
    if p.GetName().endswith("Joint"):
        joint_prim = p
        break

box = np.array(box_pts)
box_lo, box_hi = box.min(0), box.max(0)


def pose():
    t = view.get_transforms()[0]
    t = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
    p = t[0:3].astype(float); qx, qy, qz, qw = (float(v) for v in t[3:7])
    Rm = np.array([[1-2*(qy*qy+qz*qz),2*(qx*qy-qw*qz),2*(qx*qz+qw*qy)],
                   [2*(qx*qy+qw*qz),1-2*(qx*qx+qz*qz),2*(qy*qz-qw*qx)],
                   [2*(qx*qz-qw*qy),2*(qy*qz+qw*qx),1-2*(qx*qx+qy*qy)]])
    return p, Rm


p0, Rm0 = pose()
lid_w0 = np.array(R._mesh_world_points(lid_path, max_pts=800))
lid_local = (Rm0.T @ (lid_w0 - p0).T).T
cov = np.cov((lid_local - lid_local.mean(0)).T); _, evecs = np.linalg.eigh(cov); normal_local = evecs[:, 0]


def metrics():
    p, Rm = pose(); lw = (Rm @ lid_local.T).T + p
    n = Rm @ normal_local
    tilt = float(np.degrees(np.arccos(min(1.0, abs(n[2]) / (np.linalg.norm(n)+1e-12)))))
    com = float(lw[:, 2].mean())
    d = float(np.sqrt(((lw[:, None, :]-box[None, :, :])**2).sum(-1)).min())
    inside = int(((lw >= box_lo+0.002) & (lw <= box_hi-0.002)).all(1).sum())
    return tilt, com, d, inside


t0 = metrics(); peak = t0[1]
lines.append("=== DYNAMICS (PCA panel-normal tilt) ===")
lines.append(f"  spawn: tilt={t0[0]:.1f}  CoM_z={t0[1]:.3f}")
for i in range(250):
    sim.step(); m = metrics(); peak = max(peak, m[1])
ta = metrics()
rest_clean = abs(ta[1]-t0[1]) < 0.005 and (peak-t0[1]) < 0.010
lines.append(f"  after-grav: tilt={ta[0]:.1f}  CoM_z={ta[1]:.3f}  peak_CoM_rise={(peak-t0[1])*1000:.1f}mm  REST CLEAN={rest_clean}")

# MODERATE position drive to closed (forces over the apex like the arm, without an explosive force)
drive = UsdPhysics.DriveAPI.Get(joint_prim, "angular") or UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
drive.CreateStiffnessAttr(300.0); drive.CreateDampingAttr(60.0); drive.CreateMaxForceAttr(1500.0)
(drive.GetTargetPositionAttr() or drive.CreateTargetPositionAttr(0.0)).Set(-150.0)
for i in range(700):
    sim.step()
tb = metrics()
seated = tb[0] < 15 and tb[3] <= 1 and tb[2]*1000 < 8.0
lines.append(f"  after-close: tilt={tb[0]:.1f}  CoM_z={tb[1]:.3f}  gap={tb[2]*1000:.1f}mm  inside={tb[3]}  SEATED={seated}")
lines.append("=== VERDICT ===")
lines.append(f"  VISUAL INTACT={visual_below == 2 and bool(trim_present) and bool(visual_coll_disabled)}  "
             f"REST CLEAN={rest_clean}  CLOSED SEATED ON BOX={seated}")
OUT.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"[PROD] wrote {OUT}")
os._exit(0)
