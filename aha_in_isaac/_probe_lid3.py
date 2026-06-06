"""Probe v3: spatial truth of the close_box lid at spawn vs after gravity settle.

Reports, at step 0 (recorded open pose) and after 300 steps (gravity, no pin/servo):
  - lid panel TILT from horizontal (panel-normal angle off world +Z): ~0 = lid lying
    flat/open, ~90 = lid standing vertical.
  - min vertex distance lid<->box, and # lid verts sitting INSIDE the box AABB
    (a proxy for spawn interpenetration that the depenetration impulse resolves).
  - lid AABB z-range.
Results -> /tmp/lid_probe3.txt
"""
import os, sys
from pathlib import Path
import numpy as np

OUT = Path("/tmp/lid_probe3.txt")
sys.argv = [
    "run_scene.py",
    "--scene-context", "/home/ramtin/AHA/portable_scene_reports/close_box.scene_context.md",
    "--usd-dir", "/home/ramtin/IsaacLab/task_usds/close_box_physics",
    "--hide-root", "--no-robot", "--device", "cpu", "--headless",
]
import run_scene as R
import isaaclab.sim as sim_utils

# normal physics: gravity on, no pin, no servo
R.CONTEXT.report.setdefault("physics", {}).setdefault("shapes", {}).setdefault("box_lid", {})["disable_gravity"] = False
jc = R.MOTION_CONFIG.setdefault("joints", {}).setdefault("box_lid", {})
for k in ("close_at_waypoint", "close_speed", "close_damping"):
    jc.pop(k, None)

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0/120.0, device="cpu"))
builder = R.SceneBuilder(R.args_cli, R.CONTEXT, R.APPEARANCE_CONFIG)
builder.design_scene()
R._add_articulation_joints(builder)
sim.reset()

lid_path = builder.body_prim_paths.get("box_lid")
box_path = builder.body_prim_paths.get("box_base")
view = R._wand_rigid_view(lid_path)

# static box verts (kinematic, world space, fixed)
box = np.array(R._mesh_world_points(box_path, max_pts=3000))
box_lo, box_hi = box.min(0), box.max(0)

# lid body-frame collider points captured at spawn so we can re-place them each step
def lid_pose():
    t = view.get_transforms()[0]
    t = t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
    p = t[0:3].astype(float)
    qx, qy, qz, qw = (float(v) for v in t[3:7])
    Rm = np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
        [2*(qx*qy+qw*qz), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx*qx+qy*qy)]])
    return p, Rm

p0, Rm0 = lid_pose()
lid_w0 = np.array(R._mesh_world_points(lid_path, max_pts=800))
lid_local = (Rm0.T @ (lid_w0 - p0).T).T
# panel normal = smallest-variance PCA axis of the lid verts (panel thickness dir), body frame
cov = np.cov((lid_local - lid_local.mean(0)).T)
evals, evecs = np.linalg.eigh(cov)
normal_local = evecs[:, 0]  # smallest eigenvalue

def metrics():
    p, Rm = lid_pose()
    lw = (Rm @ lid_local.T).T + p
    # tilt of panel normal from world +Z
    n_world = Rm @ normal_local
    tilt = float(np.degrees(np.arccos(min(1.0, abs(n_world[2]) / (np.linalg.norm(n_world)+1e-12)))))
    # min vertex distance lid<->box
    d = float(np.sqrt(((lw[:, None, :] - box[None, :, :])**2).sum(-1)).min())
    inside = ((lw >= box_lo+0.002) & (lw <= box_hi-0.002)).all(1).sum()
    return tilt, d, int(inside), float(lw[:,2].min()), float(lw[:,2].max())

lines = [f"box AABB z=[{box_lo[2]:.3f},{box_hi[2]:.3f}]  x=[{box_lo[0]:.3f},{box_hi[0]:.3f}] y=[{box_lo[1]:.3f},{box_hi[1]:.3f}]"]
t0 = metrics()
lines.append(f"SPAWN (recorded open): panel_tilt_from_horizontal={t0[0]:.1f}deg  min_lid_box_dist={t0[1]*1000:.1f}mm  lid_verts_inside_box={t0[2]}  lid_z=[{t0[3]:.3f},{t0[4]:.3f}]")
for i in range(300):
    sim.step()
    if i in (29, 119, 299):
        t = metrics()
        lines.append(f"step {i:3d}: panel_tilt={t[0]:.1f}deg  min_lid_box_dist={t[1]*1000:.1f}mm  inside={t[2]}  lid_z=[{t[3]:.3f},{t[4]:.3f}]")
lines.append("INTERPRETATION: tilt ~0=lid flat/open, ~90=vertical. dist~0 & inside=0 => resting in contact (no penetration).")
OUT.write_text("\n".join(lines)+"\n")
print("\n".join(lines))
print(f"[PROBE3] wrote {OUT}")
os._exit(0)
