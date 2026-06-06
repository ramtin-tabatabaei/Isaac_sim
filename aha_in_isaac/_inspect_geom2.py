"""Geometry inspection inside a booted Isaac scene: dump lid/box mesh world points
stats + the hinge-edge overlap, to design a collider trim. Results -> /tmp/close_box_geom.txt
"""
import os, sys
from pathlib import Path
import numpy as np

OUT = Path("/tmp/close_box_geom.txt")
sys.argv = [
    "run_scene.py",
    "--scene-context", "/home/ramtin/AHA/portable_scene_reports/close_box.scene_context.md",
    "--usd-dir", "/home/ramtin/IsaacLab/task_usds/close_box_physics",
    "--hide-root", "--no-robot", "--device", "cpu", "--headless",
]
import run_scene as R
import isaaclab.sim as sim_utils

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device="cpu"))
builder = R.SceneBuilder(R.args_cli, R.CONTEXT, R.APPEARANCE_CONFIG)
builder.design_scene()
lid_path = builder.body_prim_paths.get("box_lid")
box_path = builder.body_prim_paths.get("box_base")

lid = np.array(R._mesh_world_points(lid_path, max_pts=4000))
box = np.array(R._mesh_world_points(box_path, max_pts=4000))
lo_l, hi_l = lid.min(0), lid.max(0)
lo_b, hi_b = box.min(0), box.max(0)
box_top = hi_b[2]

lines = [
    "=== LID (box_lid) world geometry at spawn (open pose) ===",
    f"  nverts={len(lid)}  AABB x=[{lo_l[0]:.4f},{hi_l[0]:.4f}] y=[{lo_l[1]:.4f},{hi_l[1]:.4f}] z=[{lo_l[2]:.4f},{hi_l[2]:.4f}]",
    f"  extent dx={hi_l[0]-lo_l[0]:.4f} dy={hi_l[1]-lo_l[1]:.4f} dz={hi_l[2]-lo_l[2]:.4f}  centroid=({lid[:,0].mean():.4f},{lid[:,1].mean():.4f},{lid[:,2].mean():.4f})",
    "=== BOX (box_base) world geometry ===",
    f"  nverts={len(box)}  AABB x=[{lo_b[0]:.4f},{hi_b[0]:.4f}] y=[{lo_b[1]:.4f},{hi_b[1]:.4f}] z=[{lo_b[2]:.4f},{hi_b[2]:.4f}]   box_top_z={box_top:.4f}",
]

# Lid verts dipping below the box top (the hinge-edge overlap region)
below = lid[lid[:, 2] < box_top]
lines.append("=== HINGE-EDGE OVERLAP: lid verts below box top ===")
lines.append(f"  count={len(below)}/{len(lid)}")
if len(below):
    lines.append(f"  z=[{below[:,2].min():.4f},{below[:,2].max():.4f}]  max_depth_below_top={(box_top-below[:,2].min())*1000:.1f}mm")
    lines.append(f"  x=[{below[:,0].min():.4f},{below[:,0].max():.4f}]  y=[{below[:,1].min():.4f},{below[:,1].max():.4f}]")
    # Are these below-top verts within the box XY footprint (truly inside the wall) or outside it?
    inside_xy = below[(below[:,0]>=lo_b[0]) & (below[:,0]<=hi_b[0]) & (below[:,1]>=lo_b[1]) & (below[:,1]<=hi_b[1])]
    lines.append(f"  of those, within box XY footprint (truly overlapping the wall): {len(inside_xy)}")

# Lid lowest verts = the hinge edge; how far is the hinge edge from the rest of the panel?
order = np.argsort(lid[:, 2])
hinge_edge = lid[order[:40]]
lines.append("=== LID HINGE EDGE (lowest 40 verts) ===")
lines.append(f"  z=[{hinge_edge[:,2].min():.4f},{hinge_edge[:,2].max():.4f}] centroid=({hinge_edge[:,0].mean():.4f},{hinge_edge[:,1].mean():.4f},{hinge_edge[:,2].mean():.4f})")
lines.append(f"  hinge-edge y-centroid={hinge_edge[:,1].mean():.4f} vs lid y-centroid={lid[:,1].mean():.4f}  (which side is the hinge)")
# distance from box top to lid free edge (highest verts)
free_edge = lid[order[-40:]]
lines.append(f"  free edge (highest 40) centroid=({free_edge[:,0].mean():.4f},{free_edge[:,1].mean():.4f},{free_edge[:,2].mean():.4f})")

OUT.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"[GEOM2] wrote {OUT}")
os._exit(0)
