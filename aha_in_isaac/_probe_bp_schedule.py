"""Print the block_pyramid gripper schedule produced by build_arm_motion with the new
pick_place_sequence mode (expect 6 CLOSE + 6 OPEN events). -> /tmp/bp_schedule.txt"""
import os, sys
from pathlib import Path
sys.argv = [
    "run_scene.py",
    "--scene-context", "/home/ramtin/AHA/portable_scene_reports/block_pyramid.scene_context.md",
    "--usd-dir", "/home/ramtin/IsaacLab/task_usds/block_pyramid_physics",
    "--hide-root", "--no-robot", "--device", "cpu", "--headless",
]
import run_scene as R
from arm_motion import build_arm_motion

motion = build_arm_motion(
    R.CONTEXT.waypoints, R.MOTION_CONFIG,
    graspable_name=R._graspable_object_name(),
    graspable_names=R._dynamic_body_names(),
    curvy=False,
)
lines = [
    f"pick_place_sequence = {R.MOTION_CONFIG.get('pick_place_sequence')}",
    f"graspable (dynamic) bodies: {sorted(R._dynamic_body_names())}",
    f"motion steps: {len(motion)}",
    "schedule (only grip CHANGES shown):",
]
last = None; closes = 0; opens = 0
for w in motion:
    g = getattr(w, "gripper", "?")
    if g != last:
        if g == "closed":
            closes += 1
        if g == "open":
            opens += 1
        lines.append(f"  -> {str(g):7} {getattr(w, 'label', '?')}")
    last = g
lines.append(f"\nCLOSE events: {closes}   OPEN events: {opens}   (expect 6 and 6)")
Path("/tmp/bp_schedule.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
os._exit(0)
