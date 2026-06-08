"""Headless pose-dump: build a task scene exactly like run_scene.py and print where
each object actually lands (world position + yaw), compared to the report. Lets us
verify object placement without the GUI or the robot motion.

Usage (from ~/IsaacLab):
    ./isaaclab.sh -p scripts/aha_in_isaac/_probe_poses.py \
        --scene-context /home/ramtin/AHA/portable_scene_reports/change_channel.scene_context.md \
        --usd-dir task_usds/change_channel_physics --hide-root --no-robot --headless
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
ISAACLAB_ROOT = Path(__file__).resolve().parents[2]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
for _pkg in ("isaaclab", "isaaclab_assets", "isaaclab_tasks"):
    _p = ISAACLAB_ROOT / "source" / _pkg
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from isaaclab.app import AppLauncher  # noqa: E402
from cli import build_parser  # noqa: E402
from scene_context import SceneContext  # noqa: E402

parser = build_parser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
CONTEXT = SceneContext.load(args_cli)
APPEARANCE_CONFIG = (
    json.loads(args_cli.appearance_config.read_text(encoding="utf-8"))
    if args_cli.appearance_config.is_file() else {}
)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
from scene_builder import SceneBuilder, _prim_name  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402


def _yaw_from_quat_wxyz(w, x, y, z):
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


OUT = open("/tmp/probe_result.txt", "w", encoding="utf-8")


def emit(line=""):
    OUT.write(str(line) + "\n")
    OUT.flush()


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    builder = SceneBuilder(args_cli, CONTEXT, APPEARANCE_CONFIG)
    builder.design_scene()  # spawns objects + runs the graspable/mounted snap
    sim.reset()

    stage = sim_utils.get_current_stage()
    xcache = UsdGeom.XformCache(Usd.TimeCode.Default())
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])

    report_objs = {e["name"]: e for e in CONTEXT.report["objects"]}
    # Auto-pick what to inspect for ANY task: the graspables and the objects mounted on
    # them (the assembly the fix targets), plus the first fixed-to-root object as a
    # reference. Capped so big tasks (setup_chess) stay readable.
    graspables = sorted(CONTEXT.graspable_names)
    mounted = [e["name"] for e in CONTEXT.report["objects"] if e.get("mounted_on_graspable")]
    fixed = [
        e["name"] for e in CONTEXT.report["objects"]
        if not e.get("mounted_on_graspable") and e["name"] not in CONTEXT.graspable_names
    ]
    interesting = (graspables + mounted)[:14] + fixed[:1]

    emit("==== PROBE: actual scene pose vs report ====")
    emit(f"{'object':26s} {'actual_xy':>22s} {'actYaw':>8s} | {'report_xy':>22s} {'repYaw':>8s} | {'dxy(cm)':>8s}")
    parent_path = "/World/DesignScene/TaskRoot"
    for name in interesting:
        path = f"{parent_path}/{_prim_name(name)}"
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            # fall back: object may live directly under DesignScene
            path = f"/World/DesignScene/{_prim_name(name)}"
            prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            emit(f"{name:26s}  <prim not found at {path}>")
            continue
        m = xcache.GetLocalToWorldTransform(prim)
        t = m.ExtractTranslation()
        q = m.ExtractRotationQuat()  # Gf.Quatd, real + imaginary
        w = q.GetReal(); im = q.GetImaginary()
        act_yaw = _yaw_from_quat_wxyz(w, im[0], im[1], im[2])
        rng = bbox.ComputeWorldBound(prim).ComputeAlignedRange()
        cen = rng.GetMidpoint() if not rng.IsEmpty() else t
        e = report_objs.get(name, {})
        wl = e.get("world_location", {})
        rp = wl.get("position_xyz_m", [0, 0, 0])
        rq = wl.get("quaternion_xyzw", [0, 0, 0, 1])
        rep_yaw = _yaw_from_quat_wxyz(rq[3], rq[0], rq[1], rq[2])
        dxy = math.hypot(cen[0] - rp[0], cen[1] - rp[1]) * 100.0
        emit(f"{name:26s} ({cen[0]:+.3f},{cen[1]:+.3f},{cen[2]:+.3f}) {act_yaw:+.3f} | "
             f"({rp[0]:+.3f},{rp[1]:+.3f},{rp[2]:+.3f}) {rep_yaw:+.3f} | {dxy:7.2f}")
    emit("==== END PROBE ====")
    emit("dxy = horizontal distance of object centroid from its report position (cm).")
    emit("For a correct assembly: tv_remote + all buttons share ~the same actYaw==repYaw and small dxy.")


try:
    main()
except Exception:
    import traceback
    emit("PROBE EXCEPTION:\n" + traceback.format_exc())
finally:
    OUT.flush()
    OUT.close()
    simulation_app.close()
