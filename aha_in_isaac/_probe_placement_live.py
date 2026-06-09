"""In-sim placement probe: build a task's scene with the REAL SceneBuilder, measure
each VISIBLE object's actual spawned world-bbox centre, compare to the report
world_location, write a JSON result to a file, then close cleanly (no infinite loop,
results survive app.close() because they are flushed to disk first).

Usage (inside isaaclab.sh -p):
  _probe_placement_live.py --scene-context <ctx.md> --usd-dir <task_physics> \
      --hide-root --no-robot --headless --probe-out /tmp/<task>_verify.json
"""
import sys, json
from pathlib import Path

ISAACLAB_ROOT = Path("/home/ramtin/IsaacLab")
sys.path.insert(0, str(ISAACLAB_ROOT / "scripts/aha_in_isaac"))
for _pkg in ("isaaclab", "isaaclab_assets", "isaaclab_tasks"):
    _p = ISAACLAB_ROOT / "source" / _pkg
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from isaaclab.app import AppLauncher
from appearance_config import load_appearance_config
from cli import build_parser
from scene_context import SceneContext

parser = build_parser()
parser.add_argument("--probe-out", type=Path, default=Path("/tmp/placement_verify.json"))
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
OUT = args_cli.probe_out

CONTEXT = SceneContext.load(args_cli)
APPEARANCE = load_appearance_config(args_cli.appearance_config, CONTEXT.task_name)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402
from scene_builder import SceneBuilder  # noqa: E402

result = {"task": CONTEXT.task_name, "objects": [], "error": None}
try:
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    builder = SceneBuilder(args_cli, CONTEXT, APPEARANCE)
    builder.canonical_used = None
    builder.design_scene()
    sim.reset()

    stage = sim_utils.get_current_stage()
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])

    # Measure the actual spawned world centre of every spawned body + visible skin.
    measured = {}
    for name, path in list(builder.body_prim_paths.items()) + list(builder.skin_prim_paths.items()):
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            continue
        rng = bbox.ComputeWorldBound(prim).ComputeAlignedRange()
        if rng.IsEmpty():
            continue
        c = rng.GetMidpoint()
        measured[name] = (float(c[0]), float(c[1]), float(c[2]))

    for name, entry in CONTEXT.objects.items():
        rep = (entry.get("world_location") or {}).get("position_xyz_m")
        if name not in measured or not rep:
            continue
        m = measured[name]
        err_cm = sum((m[i] - float(rep[i])) ** 2 for i in range(3)) ** 0.5 * 100.0
        result["objects"].append({
            "name": name,
            "measured_world_center": [round(v, 4) for v in m],
            "report_world_pos": [round(float(v), 4) for v in rep],
            "err_cm": round(err_cm, 2),
        })
except Exception as exc:  # noqa: BLE001
    import traceback
    result["error"] = traceback.format_exc()

result["objects"].sort(key=lambda r: r["err_cm"], reverse=True)
result["max_err_cm"] = max((r["err_cm"] for r in result["objects"]), default=None)
OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(f"[PROBE]: wrote {OUT} (max_err_cm={result['max_err_cm']})")
sys.stdout.flush()

simulation_app.close()
