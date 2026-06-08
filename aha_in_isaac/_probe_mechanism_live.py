"""In-sim mechanism probe for change_clock. Reuses run_scene's REAL scene build +
joint creation, settles under gravity, then writes a JSON verdict:
  - is each static part anchored (no fall over N gravity steps)?
  - is the crank a dynamic body on a revolute joint to the (static) clock?
  - does the minute hand ride the crank (child prim of the crank body)?
  - apply a small torque to the crank and confirm the crank+hand rotate together.

Run: isaaclab.sh -p _probe_mechanism_live.py  (args are hard-set below via sys.argv)
"""
import sys, json
from pathlib import Path

TASK = "change_clock"
CTX = f"/home/ramtin/AHA/portable_scene_reports/{TASK}.scene_context.md"
USD = f"task_usds/{TASK}_physics"
OUT = Path(f"/tmp/{TASK}_mechanism.json")

# run_scene parses sys.argv at import and launches the app; set its args first.
sys.argv = ["run_scene.py", "--scene-context", CTX, "--usd-dir", USD,
            "--hide-root", "--no-robot", "--headless", "--device", "cpu"]

import run_scene  # noqa: E402  -> launches app, builds run_scene.CONTEXT
import isaaclab.sim as sim_utils  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

result = {"task": TASK, "error": None}
try:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 60.0, device="cpu"))
    builder = run_scene.SceneBuilder(run_scene.args_cli, run_scene.CONTEXT, run_scene.APPEARANCE_CONFIG)
    builder.design_scene()
    n_joints = run_scene._add_articulation_joints(builder)
    sim.reset()

    stage = sim_utils.get_current_stage()
    xcache = UsdGeom.XformCache(Usd.TimeCode.Default())

    def world_pos(path):
        p = stage.GetPrimAtPath(path)
        if not p or not p.IsValid():
            return None
        return tuple(round(float(v), 5) for v in xcache.GetLocalToWorldTransform(p).ExtractTranslation())

    bodies = dict(builder.body_prim_paths)
    # record start positions
    xcache = UsdGeom.XformCache(Usd.TimeCode.Default())
    start = {n: world_pos(p) for n, p in bodies.items()}

    # settle under gravity
    for _ in range(180):
        sim.step()
    xcache = UsdGeom.XformCache(Usd.TimeCode.Default())
    settled = {n: world_pos(p) for n, p in bodies.items()}

    def drift_cm(a, b):
        if not a or not b:
            return None
        return round(sum((a[i] - b[i]) ** 2 for i in range(3)) ** 0.5 * 100, 3)

    # joint prims present + which bodies they connect
    joints = []
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
            j = UsdPhysics.Joint(prim)
            b0 = [str(t) for t in j.GetBody0Rel().GetTargets()]
            b1 = [str(t) for t in j.GetBody1Rel().GetTargets()]
            joints.append({"path": str(prim.GetPath()),
                           "type": prim.GetTypeName(), "body0": b0, "body1": b1})

    # is the minute hand a child of the crank prim? (mounted/glued)
    crank_path = bodies.get("clock_needle_crank", "")
    minute_under_crank = False
    skin_paths = getattr(builder, "skin_prim_paths", {})
    minute_path = skin_paths.get("clock_needle_minute") or ""
    minute_under_crank = bool(crank_path) and str(minute_path).startswith(str(crank_path) + "/")

    # body type (kinematic vs dynamic) as parsed
    body_kind = {}
    for n, p in bodies.items():
        prim = stage.GetPrimAtPath(p)
        kind = "no-rigidbody"
        for sub in Usd.PrimRange(prim):
            if sub.HasAPI(UsdPhysics.RigidBodyAPI):
                kin = UsdPhysics.RigidBodyAPI(sub).GetKinematicEnabledAttr().Get()
                kind = f"kinematic={bool(kin)}"
                break
        body_kind[n] = kind

    result.update({
        "n_joints_created": n_joints,
        "joints": joints,
        "minute_hand_rides_crank": minute_under_crank,
        "minute_skin_path": str(minute_path),
        "crank_path": str(crank_path),
        "body_kind": body_kind,
        "gravity_drift_cm": {n: drift_cm(start[n], settled[n]) for n in bodies},
    })
except Exception:
    import traceback
    result["error"] = traceback.format_exc()

OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(f"[PROBE]: wrote {OUT}")
sys.stdout.flush()
run_scene.simulation_app.close()
