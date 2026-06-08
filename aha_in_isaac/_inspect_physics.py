"""Dump the baked physics of a task's USDs: per object, whether it has a RigidBody
(and if kinematic), its collider approximation, and any physics material friction."""
import sys, json
from pathlib import Path
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

TASK = sys.argv[1] if len(sys.argv) > 1 else "change_clock"
USD_DIR = Path(f"/home/ramtin/IsaacLab/task_usds/{TASK}_physics")
TD = Path("/home/ramtin/IsaacLab/scripts/aha_in_isaac/task_data")
phys = json.loads((TD / "physics" / f"{TASK}.json").read_text())
shapes = phys.get("shapes", {})
joints = phys.get("joints", {})
objs = json.loads((TD / "objects" / f"{TASK}.json").read_text())


def usd_path(name):
    for stem in (f"{TASK}_{name}", name):
        for ext in (".usd", ".usdc", ".usda"):
            p = USD_DIR / f"{stem}{ext}"
            if p.is_file():
                return p
    for p in sorted(USD_DIR.iterdir()):
        if p.suffix in (".usd", ".usdc", ".usda") and p.stem.endswith(f"_{name}"):
            return p
    return None


print(f"=== baked physics for {TASK} ===\n")
print(f"{'object':28s} {'real.dynamic':12s} {'baked body':22s} {'collider':18s} {'mat.friction'}")
for o in objs:
    name = o["name"]
    p = usd_path(name)
    real = shapes.get(name, {})
    if not p:
        print(f"{name:28s} {str(real.get('dynamic')):12s} NO_USD")
        continue
    stage = Usd.Stage.Open(str(p))
    body_desc, coll_desc, fric = "none", "none", "-"
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rb = UsdPhysics.RigidBodyAPI(prim)
            kin = rb.GetKinematicEnabledAttr().Get()
            body_desc = f"RigidBody(kinematic={kin})"
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            approx = "-"
            if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                approx = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
            coll_desc = f"coll(en={enabled},{approx})"
        if prim.IsA(UsdShade.Material) or prim.HasAPI(UsdPhysics.MaterialAPI):
            try:
                m = UsdPhysics.MaterialAPI(prim)
                sf = m.GetStaticFrictionAttr().Get()
                if sf is not None:
                    fric = f"s={sf}"
            except Exception:
                pass
    print(f"{name:28s} {str(real.get('dynamic')):12s} {body_desc:22s} {coll_desc:18s} {fric}")

print(f"\nreal joints: {json.dumps(joints)}")
