"""Verify every non-excluded task's BAKED USDs now carry the authoritative body type:
  rigid     -> a non-kinematic RigidBody is present
  kinematic -> a kinematic RigidBody is present
  visual    -> NO RigidBody (pure decoration)
Reports any mismatches per task. Offline (pxr only)."""
import json, sys
from pathlib import Path
from pxr import Usd, UsdPhysics

REPO = Path("/home/ramtin/IsaacLab")
TD = REPO / "scripts/aha_in_isaac/task_data"
USD_ROOT = REPO / "task_usds"
EXCLUDE = {"basketball_in_hoop", "beat_the_buzz", "change_channel",
           "close_box", "wipe_desk", "weighing_scales"}


def is_visual_name(n):
    return "_vis" in n.lower()


def derive_type(name, shape):
    if is_visual_name(name):
        return "visual"
    dyn = shape.get("dynamic")
    if dyn is None:
        return None
    if dyn:
        return "rigid"
    if shape.get("respondable") or shape.get("collidable"):
        return "kinematic"
    return "visual"


def usd_path(usd_dir, task, name):
    for stem in (f"{task}_{name}", name):
        for ext in (".usd", ".usdc", ".usda"):
            p = usd_dir / f"{stem}{ext}"
            if p.is_file():
                return p
    if usd_dir.is_dir():
        for p in sorted(usd_dir.iterdir()):
            if p.suffix in (".usd", ".usdc", ".usda") and p.stem.endswith(f"_{name}"):
                return p
    return None


def baked_kind(path):
    stage = Usd.Stage.Open(str(path))
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            kin = UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get()
            return "kinematic" if kin else "rigid"
    return "visual"  # no rigid body


tasks = sorted(p.stem for p in (TD / "objects").glob("*.json"))
ok_tasks, bad = [], {}
for task in tasks:
    if task in EXCLUDE:
        continue
    phys = json.loads((TD / "physics" / f"{task}.json").read_text()) if (TD / "physics" / f"{task}.json").is_file() else None
    if not isinstance(phys, dict) or not phys.get("shapes"):
        continue
    usd_dir = USD_ROOT / f"{task}_physics"
    if not usd_dir.is_dir():
        continue
    mism = []
    for name, shape in phys["shapes"].items():
        want = derive_type(name, shape)
        if want is None:
            continue
        p = usd_path(usd_dir, task, name)
        if not p:
            continue
        got = baked_kind(p)
        # "visual" want allows baked "visual" (no body). kinematic/rigid must match.
        if got != want:
            mism.append((name, want, got))
    if mism:
        bad[task] = mism
    else:
        ok_tasks.append(task)

print(f"OK (all objects match authoritative type): {len(ok_tasks)} tasks")
print(f"MISMATCH: {len(bad)} tasks")
for task, mism in bad.items():
    print(f"\n  {task}:")
    for name, want, got in mism[:20]:
        print(f"      {name:30s} want={want:10s} baked={got}")
