"""Classify every task's placement health into:
  OK            : all non-anchor objects land within 3cm of report.
  TRANSLATION   : non-anchor objects share ONE wrong offset (spread<2cm) -> the
                  task-root ANCHOR object is baked in a different frame than the
                  rest of the assembly (change_clock-style). A corrected anchor
                  reference fixes the whole task.
  ROTATION/MIXED: per-object errors fan out -> some objects are baked at a layout
                  the single task-root transform can't reproduce (independently
                  randomized instances not tagged graspable, or rotated baked frame).
"""
import sys, json
from pathlib import Path
from pxr import Usd, UsdGeom

sys.path.insert(0, "/home/ramtin/IsaacLab/scripts/aha_in_isaac")
from scene_context import (pose_from_location, pose_from_world_location, _qapply,
                           _subtract_pose, task_root_object)

REPO = Path("/home/ramtin/IsaacLab")
TD = REPO / "scripts/aha_in_isaac/task_data"
USD_ROOT = REPO / "task_usds"
_c = {}


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


def bbox_center(path):
    if str(path) in _c:
        return _c[str(path)]
    stage = Usd.Stage.Open(str(path))
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], useExtentsHint=True)
    b = cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedBox()
    cc = (b.GetMin() + b.GetMax()) * 0.5
    r = (float(cc[0]), float(cc[1]), float(cc[2]))
    _c[str(path)] = r
    return r


def grasp_snapped(task, obj_by_name):
    g = TD / "graspables" / f"{task}.json"
    s = set()
    if g.is_file():
        d = json.loads(g.read_text())
        if isinstance(d, dict):
            s |= {e.get("name") for e in (d.get("graspable_objects") or []) if e.get("name")}
    s |= {n for n, e in obj_by_name.items() if e.get("mounted_on_graspable")}
    return s


def classify(task):
    objs_path = TD / "objects" / f"{task}.json"
    if not objs_path.is_file():
        return None
    usd_dir = USD_ROOT / f"{task}_physics"
    if not usd_dir.is_dir():
        usd_dir = USD_ROOT / task
    if not usd_dir.is_dir():
        return None
    objects = json.loads(objs_path.read_text())
    obj_by_name = {o["name"]: o for o in objects}
    try:
        root_entry = task_root_object(obj_by_name, task)
    except Exception:
        return None
    anchor = root_entry["name"]
    if not root_entry.get("task_root_local_location"):
        return None
    rwp, rwq = pose_from_world_location(root_entry)
    rlp, rlq = pose_from_location(root_entry.get("task_root_local_location"))
    sampled_pos, sampled_quat = _subtract_pose(rwp, rwq, rlp, rlq)
    rp = usd_path(usd_dir, task, anchor)
    if not rp:
        return None
    canonical = tuple(bbox_center(rp)[i] - rlp[i] for i in range(3))
    child_t = tuple(-v for v in canonical)
    snapped = grasp_snapped(task, obj_by_name)
    deltas = []
    for name, entry in obj_by_name.items():
        if name == anchor or name in snapped:
            continue
        rep = entry.get("world_location", {}).get("position_xyz_m")
        p = usd_path(usd_dir, task, name)
        if not rep or not p:
            continue
        bc = bbox_center(p)
        local = tuple(child_t[i] + bc[i] for i in range(3))
        placed = tuple(sampled_pos[i] + _qapply(sampled_quat, local)[i] for i in range(3))
        deltas.append(tuple((placed[i]-rep[i])*100 for i in range(3)))
    if not deltas:
        return (task, anchor, "OK", 0.0, 0.0)
    n = len(deltas)
    maxerr = max(sum(d[i]**2 for i in range(3))**0.5 for d in deltas)
    mean = [sum(d[i] for d in deltas)/n for i in range(3)]
    spread = max((sum((d[i]-mean[i])**2 for i in range(3)))**0.5 for d in deltas)
    if maxerr < 3.0:
        cat = "OK"
    elif spread < 2.0:
        cat = "TRANSLATION"
    else:
        cat = "ROTATION/MIXED"
    return (task, anchor, cat, maxerr, spread)


tasks = sorted(p.stem for p in (TD / "objects").glob("*.json"))
rows = [r for t in tasks if (r := classify(t))]
order = {"TRANSLATION": 0, "ROTATION/MIXED": 1, "OK": 2}
rows.sort(key=lambda r: (order[r[2]], -r[3]))
cur = None
for task, anchor, cat, maxerr, spread in rows:
    if cat != cur:
        print(f"\n##### {cat} #####")
        cur = cat
    if cat != "OK":
        print(f"  {task:34s} anchor={anchor:22s} maxErr={maxerr:6.1f}cm spread={spread:6.1f}cm")
    else:
        print(f"  {task:34s} maxErr={maxerr:5.1f}cm")
from collections import Counter
print("\nsummary:", dict(Counter(r[2] for r in rows)))
