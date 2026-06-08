"""Scan ALL tasks: reproduce task-root placement and report objects whose placed
bbox-center diverges from the report world_location (the uniform-shift signature of
the change_clock bug = anchor baked in a different frame than the rest)."""
import sys, json
from pathlib import Path
from pxr import Usd, UsdGeom, Gf

sys.path.insert(0, "/home/ramtin/IsaacLab/scripts/aha_in_isaac")
from scene_context import (pose_from_location, pose_from_world_location, _qapply,
                           _subtract_pose, task_root_object)

REPO = Path("/home/ramtin/IsaacLab")
TD = REPO / "scripts/aha_in_isaac/task_data"
USD_ROOT = REPO / "task_usds"


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


_cache = {}
def bbox_center(path):
    key = str(path)
    if key in _cache:
        return _cache[key]
    stage = Usd.Stage.Open(str(path))
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], useExtentsHint=True)
    box = cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedBox()
    c = (box.GetMin() + box.GetMax()) * 0.5
    r = (float(c[0]), float(c[1]), float(c[2]))
    _cache[key] = r
    return r


def graspable_names(task):
    g = TD / "graspables" / f"{task}.json"
    if not g.is_file():
        return set()
    d = json.loads(g.read_text())
    return {e.get("name") for e in (d.get("graspable_objects") or []) if e.get("name")}


def scan(task):
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
    rwp, rwq = pose_from_world_location(root_entry)
    rlp, rlq = pose_from_location(root_entry.get("task_root_local_location"))
    sampled_pos, sampled_quat = _subtract_pose(rwp, rwq, rlp, rlq)
    rp = usd_path(usd_dir, task, anchor)
    if not rp:
        return None
    root_center = bbox_center(rp)
    canonical = tuple(root_center[i] - rlp[i] for i in range(3))
    child_t = tuple(-v for v in canonical)
    grasp = graspable_names(task)
    mounted = {n for n, e in obj_by_name.items() if e.get("mounted_on_graspable")}
    snapped = grasp | mounted
    errs = []
    for name, entry in obj_by_name.items():
        if name == anchor or name in snapped:
            continue
        p = usd_path(usd_dir, task, name)
        rep = entry.get("world_location", {}).get("position_xyz_m")
        if not p or not rep:
            continue
        bc = bbox_center(p)
        local = tuple(child_t[i] + bc[i] for i in range(3))
        rot = _qapply(sampled_quat, local)
        placed = tuple(sampled_pos[i] + rot[i] for i in range(3))
        err = sum((placed[i]-rep[i])**2 for i in range(3))**0.5 * 100
        errs.append((err, name))
    if not errs:
        return (task, anchor, 0.0, 0, 0)
    errs.sort(reverse=True)
    maxerr = errs[0][0]
    nbad = sum(1 for e, _ in errs if e > 3.0)
    return (task, anchor, maxerr, nbad, len(errs))


tasks = sorted(p.stem for p in (TD / "objects").glob("*.json"))
rows = []
for t in tasks:
    try:
        r = scan(t)
    except Exception as e:
        r = (t, "ERR:"+str(e)[:40], -1, -1, -1)
    if r:
        rows.append(r)

rows.sort(key=lambda r: r[2], reverse=True)
print(f"{'task':28s} {'anchor':22s} {'maxErr(cm)':>10s} {'#>3cm':>6s} {'#chk':>5s}")
print("-"*78)
for task, anchor, maxerr, nbad, n in rows:
    flag = "  <<< BROKEN" if maxerr > 3 else ""
    print(f"{task:28s} {anchor:22s} {maxerr:10.2f} {nbad:6d} {n:5d}{flag}")
