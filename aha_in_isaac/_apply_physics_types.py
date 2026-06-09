"""Apply authoritative-physics body types to non-excluded tasks, surgically.

For each task (except the 6 curated/excluded):
  * object_physics/<task>.json : correct ONLY the "type" of objects whose current type
    disagrees with the type derived from the authoritative CoppeliaSim physics
    (dynamic->rigid, static-collidable->kinematic, non-colliding->visual). All other
    fields (collider, density, disable_gravity, ccd, contact_offset, uv, ...) are kept.
  * physics/<task>.json : add "mount_on_parent": true to a non-colliding static shape
    whose parent is a DYNAMIC body, so it rides that body (e.g. a clock hand on a crank).

Backups of every touched file are written under /tmp/physics_apply_backup/<...>.
Pass --dry to print without writing.
"""
import json, sys, shutil
from pathlib import Path

DRY = "--dry" in sys.argv
TD = Path("/home/ramtin/IsaacLab/scripts/aha_in_isaac/task_data")
BACKUP = Path("/tmp/physics_apply_backup")
EXCLUDE = {"basketball_in_hoop", "beat_the_buzz", "change_channel",
           "close_box", "wipe_desk", "weighing_scales"}


def is_visual_name(n: str) -> bool:
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


def load(p):
    return json.loads(p.read_text()) if p.is_file() else None


def backup_and_write(path, data):
    rel = path.relative_to(TD)
    dst = BACKUP / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and not dst.is_file():
        shutil.copy2(path, dst)
    if not DRY:
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


tasks = sorted(p.stem for p in (TD / "objects").glob("*.json"))
n_tasks = n_types = n_mounts = 0
for task in tasks:
    if task in EXCLUDE:
        continue
    phys = load(TD / "physics" / f"{task}.json")
    objp = load(TD / "object_physics" / f"{task}.json")
    objs = load(TD / "objects" / f"{task}.json") or []
    if not (isinstance(phys, dict) and phys.get("shapes")) or not isinstance(objp, dict):
        continue
    shapes = phys["shapes"]
    parent_of = {o["name"]: o.get("parent") for o in objs}

    changed_types, added_mounts = [], []
    for name, shape in shapes.items():
        want = derive_type(name, shape)
        if want is None:
            continue
        if name in objp and (objp[name] or {}).get("type") not in (None, want):
            changed_types.append((name, objp[name]["type"], want))
            objp[name]["type"] = want
        # riding child: non-colliding static whose parent is a dynamic shape
        p = parent_of.get(name)
        if (not is_visual_name(name) and not shape.get("respondable") and not shape.get("collidable")
                and not shape.get("dynamic") and p and shapes.get(p, {}).get("dynamic")
                and not shape.get("mount_on_parent")):
            shape["mount_on_parent"] = True
            added_mounts.append((name, p))

    if changed_types or added_mounts:
        n_tasks += 1
        n_types += len(changed_types)
        n_mounts += len(added_mounts)
        if changed_types:
            backup_and_write(TD / "object_physics" / f"{task}.json", objp)
        if added_mounts:
            backup_and_write(TD / "physics" / f"{task}.json", phys)
        print(f"{task}: {len(changed_types)} type(s), {len(added_mounts)} mount(s)")

print(f"\n{'DRY-RUN: would change' if DRY else 'APPLIED to'} {n_tasks} tasks; "
      f"{n_types} type fixes, {n_mounts} mount_on_parent additions.")
print(f"Backups under {BACKUP}" if not DRY else "(no files written)")
