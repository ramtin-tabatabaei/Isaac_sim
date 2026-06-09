"""Dry-run: for every non-excluded task, compare each object's CURRENT object_physics
type against the type DERIVED from the authoritative CoppeliaSim physics JSON, and flag
non-colliding children of a dynamic body that should ride it (mount_on_parent).

Prints, per task, the type changes it would make and the mount_on_parent additions.
No Isaac/pxr needed (pure JSON)."""
import json
from pathlib import Path

TD = Path("/home/ramtin/IsaacLab/scripts/aha_in_isaac/task_data")
EXCLUDE = {"basketball_in_hoop", "beat_the_buzz", "change_channel",
           "close_box", "wipe_desk", "weighing_scales"}


def is_visual_name(n: str) -> bool:
    # Match scene_builder._is_render exactly: any "_vis" substring (covers _visual,
    # _vis, and trailing-digit forms like pepper_visual2 / book0_visual_book0_side).
    return "_vis" in n.lower()


def derive_type(name, shape):
    """Authoritative type from CoppeliaSim shape flags, or None if no authority."""
    if is_visual_name(name):
        return "visual"
    dyn = shape.get("dynamic")
    if dyn is None:
        return None
    if dyn:
        return "rigid"
    if shape.get("respondable") or shape.get("collidable"):
        return "kinematic"
    return "visual"  # static, non-colliding decoration (a fixed hand, a label)


def load(p):
    return json.loads(p.read_text()) if p.is_file() else None


tasks = sorted(p.stem for p in (TD / "objects").glob("*.json"))
summary = {"no_physics": [], "changed": [], "clean": [], "excluded": []}
total_type_changes = 0
total_mounts = 0

for task in tasks:
    if task in EXCLUDE:
        summary["excluded"].append(task)
        continue
    phys = load(TD / "physics" / f"{task}.json")
    objp = load(TD / "object_physics" / f"{task}.json")
    objs = load(TD / "objects" / f"{task}.json") or []
    if not phys or not isinstance(phys, dict) or not phys.get("shapes"):
        summary["no_physics"].append(task)
        continue
    shapes = phys["shapes"]
    pjoints = phys.get("joints", {}) or {}
    parent_of = {o["name"]: o.get("parent") for o in objs}
    objp = objp if isinstance(objp, dict) else {}

    def has_joint(name):
        """True if this object would get a USD joint: its parent is a '*_joint[N]'
        node that has a physics-joint entry (mirrors run_scene._add_articulation_joints)."""
        p = parent_of.get(name) or ""
        tail = p.rsplit("_joint", 1)
        return len(tail) == 2 and (not tail[1] or tail[1].isdigit()) and bool(pjoints.get(p))

    type_changes = []
    mounts = []
    risky = []
    for name, shape in shapes.items():
        want = derive_type(name, shape)
        if want is None:
            continue
        cur = (objp.get(name) or {}).get("type")
        if cur is None:
            # object not in object_physics (rare) - baker would derive it anyway
            continue
        if cur != want:
            jtag = ""
            if want == "rigid":
                p = parent_of.get(name) or ""
                if "_joint" in p:
                    jtag = " [JOINTED]" if has_joint(name) else " [!!! parent is a joint but NO joint entry -> would fall]"
                    if not has_joint(name):
                        risky.append((name, p))
                else:
                    jtag = " [free graspable]"
            type_changes.append((name, cur, want, jtag))
        # riding child: non-colliding static, parent is a DYNAMIC shape
        p = parent_of.get(name)
        if (not is_visual_name(name) and not shape.get("respondable") and not shape.get("collidable")
                and not shape.get("dynamic") and p and shapes.get(p, {}).get("dynamic")):
            already = (objp.get(name) or {}).get("mount_on_parent") or shape.get("mount_on_parent")
            if not already:
                mounts.append((name, p))

    if risky:
        summary.setdefault("risky", []).append((task, risky))
    if type_changes or mounts:
        summary["changed"].append(task)
        total_type_changes += len(type_changes)
        total_mounts += len(mounts)
        print(f"\n### {task}")
        for n, c, w, jtag in type_changes:
            print(f"    type  {n:28s} {c:10s} -> {w}{jtag}")
        for n, p in mounts:
            print(f"    mount {n:28s} ride -> {p}")
    else:
        summary["clean"].append(task)

print("\n" + "=" * 70)
print(f"excluded (untouched): {len(summary['excluded'])} -> {sorted(summary['excluded'])}")
print(f"no authoritative physics JSON: {len(summary['no_physics'])} -> {sorted(summary['no_physics'])}")
print(f"already clean (no change): {len(summary['clean'])}")
print(f"WOULD CHANGE: {len(summary['changed'])} tasks, "
      f"{total_type_changes} type fixes, {total_mounts} mount_on_parent additions")
risky_tasks = summary.get("risky", [])
print(f"\nRISKY (made rigid, parent is a joint but no joint entry -> may fall): {len(risky_tasks)} task(s)")
for task, rk in risky_tasks:
    print(f"    {task}: {[n for n, _ in rk]}")
