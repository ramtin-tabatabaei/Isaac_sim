"""build_task_physics.py

Read the REAL physics CoppeliaSim stores in each RLBench ``.ttm`` and write it to
``aha_in_isaac/task_data/physics/<task>.json`` -- the per-shape dynamics and the
per-joint type/limits Isaac would otherwise have to hand-author.

Per task::

    {
      "shapes": { "<name>": {dynamic, respondable, collidable, mass, friction} },
      "joints": { "<joint_name>": {type, cyclic, lower, upper, position} }   # deg / m
    }

Needs CoppeliaSim + PyRep (run in the same env as export_ttm_scene_context_Isaac.py):

    python aha_in_isaac/build_task_physics.py            # all tasks with a .ttm
    python aha_in_isaac/build_task_physics.py close_box  # only the named task(s)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "/home/ramtin/AHA/aha_scripts")
import export_ttm_scene_context_Isaac as ex  # constants + configure_coppeliasim_env

OUT_DIR = Path(__file__).resolve().parent / "task_data" / "physics"
RELAUNCH_EVERY = 15  # rebuild the PyRep session periodically to avoid drift/leaks


def _safe(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception:
        return None


def _read_model(pr, ttm: Path) -> dict:
    from pyrep.const import JointType, ObjectType
    from pyrep.objects.joint import Joint
    from pyrep.objects.shape import Shape

    root = pr.import_model(str(ttm))
    try:
        objs = root.get_objects_in_tree(exclude_base=False)
        shapes: dict = {}
        joints: dict = {}
        for o in objs:
            kind = _safe(o.get_type)
            name = _safe(o.get_name)
            if name is None:
                continue
            if kind == ObjectType.SHAPE:
                s = Shape(o.get_handle())
                shapes[name] = {
                    "dynamic": _safe(s.is_dynamic),
                    "respondable": _safe(s.is_respondable),
                    "collidable": _safe(s.is_collidable),
                    "mass": _safe(s.get_mass),
                    "friction": _safe(s.get_bullet_friction),
                }
            elif kind == ObjectType.JOINT:
                j = Joint(o.get_handle())
                jt = _safe(j.get_joint_type)
                iv = _safe(j.get_joint_interval)
                entry = {
                    "type": jt.name.lower() if jt is not None else None,
                    "cyclic": None,
                    "lower": None,
                    "upper": None,
                    "position": None,
                }
                pos = _safe(j.get_joint_position)
                if pos is not None:
                    entry["position"] = (
                        math.degrees(float(pos)) if jt == JointType.REVOLUTE else float(pos)
                    )
                if isinstance(iv, tuple) and iv[1] is not None:
                    entry["cyclic"] = bool(iv[0])
                    lo = float(iv[1][0])
                    hi = lo + float(iv[1][1])
                    if jt == JointType.REVOLUTE:  # store deg for revolute, m for prismatic
                        entry["lower"], entry["upper"] = math.degrees(lo), math.degrees(hi)
                    else:
                        entry["lower"], entry["upper"] = lo, hi
                joints[name] = entry
        return {"shapes": shapes, "joints": joints}
    finally:
        _safe(root.remove)


def main(argv: list[str]) -> int:
    ex.configure_coppeliasim_env()
    from pyrep import PyRep

    if argv:
        tasks = list(argv)
    else:
        tasks = sorted(p.stem for p in ex.TTM_DIR.glob("*.ttm"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pr = None
    done = failed = 0
    try:
        for i, task in enumerate(tasks):
            ttm = ex.TTM_DIR / f"{task}.ttm"
            if not ttm.is_file():
                print(f"[skip] {task}: no .ttm", flush=True)
                continue
            if pr is None or (i % RELAUNCH_EVERY == 0):
                if pr is not None:
                    _safe(pr.stop)
                    _safe(pr.shutdown)
                pr = PyRep()
                pr.launch(str(ex.BASE_SCENE), headless=True)
                pr.start()
            try:
                data = _read_model(pr, ttm)
                (OUT_DIR / f"{task}.json").write_text(
                    json.dumps(data, indent=2) + "\n", encoding="utf-8"
                )
                done += 1
                print(
                    f"[ok] {task}: {len(data['shapes'])} shapes, {len(data['joints'])} joints",
                    flush=True,
                )
            except Exception as exc:
                failed += 1
                print(f"[FAIL] {task}: {type(exc).__name__}: {exc}", flush=True)
    finally:
        if pr is not None:
            _safe(pr.stop)
            _safe(pr.shutdown)
    print(f"\nDone: {done} written, {failed} failed -> {OUT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
