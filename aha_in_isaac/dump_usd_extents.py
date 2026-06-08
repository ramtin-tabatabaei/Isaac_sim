"""
dump_usd_extents.py

One-off (cacheable) pass over the baked task USDs to record each object's mesh
extent, so texture painters can size each image to the object's true face aspect
ratio. Writes ``textures/_usd_extents.json``:

    { "<task>": { "<object>": {"ext_cm": [ex, ey, ez] } }, ... }

``<object>`` is the USD stem with the leading "<task>_" stripped (matching the
keys in object_appearance_config.json), lower-cased for robust lookup.

Needs the bundled USD ``pxr`` (only present in the isaacsim env). Run via
``run_dump_usd_extents.sh`` which sets PYTHONPATH/LD_LIBRARY_PATH, or replicate
those env vars. Plain ``python`` without those will fail to import pxr.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

from pxr import Gf, Usd, UsdGeom

USD_ROOT = Path("/home/ramtin/Downloads/task_usds-20260530T142739Z-3-001/task_usds")
OUT = Path(__file__).with_name("textures") / "_usd_extents.json"


def mesh_extent_cm(path: str):
    """World-space bounding-box extent (cm) over all meshes in the stage, or None."""
    try:
        stage = Usd.Stage.Open(path)
    except Exception:
        return None
    if stage is None:
        return None
    mn = [1e30] * 3
    mx = [-1e30] * 3
    npts = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if not pts:
            continue
        xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        for p in pts:
            wp = xf.Transform(Gf.Vec3d(p[0], p[1], p[2]))
            for k in range(3):
                mn[k] = min(mn[k], wp[k])
                mx[k] = max(mx[k], wp[k])
            npts += 1
    if npts == 0:
        return None
    return [round((mx[k] - mn[k]) * 100.0, 3) for k in range(3)]


def main():
    if not USD_ROOT.is_dir():
        raise SystemExit(f"USD root not found: {USD_ROOT}")
    out: dict[str, dict] = {}
    task_dirs = sorted(d for d in USD_ROOT.iterdir() if d.is_dir())
    total = 0
    for td in task_dirs:
        task = td.name
        block: dict[str, list] = {}
        for f in sorted(glob.glob(os.path.join(str(td), "*.usd"))):
            stem = Path(f).stem
            obj = stem[len(task) + 1:] if stem.lower().startswith(task.lower() + "_") else stem
            ext = mesh_extent_cm(f)
            if ext is None:
                continue
            block[obj.lower()] = ext
            total += 1
        if block:
            out[task] = block
        print(f"  {task:42s} {len(block)} objects")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {OUT} : {len(out)} tasks, {total} objects.")


if __name__ == "__main__":
    main()
