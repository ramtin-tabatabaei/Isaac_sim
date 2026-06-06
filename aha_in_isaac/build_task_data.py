"""build_task_data.py

Split each ``portable_scene_reports/<task>.scene_context.md`` into the structured,
per-category JSON files Isaac actually loads from, under ``aha_in_isaac/task_data/``:

    task_data/
      objects/<task>.json      - scene objects (pose, parent, hierarchy, joint frame)
      waypoints/<task>.json     - ordered end-effector waypoints (+ cartesian paths)
      robot_base/<task>.json    - Franka base pose (or null if the report has none)
      graspables/<task>.json    - placement_distribution (which objects RLBench randomises)
      meta/<task>.json          - task_name, root_object, ranges, units, schema, source ttm

The ``.md`` reports stay the source of truth (the export writes them); this just
re-materialises the useful parts into folders the runtime reads via
``scene_context.load_report``. Re-run after exporting new/updated tasks:

    python aha_in_isaac/build_task_data.py            # all tasks
    python aha_in_isaac/build_task_data.py close_box  # one or more named tasks
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPORTS_DIR = Path("/home/ramtin/AHA/portable_scene_reports")
TASK_DATA_DIR = Path(__file__).resolve().parent / "task_data"

# category name -> how to pull that slice out of a parsed report dict.
META_KEYS = (
    "task_name",
    "root_object",
    "placement_ranges",
    "units_and_conventions",
    "schema_name",
    "schema_version",
    "source_ttm_path",
    "runtime_generated_scene_objects_excluded",
    "end_effector_collection_error",
)
CATEGORIES = {
    "objects": lambda r: r.get("objects", []),
    "waypoints": lambda r: r.get("waypoints", []),
    "robot_base": lambda r: r.get("robot_base"),
    "graspables": lambda r: r.get("placement_distribution"),
    "meta": lambda r: {k: r.get(k) for k in META_KEYS},
}


def parse_report_md(path: Path) -> dict:
    """Pull the fenced ```json``` scene block out of a scene-context .md file."""
    text = path.read_text(encoding="utf-8")
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if match is None:
        raise RuntimeError(f"No fenced JSON scene data found in {path}")
    return json.loads(match.group(1))


def write_task(task: str, report: dict) -> None:
    for category, extract in CATEGORIES.items():
        out_dir = TASK_DATA_DIR / category
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{task}.json").write_text(
            json.dumps(extract(report), indent=2) + "\n", encoding="utf-8"
        )


def main(argv: list[str]) -> int:
    if argv:
        mds = [REPORTS_DIR / f"{name}.scene_context.md" for name in argv]
    else:
        mds = sorted(REPORTS_DIR.glob("*.scene_context.md"))

    written = 0
    for md in mds:
        if not md.is_file():
            print(f"[skip] {md.name}: not found")
            continue
        task = md.name[: -len(".scene_context.md")]
        write_task(task, parse_report_md(md))
        written += 1
    print(
        f"Wrote {written} task(s) x {len(CATEGORIES)} categories "
        f"({', '.join(CATEGORIES)}) under {TASK_DATA_DIR}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
