# task_data — structured scene info Isaac loads

This folder is the **structured form of the AHA scene reports** that the Isaac
pipeline reads at runtime. Each `portable_scene_reports/<task>.scene_context.md`
is split, by [`build_task_data.py`](../build_task_data.py), into one JSON file per
task per category:

| folder | per-task file | what it holds | who uses it |
|---|---|---|---|
| `objects/`    | `<task>.json` | scene objects: `world_location` (pose), `parent`, `hierarchy_path`, `parent_local_location` (joint frame) | `scene_builder` (placement), `run_scene._add_articulation_joints` (hinges), `scene_context` (task root, table height) |
| `waypoints/`  | `<task>.json` | ordered end-effector waypoints + `cartesian_path_samples` (with per-sample `orientation_rpy_rad`) | `arm_motion` (the motion), `scene_builder` (waypoint markers) |
| `robot_base/` | `<task>.json` | Franka base pose, or `null` (a default base is then used) | `scene_context.robot_base_pose` |
| `graspables/` | `<task>.json` | `placement_distribution`: which objects RLBench randomises each reset | `scene_context.graspable_names` (snap weights to the report pose) |
| `physics/`    | `<task>.json` | **real CoppeliaSim physics**: per-shape `mass`/`friction`/`dynamic`/`respondable`, per-joint `type`+`lower`/`upper` limits | `run_scene._add_articulation_joints` (joint type+limits), `add_physics_to_usds` (mass/friction defaults) |
| `meta/`       | `<task>.json` | `task_name`, `root_object`, `placement_ranges`, `units_and_conventions`, schema, source `.ttm` | misc / provenance |

`objects/waypoints/robot_base/graspables/meta` are split from the `.md` by
`build_task_data.py`. `physics/` is different — it is read straight from the
CoppeliaSim `.ttm` (mass, friction, joint limits aren't in the `.md`) by
[`build_task_physics.py`](../build_task_physics.py), which needs PyRep/CoppeliaSim.

## How the runtime reads it

`scene_context.load_report(path)` derives the task name from the path and
**reassembles the report dict from these folders** (`_report_from_task_data`). If a
task has no folder entry it falls back to parsing the `.md`, so nothing breaks for
an un-built task. The rest of the pipeline is unchanged — it still sees one report
dict, now sourced from here.

## Regenerating

The `.md` reports stay the source of truth (the export writes them). After
exporting new or updated tasks, re-materialise the folders:

```bash
python aha_in_isaac/build_task_data.py            # all tasks
python aha_in_isaac/build_task_data.py close_box  # only the named task(s)
```
