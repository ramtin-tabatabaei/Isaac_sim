# aha_in_isaac failgen — waypoint-tied failure injection

Port of the RLBench `failgen` failures
(`AHA/aha/Data_Generation/rlbench-failgen/failgen/`) to the `aha_in_isaac`
task runner. It injects a failure into a real `run_scene.py` rollout by
patching the `FrankaWaypointController` at runtime — **no existing file is
edited**, so the normal pipeline and any in-flight runs are untouched.

## Layout

```
aha_in_isaac/
├── run_scene.py            # unchanged stock runner
├── run_task.sh             # unchanged stock launcher
├── run_scene_failgen.py    # NEW: wrapper that strips --failure* then runs run_scene
├── run_task_failgen.sh     # NEW: failgen twin of run_task.sh
└── failgen/                # NEW: this package
    ├── waypoint_failure.py # the failure classes (slip/grasp/translation/rotation/…)
    ├── fail_manager.py     # FailureManager (fans hooks out to each failure)
    └── injector.py         # monkeypatches FrankaWaypointController.follow
```

## How it works

`run_scene_failgen.py` parses and removes the `--failure*` args, lets
`run_scene.py` launch the sim and build the scene exactly as usual, then
`injector.install()` patches `FrankaWaypointController.follow`. On the first
`follow` of a controller it:

1. **mutates the waypoint list** (translation / rotation / no_rotation / grasp), then
2. **wraps that controller's `_set_gripper`** (so `slip` can force the gripper open)
   **and `_step`** (so `freezing` can hold the arm, and every failure advances its
   per-step state machine each physics step).

With `--failure none` (default) nothing is patched → identical to `run_scene.py`.

## Usage

```bash
cd ~/IsaacLab     # in your isaaclab conda env (env_isaacsim51)

# via the launcher (task discovery + menu, like run_task.sh):
scripts/aha_in_isaac/run_task_failgen.sh basketball_in_hoop \
    --failure slip --failure-label lift --failure-after 5

# or directly:
./isaaclab.sh -p scripts/aha_in_isaac/run_scene_failgen.py \
    --scene-context /home/ramtin/AHA/portable_scene_reports/basketball_in_hoop.scene_context.md \
    --usd-dir task_usds/basketball_in_hoop_physics --hide-root \
    --failure rotation_z --failure-range -0.6 0.6 --failure-seed 0
```

## Failure types

| `--failure`        | kind            | effect |
|--------------------|-----------------|--------|
| `none`             | —               | clean rollout (no patch) |
| `slip`             | step-level      | re-open the gripper `--failure-after` steps after the target waypoint; object drops and cannot be re-grasped |
| `grasp`            | waypoint-target | neutralise the open/close at the target waypoint → object is never secured |
| `translation[_x/_y/_z]` | waypoint-target | shift the target waypoint position by a random offset in `--failure-range` metres |
| `rotation[_x/_y/_z]`    | waypoint-target | rotate the target waypoint orientation by a random angle in `--failure-range` radians |
| `no_rotation`      | waypoint-target | reuse the previous waypoint's orientation (skip the reorient) |
| `freezing`         | step-level      | hold the arm still for `--failure-freeze-steps` steps after the target waypoint |

For the bare `translation` / `rotation` types use `--failure-axis {x,y,z}`
(the `_x/_y/_z` variants fix the axis).

## Targeting a waypoint

In priority order:
1. `--failure-label SUBSTR` — first waypoint whose label contains `SUBSTR` (case-insensitive),
2. `--failure-waypoint INT` — explicit index,
3. otherwise a random waypoint (seed with `--failure-seed`).

Print the waypoint labels for a task by doing a normal `run_scene.py` run and
reading the `[INFO]: Arm waypoint '<label>' -> …` lines.

## Notes / limitations

- **Waypoint-target** failures work with every planner (diffik / rrt / curobo /
  rmpflow). **Step-level** failures (`slip`, `freezing`) arm off
  `controller._active_label`, which is set inside `_diffik_segment` — so they
  apply to the default `diffik` path and the rrt/curobo per-segment fallback. A
  pure RMPFlow reactive path does not set that label, so step-level failures are
  a no-op there (the waypoint-target ones still work).
- `slip` opening the gripper relies on the object being held by friction; it
  drops whatever the fingers hold (matching the RLBench `slip` semantics).
- Add a new failure: implement a `WaypointFailure` subclass in
  `waypoint_failure.py` and add it to `FAILURE_TYPES`.
