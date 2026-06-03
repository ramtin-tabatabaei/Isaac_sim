# aha_in_isaac

Place an exported AHA task scene in Isaac Sim and drive a Franka arm through the
task's waypoints.

## Run

```bash
cd ~/IsaacLab
./isaaclab.sh -p scripts/aha_in_isaac/run_scene.py \
    --scene-context /home/ramtin/AHA/portable_scene_reports/wipe_desk.scene_context.md \
    --usd-dir .../task_usds/wipe_desk --hide-root
```

For `beat_the_buzz`, this command auto-selects the deterministic lift-over carry
path (`--carry-lift 0.35`) so the wand is carried clear of the cuboid. The wand is
also pinned in its authored pose until the gripper closes, so gravity stays on
without letting it fall before grasp:

```bash
./isaaclab.sh -p scripts/aha_in_isaac/run_scene.py \
    --scene-context /home/ramtin/AHA/portable_scene_reports/beat_the_buzz.scene_context.md \
    --usd-dir task_usds/beat_the_buzz_physics --hide-root
```

Pass `--carry-lift 0` to force the recorded, non-lifted path for diagnostics.

## Files (one responsibility each)

| File | Responsibility |
|------|----------------|
| `run_scene.py` | Entry point. Parses args + report, launches the sim, builds the scene, runs the arm. |
| `cli.py` | Command-line arguments (pure Python). |
| `scene_context.py` | Parse the `.scene_context.md` report; derive object poses, table-top height, task-root transform, robot base (pure Python). |
| `motion_config.py` | Load per-task step counts from `task_motion_config.json`. |
| `scene_builder.py` | Spawn floor, table, objects, waypoint markers, robot. |
| `arm_motion.py` | Turn the report's waypoints into the controller's `Waypoint` list. |
| `robot_arm.py` | Spawn the Franka articulation. |
| `robot_controller.py` | Waypoint controller (drives the arm, gripper). Default differential-IK; optional RMPFlow planner via `--planner rmpflow`. |
| `rmpflow_planner.py` | **Experimental** RMPFlow (Lula) reactive collision-avoiding driver (`--planner rmpflow`). |
| `curobo_planner.py` | **Experimental** cuRobo global collision-free planner (`--planner curobo`); plans a joint trajectory per waypoint around the scene's mesh/bbox obstacles. By default (`--curobo-graph on`) it routes each plan with cuRobo's sampling-based **PRM** graph planner (collision-free through free space, then trajopt smooths it) and keeps a `--curobo-safety-margin` (default 0.03 m) clearance, so the path no longer hugs/clips the cuboid or carried wand. `--curobo-graph off` restores trajopt-only. Requires `pip install` of cuRobo. |
| `task_motion_config.json` | Per-task `waypoint_steps` + `gripper_closed` (arm/gripper motion). |
| `add_physics_to_usds.py` | **Offline tool** — bake physics onto object USDs. One task (`--task`/`--input-dir`/`--output-dir`) or all at once (`--batch-root task_usds`). Then point `--usd-dir` at the output. |
| `object_physics_config.json` | Per-task, per-object physics (`type` = rigid/kinematic/visual/deformable, `density`, friction, ...) consumed by the baker via `--task`. |
| `object_appearance_config.json` | Per-task, per-object appearance (`visible`, `texture`, `color`, roughness, metallic) applied at scene-build time. |
| `object_appearance_config.json` *(cont.)* | Has a `_scene` block for the shared table/floor too. |
| `generate_task_configs.py` | **Generator** — fills both configs for every task with a scene report and synthesises a per-object texture, inferring type/color from names. Pure Python: `python3 generate_task_configs.py`. Skips already-configured tasks. |
| `usd_uv.py` | Shared UV-generation helper (auto/spherical/box/planar), used by the baker and the runtime. |
| `textures/` | Object textures. Shared ones (`basketball.png`, `wood.png`, ...) plus a per-task subfolder `textures/<task>/<object>.png` for the generated ones. |

## How it works

`scene_context.md` (where things are) **+** `task_motion_config.json` (how fast)
→ `run_scene` parses both, launches Isaac, then `scene_builder` spawns the scene
and `robot_arm` spawns the Franka. `arm_motion` builds the waypoint list and
`robot_controller` follows it with differential IK, grasping the physics-baked
object (from `add_physics_to_usds.py`) at the lowest waypoint and releasing at the
last.

Loading order matters: argument parsing and report parsing are pure Python and
run **before** the simulator launches; the Isaac-dependent modules
(`scene_builder`, `arm_motion`, `robot_*`) are imported **after** `AppLauncher`
starts, because they import Isaac Lab at module load.
