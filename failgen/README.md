# IsaacLab Failgen

This folder separates clean task definitions from failure injection.

## Layout

- `tasks/`: clean task modules. A task should expose `run_task(simulation_app, failure=None)`.
- `failures/`: failure injectors. A failure can observe subgoal events and override commands.
- `../run_failgen_task.py`: main entry point for selecting task and failure type.

## Run Examples

Successful task:

```bash
./isaaclab.sh -p scripts/run_failgen_task.py --task basketball_in_hoop --failure none
```

Slip/drop failure:

```bash
./isaaclab.sh -p scripts/run_failgen_task.py --task basketball_in_hoop --failure slip
```

Slip after a different subgoal:

```bash
./isaaclab.sh -p scripts/run_failgen_task.py --task basketball_in_hoop --failure slip --failure-after-subgoal "approach waypoint3 above hoop"
```

## Add A Task

1. Create `tasks/my_task.py`.
2. Implement `run_task(simulation_app, failure=None)`.
3. Add it to `TASK_TYPES` in `tasks/__init__.py`.

## Add A Failure

1. Create a file in `failures/`.
2. Implement these methods:
   - `on_task_start(context)`
   - `on_subgoal_start(label, context)`
   - `on_subgoal_complete(label, context)`
   - `update_gripper_target(nominal_grip, context)`
3. Register it in `FAILURE_TYPES` in `failures/__init__.py`.
