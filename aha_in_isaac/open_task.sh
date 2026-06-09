#!/usr/bin/env bash
# Interactive INSPECT launcher for the AHA-in-Isaac tasks.
#
# Same numbered task menu as run_task.sh, but instead of driving the Franka arm through
# the recorded motion, it just PLACES the scene and leaves the Isaac Sim GUI open and live
# so you can drive Isaac yourself: select objects, read their names / world poses in the
# Stage tree + Property panel, toggle visibility, watch physics settle, eyeball colliders, etc.
#
# Use it to figure out WHAT to change, then edit the per-task config files and re-run:
#   task_data/physics/<task>.json      -> planner, rrt_safety_margin, body types, joints
#   task_data/motion/<task>.json       -> waypoint dwell steps, gripper widths, carry-lift, ...
#   task_data/appearance/<task>.json   -> per-object visibility / texture / colour
# Then check the result with run_task.sh (which runs the arm).
#
# Usage:
#   scripts/aha_in_isaac/open_task.sh                  # show the full menu, pick, open
#   scripts/aha_in_isaac/open_task.sh block_pyramid    # exact task name -> skip the menu, open it
#   scripts/aha_in_isaac/open_task.sh close            # not an exact name -> filtered menu (close_*)
#   scripts/aha_in_isaac/open_task.sh block_pyramid --show-colliders
#       (anything after the task name/filter passes straight through to run_scene.py)
#   scripts/aha_in_isaac/open_task.sh block_pyramid --with-robot
#       (--with-robot: ALSO run the arm motion, like run_task.sh; otherwise the scene is idle)
#
# Run it from whatever conda env you normally use for isaaclab.sh (env_isaacsim51).
set -euo pipefail

REPO=/home/ramtin/IsaacLab
REPORTS=/home/ramtin/AHA/portable_scene_reports
RUN="scripts/aha_in_isaac/run_scene.py"
cd "$REPO"

# --- discover runnable tasks: a non-empty <task>_physics USD dir + a scene-context report ---
mapfile -t tasks < <(
  for d in task_usds/*_physics; do
    [ -d "$d" ] && [ -n "$(ls -A "$d" 2>/dev/null)" ] || continue
    t=$(basename "$d"); t=${t%_physics}
    [ -f "$REPORTS/$t.scene_context.md" ] || [ -f "$REPORTS/$t.scene_context.json" ] || continue
    echo "$t"
  done | sort -u
)
[ ${#tasks[@]} -gt 0 ] || { echo "No runnable tasks found (need task_usds/<task>_physics + a scene-context report)."; exit 1; }

# --- pull our own meta-flag (--with-robot) out of the args before anything else ---
with_robot=0
rest=()
for a in "$@"; do
  if [ "$a" = "--with-robot" ]; then with_robot=1; else rest+=("$a"); fi
done
set -- ${rest[@]+"${rest[@]}"}

# --- first non-flag argument: exact task name opens directly; otherwise it filters the menu ---
task=""; filter=""
if [ "${1:-}" ] && [[ "$1" != -* ]]; then
  for t in "${tasks[@]}"; do
    if [ "$t" = "$1" ]; then task="$1"; break; fi
  done
  filter="$1"; shift   # consume it (as the chosen task or as a substring filter); rest pass through
fi

# --- otherwise show the menu (optionally narrowed by the filter) ---
if [ -z "$task" ]; then
  if [ -n "$filter" ]; then
    mapfile -t tasks < <(printf '%s\n' "${tasks[@]}" | grep -i -- "$filter" || true)
    [ ${#tasks[@]} -gt 0 ] || { echo "No tasks match '$filter'."; exit 1; }
  fi
  echo "Tasks to open (${#tasks[@]})${filter:+ matching '$filter'}:"
  PS3=$'\nPick a task number (Ctrl-C to quit): '
  select t in "${tasks[@]}"; do
    if [ -n "${t:-}" ]; then task="$t"; break; fi
    echo "Invalid choice, try again."
  done
fi

# --- resolve the scene context (.md preferred, .json fallback) ---
ctx="$REPORTS/$task.scene_context.md"
[ -f "$ctx" ] || ctx="$REPORTS/$task.scene_context.json"

# --- by default open the scene IDLE (no arm); --with-robot runs the motion like run_task.sh ---
robot_flag="--no-robot"
[ "$with_robot" = "1" ] && robot_flag=""

echo
if [ "$with_robot" = "1" ]; then
  echo ">> Opening '$task' WITH the arm motion  (extra args: ${*:-none})"
else
  echo ">> Opening '$task' for inspection - scene placed, arm idle  (extra args: ${*:-none})"
  echo "   The Isaac Sim GUI stays open and live. Select objects in the Stage tree to read"
  echo "   their names / world transforms, then edit task_data/{physics,motion,appearance}/$task.json"
  echo "   and re-check with run_task.sh. Add --show-colliders to see collision shapes. Ctrl-C to quit."
fi
set -x
exec ./isaaclab.sh -p "$RUN" \
  --scene-context "$ctx" \
  --usd-dir "task_usds/${task}_physics" \
  $robot_flag "$@"
