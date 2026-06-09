#!/usr/bin/env bash
# Failure-injecting twin of run_task.sh.
#
# Identical task discovery / menu as run_task.sh, but it drives run_scene_failgen.py
# (the non-invasive failgen wrapper) so you can inject a waypoint-tied failure.
# Everything after the task name is passed straight through to run_scene_failgen.py,
# including the --failure* args. With no --failure it runs a clean rollout.
#
# Usage:
#   scripts/aha_in_isaac/run_task_failgen.sh                       # menu, then clean run
#   scripts/aha_in_isaac/run_task_failgen.sh basketball_in_hoop --failure slip --failure-label lift --failure-after 5
#   scripts/aha_in_isaac/run_task_failgen.sh close_box --failure rotation_z --failure-range -0.6 0.6
#   scripts/aha_in_isaac/run_task_failgen.sh basketball_in_hoop --failure translation_x --failure-waypoint 1 --failure-range -0.1 0.1
#   scripts/aha_in_isaac/run_task_failgen.sh basketball_in_hoop --failure grasp --failure-label grasp
#   scripts/aha_in_isaac/run_task_failgen.sh basketball_in_hoop --failure freezing --failure-freeze-steps 90
#
# Run it from whatever conda env you normally use for isaaclab.sh.
set -euo pipefail

REPO=/home/ramtin/IsaacLab
REPORTS=/home/ramtin/AHA/portable_scene_reports
RUN="scripts/aha_in_isaac/run_scene_failgen.py"
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

# --- first non-flag argument: exact task name runs directly; otherwise it filters the menu ---
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
  echo "Runnable tasks (${#tasks[@]})${filter:+ matching '$filter'}:"
  PS3=$'\nPick a task number (Ctrl-C to quit): '
  select t in "${tasks[@]}"; do
    if [ -n "${t:-}" ]; then task="$t"; break; fi
    echo "Invalid choice, try again."
  done
fi

# --- resolve the scene context (.md preferred, .json fallback) ---
ctx="$REPORTS/$task.scene_context.md"
[ -f "$ctx" ] || ctx="$REPORTS/$task.scene_context.json"

echo
echo ">> Running '$task' with failgen  (extra args: ${*:-none})"
set -x
exec ./isaaclab.sh -p "$RUN" \
  --scene-context "$ctx" \
  --usd-dir "task_usds/${task}_physics" \
  --hide-root "$@"
