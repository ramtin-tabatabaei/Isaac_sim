#!/usr/bin/env bash
# Interactive launcher for the AHA-in-Isaac tasks.
#
# Lists every runnable task (one that has a baked task_usds/<task>_physics folder AND a
# scene-context report), lets you pick one from a numbered menu, then runs it with run_scene.py.
#
# Usage:
#   scripts/aha_in_isaac/run_task.sh                 # show the full menu, pick, run
#   scripts/aha_in_isaac/run_task.sh close_box       # exact task name -> skip the menu, run it
#   scripts/aha_in_isaac/run_task.sh close           # not an exact name -> filtered menu (close_*)
#   scripts/aha_in_isaac/run_task.sh close_box --device cpu --show-colliders
#       (anything after the task name/filter is passed straight through to run_scene.py)
#
# Run it from whatever conda env you normally use for isaaclab.sh.
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
echo ">> Running '$task'  (extra args: ${*:-none})"
set -x
exec ./isaaclab.sh -p "$RUN" \
  --scene-context "$ctx" \
  --usd-dir "task_usds/${task}_physics" \
  --hide-root "$@"
