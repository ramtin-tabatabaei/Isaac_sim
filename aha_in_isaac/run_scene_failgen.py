"""
run_scene_failgen.py

Run an *aha_in_isaac* task with an injected failure (failgen).

This is a thin, NON-INVASIVE wrapper around ``run_scene.py``: it peels off the
``--failure*`` arguments, hands every other argument to ``run_scene.py`` so the
simulator launches and the scene is built EXACTLY as usual, then patches the
waypoint controller to inject the failure before the motion runs. With
``--failure none`` (the default) it behaves byte-for-byte like ``run_scene.py``,
so nothing about the normal pipeline or any in-flight run changes.

Usage (all the normal run_scene.py args, plus the failure ones):

    cd ~/IsaacLab
    ./isaaclab.sh -p scripts/aha_in_isaac/run_scene_failgen.py \
        --scene-context /home/ramtin/AHA/portable_scene_reports/basketball_in_hoop.scene_context.md \
        --usd-dir task_usds/basketball_in_hoop_physics --hide-root \
        --failure slip --failure-label "lift" --failure-after 5

Failure args (see failgen/README.md):
    --failure {none,slip,grasp,translation[_x/_y/_z],rotation[_x/_y/_z],no_rotation,freezing}
    --failure-waypoint INT      target waypoint index (-1 = random / by label)
    --failure-label SUBSTR      target the first waypoint whose label contains SUBSTR
    --failure-after INT         slip: steps after the waypoint before the gripper opens
    --failure-range LOW HIGH    translation (m) / rotation (rad) perturbation range
    --failure-axis {x,y,z}      axis for the generic translation/rotation failure
    --failure-freeze-steps INT  freezing: how long the arm stays frozen
    --failure-seed INT          seed the failure RNG
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# This file lives next to run_scene.py / cli.py / robot_controller.py; make that
# directory importable (it also makes the ``failgen`` subpackage importable).
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))


def main():
    from failgen.injector import add_failure_args, install

    # 1. Split off the failure args; leave everything else for run_scene.py.
    fg_parser = add_failure_args(argparse.ArgumentParser(add_help=False))
    fg_args, remaining = fg_parser.parse_known_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining

    # 2. Importing run_scene parses the remaining args, launches the simulator,
    #    and imports the Isaac-dependent modules -- a normal run_scene.py launch.
    import run_scene

    # 3. Patch the controller to inject the failure before the motion executes.
    install(fg_args)

    # 4. Run the real entry point, then close the app (run_scene's own
    #    __main__ guard does not fire because we imported it as a module).
    try:
        run_scene.main()
    finally:
        run_scene.simulation_app.close()


if __name__ == "__main__":
    main()
