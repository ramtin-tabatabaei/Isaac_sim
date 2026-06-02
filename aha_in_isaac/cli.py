"""
cli.py

Command-line argument definitions for the AHA-in-Isaac scene runner.

Kept free of Isaac Lab imports so the parser can be built before the simulator
is launched. ``run_scene.py`` adds the ``AppLauncher`` arguments on top of this.
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_SCENE_CONTEXT = Path("/home/ramtin/AHA/portable_scene_reports/basketball_in_hoop.scene_context.md")
DEFAULT_USD_DIR = Path(
    "/home/ramtin/Downloads/basketball_in_hoop_usd-20260529T125644Z-3-001/basketball_in_hoop_usd"
)
DEFAULT_MOTION_CONFIG = Path(__file__).with_name("task_motion_config.json")
DEFAULT_APPEARANCE_CONFIG = Path(__file__).with_name("object_appearance_config.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Place an exported AHA task scene in Isaac Sim and run the arm.")
    parser.add_argument("--scene-context", type=Path, default=DEFAULT_SCENE_CONTEXT)
    parser.add_argument("--usd-dir", type=Path, default=DEFAULT_USD_DIR)
    parser.add_argument("--table-usd", type=Path, default=None)
    parser.add_argument(
        "--table-top-object",
        default="auto",
        help="Scene object whose z position defines the dining-table top. Use 'auto' to infer it.",
    )
    parser.add_argument("--no-table", action="store_true", help="Do not spawn diningTable.usdc.")
    parser.add_argument("--no-robot", action="store_true", help="Do not spawn the Franka arm or run the motion.")
    parser.add_argument(
        "--motion-config",
        type=Path,
        default=DEFAULT_MOTION_CONFIG,
        help="JSON file with per-task arm-motion config (sim steps to spend reaching each named waypoint).",
    )
    parser.add_argument(
        "--appearance-config",
        type=Path,
        default=DEFAULT_APPEARANCE_CONFIG,
        help="JSON file with per-task, per-object appearance (visibility, texture, color).",
    )
    parser.add_argument(
        "--planner",
        choices=("diffik", "rmpflow", "curobo"),
        default="diffik",
        help="Arm controller: 'diffik' (default straight-line differential IK), 'rmpflow' "
        "(reactive Lula avoidance), or 'curobo' (global collision-free planner). The last two "
        "avoid the scene's static objects and are EXPERIMENTAL.",
    )
    parser.add_argument(
        "--curobo-obstacles",
        choices=("mesh", "bbox"),
        default="mesh",
        help="For --planner curobo: obstacles as their actual mesh (accurate - lets the arm "
        "route around a thin rod) or as bounding boxes (a box can swallow a nearby grasp goal).",
    )
    parser.add_argument(
        "--curobo-graph",
        choices=("on", "off"),
        default="on",
        help="For --planner curobo: seed each plan with cuRobo's sampling-based PRM (probabilistic "
        "roadmap) graph planner so the trajectory is routed around obstacles through free space, "
        "instead of pure local trajectory optimization (which hugs/cuts corners near obstacles and "
        "can clip the cuboid/wand). 'on' (default, SAFER) plans the roadmap first; 'off' restores the "
        "trajopt-only behaviour.",
    )
    parser.add_argument(
        "--curobo-safety-margin",
        type=float,
        default=0.03,
        metavar="METRES",
        help="For --planner curobo: extra clearance the planner keeps from every obstacle "
        "(cuRobo's collision activation distance, metres). Larger = safer but can make goals very "
        "close to an obstacle unreachable (then the diff-IK fallback runs instead). Default 0.03; "
        "the old cuRobo default was 0.01.",
    )
    parser.add_argument(
        "--ee-down",
        action="store_true",
        help="Force the gripper straight down at every waypoint instead of following the "
        "report's recorded waypoint orientation (the AHA/RLBench behaviour). Useful only for "
        "simple top-down grasps; leave OFF for tasks with tilted/threaded approaches.",
    )
    parser.add_argument(
        "--straight-path",
        action="store_true",
        help="Move in straight segments between waypoints instead of the default smooth "
        "(Catmull-Rom) curved path through them.",
    )
    parser.add_argument(
        "--carry-lift",
        type=float,
        default=0.0,
        metavar="DZ",
        help="Metres to raise the grasped object during transit so it clears a tall "
        "obstacle (e.g. lift the wand's ring up and OVER the buzz-wire arch instead of "
        "dragging it through). When >0, the arm: hovers above the grasp, descends, grasps, "
        "lifts straight up by DZ, traverses the remaining waypoints at that clearance "
        "height, then releases. A deterministic, planner-free collision-free path. "
        "Recommended ~0.2 for beat_the_buzz; leave 0 to follow waypoints as recorded.",
    )
    parser.add_argument(
        "--collision-watch",
        type=Path,
        default=None,
        metavar="CSV",
        help="Diagnostics: while the arm moves, measure the closest distance between the grasped "
        "object (ring) and each obstacle (the wire/base), and between the wrist/hand/fingers and "
        "each obstacle. Writes a per-step trace CSV here and prints the global minima (a value near "
        "0 mm means the geometry is touching/penetrating). Headless-friendly way to 'see' a collision.",
    )
    parser.add_argument("--no-waypoints", action="store_true", help="Do not spawn waypoint marker spheres.")
    parser.add_argument("--hide-root", action="store_true", help="Hide inferred task-root objects.")
    parser.add_argument(
        "--hide-object", action="append", default=[], help="Hide a named scene object. Can be repeated."
    )
    parser.add_argument(
        "--show-colliders",
        action="store_true",
        help="Render physics collider wireframes (PhysX visualization) so you can SEE each "
        "object's actual collision shape - e.g. whether the wand's ring collider keeps its hole.",
    )
    parser.add_argument(
        "--filter-collision",
        nargs=2,
        action="append",
        default=[],
        metavar=("A", "B"),
        help="Disable collision between two named scene objects (repeatable). e.g. "
        "--filter-collision wand Cuboid stops the wand's handle (which overlaps the base) "
        "from ejecting the wand, so it stays threaded on the rod.",
    )
    parser.add_argument(
        "--wand-offset",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("DX", "DY", "DZ"),
        help="Shift the wand body (and its glued ring) spawn position by (dx,dy,dz) metres, "
        "e.g. to center the ring's hole on the rod for collider testing.",
    )
    parser.add_argument(
        "--object-pose-mode",
        choices=("task-root", "baked", "scene-context"),
        default="task-root",
        help=(
            "Use 'task-root' for exported AHA task USDs, 'baked' for files already at final world poses, "
            "or 'scene-context' for USDs exported around local object origins."
        ),
    )
    return parser
