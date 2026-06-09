"""
injector.py

Installs the failure injection into the live ``aha_in_isaac`` run *without
editing* ``run_scene.py`` or ``robot_controller.py`` -- so the normal pipeline
(and any in-flight runs) are completely untouched.

It monkeypatches ``FrankaWaypointController.follow`` once, at runtime. On the
first ``follow`` call of a controller it:

    1. mutates the waypoint list via the manager (translation / rotation /
       no_rotation / grasp), then
    2. wraps that controller instance's ``_set_gripper`` (so ``slip`` can force
       the gripper open) and ``_step`` (so ``freezing`` can hold the arm and so
       every failure advances its per-step state machine),

and then calls the original ``follow``. With no manager set (``--failure none``)
nothing is patched and the run is byte-for-byte the stock ``run_scene.py``.
"""

from __future__ import annotations

import argparse
from typing import Optional

import numpy as np

from .fail_manager import FailureManager
from .waypoint_failure import make_failure

# Module-global active manager (set by install(), read by the patched follow()).
_MANAGER: Optional[FailureManager] = None


def set_manager(manager: Optional[FailureManager]) -> None:
    global _MANAGER
    _MANAGER = manager


def get_manager() -> Optional[FailureManager]:
    return _MANAGER


def add_failure_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register the ``--failure*`` CLI args (shared by the wrapper entry point)."""
    group = parser.add_argument_group("failure injection (failgen)")
    group.add_argument(
        "--failure",
        default="none",
        help="Failure to inject: none, slip, grasp, translation[_x/_y/_z], "
        "rotation[_x/_y/_z], no_rotation, freezing.",
    )
    group.add_argument(
        "--failure-waypoint",
        type=int,
        default=-1,
        help="Waypoint INDEX to target (-1 = random / by label).",
    )
    group.add_argument(
        "--failure-label",
        default=None,
        help="Target the first waypoint whose label contains this substring "
        "(case-insensitive). Overrides --failure-waypoint.",
    )
    group.add_argument(
        "--failure-after",
        type=int,
        default=1,
        help="slip: sim steps after the target waypoint before the gripper opens.",
    )
    group.add_argument(
        "--failure-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("LOW", "HIGH"),
        help="translation (metres) / rotation (radians) perturbation range.",
    )
    group.add_argument(
        "--failure-axis",
        default=None,
        choices=[None, "x", "y", "z"],
        help="Axis for the generic 'translation'/'rotation' failure (x/y/z).",
    )
    group.add_argument(
        "--failure-freeze-steps",
        type=int,
        default=60,
        help="freezing: how many sim steps the arm stays frozen.",
    )
    group.add_argument(
        "--failure-seed",
        type=int,
        default=None,
        help="Seed the failure RNG for a reproducible perturbation.",
    )
    return parser


def build_manager_from_args(args) -> Optional[FailureManager]:
    """Build a FailureManager from parsed ``--failure*`` args, or None for 'none'."""
    name = getattr(args, "failure", "none") or "none"
    if name == "none":
        return None

    seed = getattr(args, "failure_seed", None)
    rng = np.random.RandomState(seed) if seed is not None else np.random

    waypoint = getattr(args, "failure_waypoint", -1)
    waypoints = [waypoint] if waypoint is not None and waypoint >= 0 else None
    value_range = getattr(args, "failure_range", None)

    failure = make_failure(
        name,
        name=name,
        waypoints=waypoints,
        target_label=getattr(args, "failure_label", None),
        axis=getattr(args, "failure_axis", None) or "x",
        value_range=tuple(value_range) if value_range else None,
        fail_after=getattr(args, "failure_after", 1),
        freeze_steps=getattr(args, "failure_freeze_steps", 60),
        rng=rng,
    )
    manager = FailureManager()
    manager.add(failure)
    return manager


def _patch_controller(cls) -> None:
    """Patch FrankaWaypointController.follow once (idempotent)."""
    if getattr(cls, "_failgen_patched", False):
        return
    orig_follow = cls.follow

    def follow(self, waypoints):
        manager = get_manager()
        if manager is not None and not getattr(self, "_failgen_wired", False):
            self._failgen_wired = True
            # 1. Waypoint-target failures: mutate the list before any motion.
            manager.apply(waypoints)
            print(f"[FAILGEN]: active -> {manager.describe()}")

            # 2. Step-level hooks on THIS controller instance.
            _orig_set_gripper = self._set_gripper

            def _set_gripper(width, _orig=_orig_set_gripper):
                _orig(manager.gripper(width, self))

            self._set_gripper = _set_gripper

            _orig_step = self._step

            def _step(_orig=_orig_step):
                manager.step(self)
                _orig()

            self._step = _step

        return orig_follow(self, waypoints)

    cls.follow = follow
    cls._failgen_patched = True


def install(args) -> Optional[FailureManager]:
    """Build the manager from ``args`` and patch the controller. Returns the
    manager, or None when no failure is requested (run stays clean)."""
    manager = build_manager_from_args(args)
    if manager is None:
        print("[FAILGEN]: --failure none -> clean rollout (no injection).")
        return None

    set_manager(manager)
    # robot_controller imports Isaac Lab at module load, so only import it here,
    # after run_scene.py has launched the simulator.
    import robot_controller

    _patch_controller(robot_controller.FrankaWaypointController)
    print(f"[FAILGEN]: installed failure injection -> {manager.describe()}")
    return manager
