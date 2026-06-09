"""
waypoint_failure.py

Waypoint-tied failure injectors for the *aha_in_isaac* pipeline. This is the
Isaac Lab port of the RLBench ``failgen`` failures (slip / grasp / translation /
rotation / no_rotation / freezing) from

    AHA/aha/Data_Generation/rlbench-failgen/failgen/

adapted to drive a ``FrankaWaypointController`` instead of a PyRep/RLBench robot.

Two kinds of failure, by how they corrupt the rollout:

* **Waypoint-target failures** mutate the scripted ``Waypoint`` list *before* the
  motion runs (``apply``): translation perturbs ``pos_w``, rotation perturbs
  ``quat_w``, no_rotation copies the previous waypoint's orientation, grasp
  neutralises the open/close at the chosen waypoint so the object is never
  grasped. These work with every planner (diffik / rrt / curobo / rmpflow),
  because they only change the goal the controller is asked to reach.

* **Step-level failures** act *during* the motion: ``slip`` re-opens the gripper
  a few steps after a chosen waypoint (the object drops and cannot be
  re-grasped), ``freezing`` holds the arm still for a while ("robot sleeps").
  These hook the controller's per-step callbacks (``step`` / ``gripper``) and
  rely on ``controller._active_label`` to arm at the right waypoint, so they
  apply to any path that runs through ``_diffik_segment`` (diffik + the rrt /
  curobo per-segment fallback).

The target waypoint is chosen, in priority order, from:
    1. ``target_label`` (case-insensitive substring of a waypoint label),
    2. ``waypoints`` (explicit indices to pick from at random),
    3. a uniformly random waypoint.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np

# ----------------------------------------------------------------------
# Small quaternion helpers (w, x, y, z), matching the controller's EE_QUAT.
# ----------------------------------------------------------------------
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def _axis_delta_quat(axis: str, angle: float) -> tuple:
    """Unit quaternion (w,x,y,z) for a rotation of ``angle`` about world ``axis``."""
    half = 0.5 * angle
    s = math.sin(half)
    vec = [0.0, 0.0, 0.0]
    vec[_AXIS_INDEX[axis]] = s
    return (math.cos(half), vec[0], vec[1], vec[2])


def _quat_mul(a: Sequence[float], b: Sequence[float]) -> tuple:
    """Hamilton product a*b for (w,x,y,z) quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _rotate_quat(q: Sequence[float], axis: str, angle: float) -> tuple:
    """Pre-multiply ``q`` by a world-frame rotation of ``angle`` about ``axis``."""
    return _quat_mul(_axis_delta_quat(axis, angle), tuple(float(v) for v in q))


# ----------------------------------------------------------------------
# Base class.
# ----------------------------------------------------------------------
class WaypointFailure:
    """Common interface for an injected failure.

    Subclasses override some of:
        apply(waypoints)        -- mutate the waypoint list before the motion
        step(controller)        -- called once per physics step during the motion
        gripper(width, ctrl)    -- filter the commanded gripper width each step
    """

    FAILURE_TYPE = "undefined"

    def __init__(
        self,
        name: str = "",
        waypoints: Optional[Sequence[int]] = None,
        target_label: Optional[str] = None,
        rng=None,
        **kwargs,  # tolerate unused kwargs so make_failure can pass a uniform set
    ):
        self.name = name or self.FAILURE_TYPE
        self.waypoints_indices = list(waypoints) if waypoints else None
        self.target_label_query = target_label
        self.rng = rng if rng is not None else np.random
        self.target_index: Optional[int] = None
        self.target_label: Optional[str] = None

    # -- target selection -------------------------------------------------
    def _choose_target(self, waypoints: list) -> int:
        n = len(waypoints)
        if self.target_label_query:
            q = self.target_label_query.lower()
            candidates = [i for i, w in enumerate(waypoints) if q in w.label.lower()]
            if not candidates:
                print(f"[FAILGEN]: no waypoint label matches '{self.target_label_query}'; "
                      "falling back to a random waypoint.")
                candidates = list(range(n))
        elif self.waypoints_indices:
            candidates = [i for i in self.waypoints_indices if 0 <= i < n]
            if not candidates:
                print(f"[FAILGEN]: waypoint indices {self.waypoints_indices} out of range "
                      f"(have {n}); falling back to a random waypoint.")
                candidates = list(range(n))
        else:
            candidates = list(range(n))
        idx = int(self.rng.choice(candidates))
        self.target_index = idx
        self.target_label = waypoints[idx].label
        return idx

    # -- hooks (overridden as needed) ------------------------------------
    def apply(self, waypoints: list) -> None:
        self._choose_target(waypoints)

    def step(self, controller) -> None:  # noqa: D401 - simple hook
        pass

    def gripper(self, width: float, controller) -> float:
        return width

    def describe(self) -> str:
        return f"{self.FAILURE_TYPE} @ waypoint[{self.target_index}] '{self.target_label}'"


# ----------------------------------------------------------------------
# No-op (successful rollout).
# ----------------------------------------------------------------------
class NoFailure(WaypointFailure):
    FAILURE_TYPE = "none"

    def apply(self, waypoints: list) -> None:
        pass

    def describe(self) -> str:
        return "none (clean rollout)"


# ----------------------------------------------------------------------
# Translation: shift a waypoint position along one axis.
# ----------------------------------------------------------------------
class TranslationFailure(WaypointFailure):
    FAILURE_TYPE = "translation"
    AXIS: Optional[str] = None
    DEFAULT_RANGE = (-0.1, 0.1)

    def __init__(self, *, axis: str = "x", value_range=None, **kwargs):
        super().__init__(**kwargs)
        self.axis = self.AXIS or axis
        self.value_range = tuple(value_range) if value_range is not None else self.DEFAULT_RANGE

    def apply(self, waypoints: list) -> None:
        idx = self._choose_target(waypoints)
        delta = float(self.rng.uniform(self.value_range[0], self.value_range[1]))
        ai = _AXIS_INDEX[self.axis]
        wp = waypoints[idx]
        pos = list(wp.pos_w)
        pos[ai] += delta
        wp.pos_w = tuple(pos)
        # If the segment follows a recorded polyline, translate the whole path so
        # the perturbation actually moves the executed motion (the controller
        # tracks via_points_w, not pos_w, when via points are present).
        if getattr(wp, "via_points_w", None):
            wp.via_points_w = [
                tuple(p[k] + (delta if k == ai else 0.0) for k in range(3)) for p in wp.via_points_w
            ]
        print(f"[FAILGEN]: translation_{self.axis} {delta * 100:+.1f} cm at "
              f"waypoint[{idx}] '{self.target_label}'.")


class TranslationXFailure(TranslationFailure):
    FAILURE_TYPE = "translation_x"
    AXIS = "x"


class TranslationYFailure(TranslationFailure):
    FAILURE_TYPE = "translation_y"
    AXIS = "y"


class TranslationZFailure(TranslationFailure):
    FAILURE_TYPE = "translation_z"
    AXIS = "z"


# ----------------------------------------------------------------------
# Rotation: perturb a waypoint orientation about one axis.
# ----------------------------------------------------------------------
class RotationFailure(WaypointFailure):
    FAILURE_TYPE = "rotation"
    AXIS: Optional[str] = None
    DEFAULT_RANGE = (-0.1 * math.pi, 0.1 * math.pi)

    def __init__(self, *, axis: str = "x", value_range=None, **kwargs):
        super().__init__(**kwargs)
        self.axis = self.AXIS or axis
        self.value_range = tuple(value_range) if value_range is not None else self.DEFAULT_RANGE

    def apply(self, waypoints: list) -> None:
        idx = self._choose_target(waypoints)
        delta = float(self.rng.uniform(self.value_range[0], self.value_range[1]))
        wp = waypoints[idx]
        wp.quat_w = _rotate_quat(wp.quat_w, self.axis, delta)
        if getattr(wp, "via_quats_w", None):
            wp.via_quats_w = [_rotate_quat(q, self.axis, delta) for q in wp.via_quats_w]
        print(f"[FAILGEN]: rotation_{self.axis} {math.degrees(delta):+.1f} deg at "
              f"waypoint[{idx}] '{self.target_label}'.")


class RotationXFailure(RotationFailure):
    FAILURE_TYPE = "rotation_x"
    AXIS = "x"


class RotationYFailure(RotationFailure):
    FAILURE_TYPE = "rotation_y"
    AXIS = "y"


class RotationZFailure(RotationFailure):
    FAILURE_TYPE = "rotation_z"
    AXIS = "z"


# ----------------------------------------------------------------------
# No-rotation: keep the previous waypoint's orientation (skip the reorient).
# ----------------------------------------------------------------------
class NoRotationFailure(WaypointFailure):
    FAILURE_TYPE = "no_rotation"

    def apply(self, waypoints: list) -> None:
        idx = self._choose_target(waypoints)
        if idx <= 0:
            print("[FAILGEN]: no_rotation needs a previous waypoint; no-op at waypoint[0].")
            return
        prev_quat = waypoints[idx - 1].quat_w
        wp = waypoints[idx]
        wp.quat_w = tuple(float(v) for v in prev_quat)
        wp.via_quats_w = None  # hold the previous orientation across the segment
        print(f"[FAILGEN]: no_rotation at waypoint[{idx}] '{self.target_label}' "
              f"(reuse waypoint[{idx - 1}] orientation).")


# ----------------------------------------------------------------------
# Grasp: neutralise the open/close at the chosen waypoint so it never grasps.
# ----------------------------------------------------------------------
class GraspFailure(WaypointFailure):
    FAILURE_TYPE = "grasp"

    def apply(self, waypoints: list) -> None:
        idx = self._choose_target(waypoints)
        prev_grip = waypoints[idx - 1].gripper if idx > 0 else "open"
        wp = waypoints[idx]
        old = wp.gripper
        wp.gripper = prev_grip
        print(f"[FAILGEN]: grasp failure at waypoint[{idx}] '{self.target_label}' "
              f"(gripper {old!r} -> hold {prev_grip!r}; object never secured).")


# ----------------------------------------------------------------------
# Slip: re-open the gripper a few steps after reaching a waypoint.
# ----------------------------------------------------------------------
class SlipFailure(WaypointFailure):
    FAILURE_TYPE = "slip"

    _IDLE, _PRE, _POST = 0, 1, 2

    def __init__(self, *, fail_after: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.fail_after = max(0, int(fail_after))
        self._state = self._IDLE
        self._counter = 0

    def apply(self, waypoints: list) -> None:
        # Only resolve the arming waypoint; the slip happens during the motion.
        self._choose_target(waypoints)

    def step(self, controller) -> None:
        if self._state == self._IDLE:
            if controller._active_label == self.target_label:
                self._state = self._PRE
                self._counter = 0
                print(f"[FAILGEN]: slip armed at '{self.target_label}'; "
                      f"opening gripper in {self.fail_after} steps.")
        elif self._state == self._PRE:
            self._counter += 1
            if self._counter >= self.fail_after:
                self._state = self._POST
                print("[FAILGEN]: slip triggered -> releasing the object now.")

    def gripper(self, width: float, controller) -> float:
        if self._state == self._POST:
            # Force (and keep) the gripper open so it cannot re-grasp.
            return controller.gripper_open
        return width


# ----------------------------------------------------------------------
# Freezing: hold the arm still for a while after a waypoint ("robot sleeps").
# ----------------------------------------------------------------------
class FreezingFailure(WaypointFailure):
    FAILURE_TYPE = "freezing"

    _IDLE, _ARMED, _FROZEN, _RELEASED = 0, 1, 2, 3

    def __init__(self, *, freeze_after_range=(3, 10), freeze_steps: int = 60, **kwargs):
        super().__init__(**kwargs)
        self.freeze_after_range = tuple(freeze_after_range)
        self.freeze_steps = max(1, int(freeze_steps))
        self._state = self._IDLE
        self._counter = 0
        self._frozen_counter = 0
        self._steps_until_freeze = self._sample_until()
        self._hold_pos = None

    def _sample_until(self) -> int:
        lo, hi = self.freeze_after_range
        lo = max(1, int(lo))
        hi = max(lo, int(hi))
        return int(self.rng.randint(lo, hi + 1))

    def apply(self, waypoints: list) -> None:
        self._choose_target(waypoints)

    def step(self, controller) -> None:
        if self._state == self._IDLE:
            if controller._active_label == self.target_label:
                self._state = self._ARMED
                self._counter = 0
                self._steps_until_freeze = self._sample_until()
                print(f"[FAILGEN]: freezing armed at '{self.target_label}'; "
                      f"will freeze in {self._steps_until_freeze} steps.")
        elif self._state == self._ARMED:
            self._counter += 1
            if self._counter >= self._steps_until_freeze:
                self._state = self._FROZEN
                self._frozen_counter = 0
                self._hold_pos = controller.robot.data.joint_pos[:, controller.arm_joint_ids].clone()
                print(f"[FAILGEN]: freezing -> robot sleeps for {self.freeze_steps} steps.")

        if self._state == self._FROZEN:
            # Override the arm target with the captured pose right before the sim
            # step, so the arm holds still regardless of what the motion commanded.
            controller.robot.set_joint_position_target(
                self._hold_pos, joint_ids=controller.arm_joint_ids
            )
            self._frozen_counter += 1
            if self._frozen_counter > self.freeze_steps:
                self._state = self._RELEASED
                print("[FAILGEN]: freezing done -> robot awake, continuing.")


# ----------------------------------------------------------------------
# Registry + factory.
# ----------------------------------------------------------------------
FAILURE_TYPES = {
    cls.FAILURE_TYPE: cls
    for cls in (
        NoFailure,
        SlipFailure,
        GraspFailure,
        TranslationFailure,
        TranslationXFailure,
        TranslationYFailure,
        TranslationZFailure,
        RotationFailure,
        RotationXFailure,
        RotationYFailure,
        RotationZFailure,
        NoRotationFailure,
        FreezingFailure,
    )
}


def make_failure(failure_type: str, **kwargs) -> WaypointFailure:
    try:
        failure_cls = FAILURE_TYPES[failure_type]
    except KeyError as exc:
        supported = ", ".join(sorted(FAILURE_TYPES))
        raise ValueError(
            f"Unknown failure '{failure_type}'. Supported failures: {supported}"
        ) from exc
    return failure_cls(**kwargs)
