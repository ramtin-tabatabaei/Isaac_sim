"""
fail_manager.py

Holds the set of active :class:`WaypointFailure` instances and fans the
controller's hooks out to each one, mirroring the RLBench ``failgen`` ``Manager``
(AHA/aha/Data_Generation/rlbench-failgen/failgen/fail_manager.py) but driving a
``FrankaWaypointController`` instead of a PyRep robot.

The injector (see ``injector.py``) calls, in order:
    apply(waypoints)        once, before the motion (waypoint-target failures)
    step(controller)        once per physics step during the motion
    gripper(width, ctrl)    each time the gripper width is commanded
"""

from __future__ import annotations

from typing import List

from .waypoint_failure import WaypointFailure


class FailureManager:
    def __init__(self):
        self._failures: List[WaypointFailure] = []

    def add(self, failure: WaypointFailure) -> None:
        self._failures.append(failure)

    @property
    def failures(self) -> List[WaypointFailure]:
        return self._failures

    def clear(self) -> None:
        self._failures.clear()

    # -- controller hooks -------------------------------------------------
    def apply(self, waypoints: list) -> None:
        for failure in self._failures:
            failure.apply(waypoints)

    def step(self, controller) -> None:
        for failure in self._failures:
            failure.step(controller)

    def gripper(self, width: float, controller) -> float:
        for failure in self._failures:
            width = failure.gripper(width, controller)
        return width

    def describe(self) -> str:
        if not self._failures:
            return "no failures"
        return "; ".join(f.describe() for f in self._failures)
