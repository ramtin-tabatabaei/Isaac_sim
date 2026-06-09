"""Waypoint-tied failure injection for the aha_in_isaac task runner.

Port of the RLBench ``failgen`` failures to the Isaac Lab
``FrankaWaypointController``. See ``README.md`` for usage.
"""

from .fail_manager import FailureManager
from .injector import (
    add_failure_args,
    build_manager_from_args,
    get_manager,
    install,
    set_manager,
)
from .waypoint_failure import FAILURE_TYPES, WaypointFailure, make_failure

__all__ = [
    "FailureManager",
    "WaypointFailure",
    "FAILURE_TYPES",
    "make_failure",
    "add_failure_args",
    "build_manager_from_args",
    "install",
    "get_manager",
    "set_manager",
]
