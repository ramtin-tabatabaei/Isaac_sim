"""Failure injectors used by scripts/run_failgen_task.py."""

from .base import NoFailure
from .slip import SlipFailure

FAILURE_TYPES = {
    "none": NoFailure,
    "slip": SlipFailure,
}


def make_failure(name: str, **kwargs):
    try:
        failure_cls = FAILURE_TYPES[name]
    except KeyError as exc:
        supported = ", ".join(sorted(FAILURE_TYPES))
        raise ValueError(f"Unknown failure '{name}'. Supported failures: {supported}") from exc
    return failure_cls(**kwargs)

