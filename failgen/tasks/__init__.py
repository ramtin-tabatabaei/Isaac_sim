"""Task registry used by scripts/run_failgen_task.py."""

import importlib

TASK_TYPES = {
    "basketball_in_hoop": "failgen.tasks.basketball_in_hoop",
}


def load_task(name: str):
    try:
        module_name = TASK_TYPES[name]
    except KeyError as exc:
        supported = ", ".join(sorted(TASK_TYPES))
        raise ValueError(f"Unknown task '{name}'. Supported tasks: {supported}") from exc
    return importlib.import_module(module_name)

