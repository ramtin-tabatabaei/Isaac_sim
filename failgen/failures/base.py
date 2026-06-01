"""Common failure-injection interface."""


class NoFailure:
    """No-op failure injector used for successful task rollouts."""

    failure_type = "none"

    def __init__(self, **kwargs):
        pass

    def on_task_start(self, context: dict):
        pass

    def on_subgoal_start(self, label: str, context: dict):
        pass

    def on_subgoal_complete(self, label: str, context: dict):
        pass

    def update_gripper_target(self, nominal_grip: float, context: dict) -> float:
        return nominal_grip
