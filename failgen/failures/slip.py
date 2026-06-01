"""Slip failure: open the gripper shortly after a selected subgoal."""

from enum import Enum


class SlipState(Enum):
    IDLE = 0
    PRE_FAIL = 1
    POST_FAIL = 2


class SlipFailure:
    """Open the gripper a few sim steps after a chosen subgoal."""

    failure_type = "slip"

    def __init__(
        self,
        arm_after_subgoal: str = "Lift straight up",
        steps_after_subgoal: int = 40,
        open_width: float | None = None,
        keep_open: bool = True,
    ):
        self.arm_after_subgoal = arm_after_subgoal
        self.steps_after_subgoal = steps_after_subgoal
        self.open_width = open_width
        self.keep_open = keep_open
        self.state = SlipState.IDLE
        self.steps_counter = 0

    def on_task_start(self, context: dict):
        if self.open_width is None:
            self.open_width = context["gripper_open"]
        print(
            f"[SLIP DEBUG]: ready, arm_after='{self.arm_after_subgoal}', "
            f"delay_steps={self.steps_after_subgoal}, open_width={self.open_width}."
        )

    def on_subgoal_start(self, label: str, context: dict):
        pass

    def on_subgoal_complete(self, label: str, context: dict):
        if self.state != SlipState.IDLE:
            return
        if label == self.arm_after_subgoal:
            self.state = SlipState.PRE_FAIL
            self.steps_counter = 0
            print(
                f"[SLIP DEBUG]: armed after '{label}'. "
                f"Will open gripper in {self.steps_after_subgoal} sim steps."
            )

    def update_gripper_target(self, nominal_grip: float, context: dict) -> float:
        if self.state == SlipState.PRE_FAIL:
            self.steps_counter += 1
            if self.steps_counter >= self.steps_after_subgoal:
                self.state = SlipState.POST_FAIL
                print("[SLIP DEBUG]: triggered. Opening gripper now.")
                return self.open_width

        if self.state == SlipState.POST_FAIL and self.keep_open:
            return self.open_width

        return nominal_grip

