"""
Run a clean IsaacLab task with an optional injected failure.

Examples:
    ./isaaclab.sh -p scripts/run_failgen_task.py --task basketball_in_hoop --failure none
    ./isaaclab.sh -p scripts/run_failgen_task.py --task basketball_in_hoop --failure slip
    ./isaaclab.sh -p scripts/run_failgen_task.py --task basketball_in_hoop --failure slip --failure-after-subgoal "approach waypoint3 above hoop"
"""

import argparse

from isaaclab.app import AppLauncher


def parse_args():
    parser = argparse.ArgumentParser(description="Run IsaacLab tasks with optional failure injection.")
    parser.add_argument("--task", default="basketball_in_hoop", help="Task name to run.")
    parser.add_argument("--failure", default="none", help="Failure type to inject, or 'none'.")
    parser.add_argument(
        "--failure-after-subgoal",
        default="Lift straight up",
        help="Subgoal label after which a delayed failure should arm.",
    )
    parser.add_argument(
        "--failure-delay-steps",
        type=int,
        default=40,
        help="Number of simulator steps to wait after the arming subgoal before triggering failure.",
    )
    parser.add_argument(
        "--allow-regrasp",
        action="store_true",
        help="For slip failures, allow later task commands to close the gripper again.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    from failgen.failures import FAILURE_TYPES, make_failure
    from failgen.tasks import TASK_TYPES, load_task

    if args.task not in TASK_TYPES:
        supported = ", ".join(sorted(TASK_TYPES))
        raise ValueError(f"Unknown task '{args.task}'. Supported tasks: {supported}")
    if args.failure not in FAILURE_TYPES:
        supported = ", ".join(sorted(FAILURE_TYPES))
        raise ValueError(f"Unknown failure '{args.failure}'. Supported failures: {supported}")

    task_module = load_task(args.task)
    failure = make_failure(
        args.failure,
        arm_after_subgoal=args.failure_after_subgoal,
        steps_after_subgoal=args.failure_delay_steps,
        keep_open=not args.allow_regrasp,
    )

    print(f"[INFO]: Running task='{args.task}' with failure='{args.failure}'.")
    task_module.run_task(simulation_app, failure=failure)
    simulation_app.close()


if __name__ == "__main__":
    main()
