"""
robot_arm.py

Reusable helper for adding a Franka Emika Panda arm to a *standalone* Isaac Lab
scene, i.e. one built by spawning raw ``sim_utils`` prims directly rather than
through an ``InteractiveScene`` (which is how ``place_usd_scene_from_context``
builds the design scene).

The robot base pose is supplied by the caller, e.g. read from an AHA
scene-context ``.md`` report. Nothing here is task specific: it only spawns the
arm and exposes a few sensible defaults that the controller module reuses.

This module imports Isaac Lab assets at import time, so it must only be imported
*after* ``AppLauncher`` has started the simulator.
"""

from __future__ import annotations

from isaaclab.assets import Articulation

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG

# Default Franka "home" arm posture (rad) and gripper opening / closing (m).
HOME_JOINTS = [0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.785]
GRIPPER_OPEN = 0.04
# Close onto (not through) the object: each finger targets a half-gap a bit
# smaller than the object's half-width so the fingers stop on it and the PD drive
# squeezes (a real friction grip). 0.02 grips the ~0.058 m sponge box; commanding
# fully shut (0.0) just shoves a light object away instead of holding it.
GRIPPER_CLOSED = 0.02

# Disabling gravity on the arm keeps scripted differential-IK tracking stable
# (no RL controller is fighting gravity here), matching the reference task in
# ``franka_basketball_in_hoop_no_failure.py``.
DISABLE_GRAVITY_FOR_STABLE_IK = True


def spawn_franka(
    prim_path: str,
    base_pos: tuple[float, float, float],
    base_quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    home_joints: list[float] = HOME_JOINTS,
    gripper_open: float = GRIPPER_OPEN,
) -> Articulation:
    """Spawn a Franka arm and return its ``Articulation``.

    Must be called while building the scene, before ``sim.reset()``. The
    articulation initialises its physics views on the first reset/play.

    Args:
        prim_path: Absolute prim path for the robot (e.g. ``/World/Robot``).
        base_pos: World-frame base position (x, y, z) in meters.
        base_quat_wxyz: World-frame base orientation as (w, x, y, z).
        home_joints: Seven arm joint targets (rad) used as the rest posture.
        gripper_open: Finger joint opening (m) at spawn.
    """
    robot_cfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path=prim_path)
    robot_cfg.spawn.rigid_props.disable_gravity = DISABLE_GRAVITY_FOR_STABLE_IK
    robot_cfg.spawn.articulation_props.solver_velocity_iteration_count = 2
    robot_cfg.init_state.pos = tuple(base_pos)
    robot_cfg.init_state.rot = tuple(base_quat_wxyz)
    robot_cfg.init_state.joint_pos = {
        "panda_joint1": home_joints[0],
        "panda_joint2": home_joints[1],
        "panda_joint3": home_joints[2],
        "panda_joint4": home_joints[3],
        "panda_joint5": home_joints[4],
        "panda_joint6": home_joints[5],
        "panda_joint7": home_joints[6],
        "panda_finger_joint.*": gripper_open,
    }
    return Articulation(cfg=robot_cfg)
