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

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim.utils import make_uninstanceable
from pxr import Usd, UsdPhysics, UsdShade

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
LOCAL_FRANKA_USD = (
    Path(__file__).resolve().parents[2]
    / "source"
    / "isaaclab_assets"
    / "data"
    / "Robots"
    / "FrankaEmika"
    / "panda_instanceable.usd"
)


def configure_franka_gripper_friction(
    prim_path: str,
    static_friction: float,
    dynamic_friction: float,
) -> int:
    """Bind a real Coulomb-friction material to the editable finger colliders.

    The bundled Franka references instanceable collision meshes. Material binding on
    an instance proxy is ignored, so the finger subtrees must be made uninstanceable
    before the first simulation reset, while PhysX is still reading the USD stage.
    """
    stage = sim_utils.get_current_stage()
    material_path = f"{prim_path}/GripperPhysicsMaterial"
    material_cfg = sim_utils.RigidBodyMaterialCfg(
        static_friction=float(static_friction),
        dynamic_friction=float(dynamic_friction),
        restitution=0.0,
        friction_combine_mode="max",
        restitution_combine_mode="multiply",
    )
    material_cfg.func(material_path, material_cfg)
    material = UsdShade.Material(stage.GetPrimAtPath(material_path))

    for finger in ("panda_leftfinger", "panda_rightfinger"):
        make_uninstanceable(f"{prim_path}/{finger}", stage=stage)

    bound = 0
    root = stage.GetPrimAtPath(prim_path)
    for prim in Usd.PrimRange(root):
        path = str(prim.GetPath())
        if not (
            ("panda_leftfinger" in path or "panda_rightfinger" in path)
            and prim.HasAPI(UsdPhysics.CollisionAPI)
        ):
            continue
        binding = UsdShade.MaterialBindingAPI.Apply(prim)
        binding.Bind(
            material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            materialPurpose="physics",
        )
        targets = binding.GetDirectBindingRel("physics").GetTargets()
        if material.GetPath() in targets:
            bound += 1

    if bound != 2:
        raise RuntimeError(
            f"Expected 2 Franka finger colliders under '{prim_path}', but bound material to {bound}."
        )
    print(
        f"[INFO]: Bound gripper friction material to {bound} real finger colliders "
        f"(static={static_friction:g}, dynamic={dynamic_friction:g})."
    )
    return bound


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
    if LOCAL_FRANKA_USD.is_file():
        robot_cfg.spawn.usd_path = str(LOCAL_FRANKA_USD)
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
