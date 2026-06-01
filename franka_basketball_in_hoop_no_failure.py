"""
franka_basketball_in_hoop_no_failure.py

Spawns a Franka Emika Panda arm, a ball, and a simple hoop, then executes the
no-failure variant of the basketball-in-hoop task inspired by:
    /home/ramtin/AHA/ttm_inspection_reports/001_basketball_in_hoop.llm_context.md

Run with:
    cd ~/IsaacLab
    ./isaaclab.sh -p scripts/franka_basketball_in_hoop_no_failure.py
"""

import argparse
import math
from enum import Enum

from isaaclab.app import AppLauncher

# ----------------------------------------------------------------------
# 1. Launch Isaac Sim before importing the rest of Isaac Lab.
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Franka basketball-in-hoop no-failure task.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ----------------------------------------------------------------------
# 2. Safe to import the rest now.
# ----------------------------------------------------------------------
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_slerp, subtract_frame_transforms
from pxr import UsdPhysics

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG

# ----------------------------------------------------------------------
# 3. Task parameters.
# ----------------------------------------------------------------------
TABLE_HEIGHT = 1.0
TABLE_SIZE = (1.6, 1.9, TABLE_HEIGHT)
GRAVITY_XYZ = (0.0, 0.0, -9.81)
ROBOT_DISABLE_GRAVITY_FOR_STABLE_IK = True

BALL_RADIUS = 0.025
BALL_MASS = 0.035
HOOP_SUPPORT_HEIGHT = 0.24

# Placement from the TTM context, shifted into a comfortable Franka workspace.
BALL_POS_XYZ = [0.42, -0.10, TABLE_HEIGHT + BALL_RADIUS]
HOOP_OFFSET_FROM_BALL_XYZ = [0.28, 0.12, HOOP_SUPPORT_HEIGHT - BALL_RADIUS]
HOOP_CENTER_XYZ = [
    BALL_POS_XYZ[0] + HOOP_OFFSET_FROM_BALL_XYZ[0],
    BALL_POS_XYZ[1] + HOOP_OFFSET_FROM_BALL_XYZ[1],
    BALL_POS_XYZ[2] + HOOP_OFFSET_FROM_BALL_XYZ[2],
]

HOOP_OPENING = 0.145
HOOP_RIM_RADIUS = HOOP_OPENING / 2.0
HOOP_RIM_THICKNESS = 0.012
HOOP_RIM_SEGMENTS = 12
HOOP_RIM_SEGMENT_LENGTH = 2.0 * math.pi * HOOP_RIM_RADIUS / HOOP_RIM_SEGMENTS * 0.94
HOOP_BACKBOARD_SIZE = (0.014, 0.30, 0.22)
HOOP_BACKBOARD_OFFSET_X = HOOP_RIM_RADIUS + 0.035
HOOP_BACKBOARD_OFFSET_Z = 0.07
HOOP_BACKBOARD_TARGET_SIZE_YZ = (0.12, 0.085)
HOOP_BACKBOARD_TARGET_LINE_THICKNESS = 0.008
HOOP_NET_LENGTH = 0.09
HOOP_NET_CORD_RADIUS = 0.0018

# The task context says ball x/y are sampled uniformly and z is fixed. This
# script uses the deterministic BALL_POS_XYZ above, but these bounds are here
# so random placement can be added by sampling from them.
BALL_X_BOUNDS = (-0.078429, 0.343683)
BALL_Y_BOUNDS = (-0.261317, 0.310108)
BALL_Z_FIXED = 0.781336

HOME_JOINTS = [0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.785]
GRIPPER_OPEN = 0.04
GRIPPER_CLOSED = 0.014
HAND_TO_BALL_Z = 0.088
BALL_PREGRASP_OFFSET_Z = 0.20
BALL_GRASP_OFFSET_Z = HAND_TO_BALL_Z
LIFT_AFTER_GRASP_DELTA_Z = 0.26
MIN_BALL_LIFT_FOR_HOOP = 0.06
HOOP_HIGH_CLEARANCE_Z = 0.32
HOOP_RELEASE_CLEARANCE_Z = 0.15
HOOP_RETREAT_OFFSET_X = -0.16
HOOP_RETREAT_CLEARANCE_Z = 0.35

# Failure injection is disabled for this no-failure variant. The normal task
# sequence still opens the gripper only at the planned release above the hoop.
FAILURE_ENABLED = False
FAILURE_TYPE = "none"
FAILURE_ARM_AFTER_SUBGOAL = "Lift straight up"
FAILURE_STEPS_AFTER_SUBGOAL = 40
FAILURE_OPEN_GRIPPER = GRIPPER_OPEN
FAILURE_KEEP_GRIPPER_OPEN = True

# Hand points down toward the table/ball.
EE_QUAT_DOWN = [0.0, 1.0, 0.0, 0.0]

HIGH_FRICTION_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    static_friction=2.4,
    dynamic_friction=1.8,
    restitution=0.0,
    friction_combine_mode="max",
    restitution_combine_mode="multiply",
)
BALL_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    static_friction=3.4,
    dynamic_friction=2.8,
    restitution=0.0,
    friction_combine_mode="max",
    restitution_combine_mode="multiply",
)
GRIPPER_FRICTION_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    static_friction=4.0,
    dynamic_friction=3.2,
    restitution=0.0,
    friction_combine_mode="max",
    restitution_combine_mode="multiply",
)
CONTACT_PROPS = sim_utils.CollisionPropertiesCfg(
    contact_offset=0.004,
    rest_offset=0.0,
)

# Subgoal parameters are relative to the live ball pose, the hoop center, or the
# previous robot target. The ball is never teleported or attached in code: the
# gripper must pick it up through real contact and friction.
SUBGOAL_PARAMS = [
    {
        "label": "Approach ball",
        "reference": "ball",
        "offset_xyz": [0.0, 0.0, BALL_PREGRASP_OFFSET_Z],
        "gripper": "open",
        "duration_steps": 220,
    },
    {
        "label": "Lower to ball",
        "reference": "ball",
        "offset_xyz": [0.0, 0.0, BALL_GRASP_OFFSET_Z],
        "gripper": "open",
        "duration_steps": 160,
    },
    {
        "label": "Close gripper",
        "reference": "ball",
        "offset_xyz": [0.0, 0.0, BALL_GRASP_OFFSET_Z],
        "gripper": "closed",
        "duration_steps": 140,
    },
    {
        "label": "Lift straight up",
        "reference": "previous",
        "delta_xyz": [0.0, 0.0, LIFT_AFTER_GRASP_DELTA_Z],
        "gripper": "closed",
        "duration_steps": 260,
    },
    {
        "label": "Move above hoop high",
        "reference": "hoop",
        "offset_xyz": [0.0, 0.0, HOOP_HIGH_CLEARANCE_Z],
        "gripper": "closed",
        "duration_steps": 320,
    },
    {
        "label": "Lower over hoop",
        "reference": "hoop",
        "offset_xyz": [0.0, 0.0, HOOP_RELEASE_CLEARANCE_Z],
        "gripper": "closed",
        "duration_steps": 180,
    },
    {
        "label": "Release above hoop",
        "reference": "hoop",
        "offset_xyz": [0.0, 0.0, HOOP_RELEASE_CLEARANCE_Z],
        "gripper": "open",
        "duration_steps": 80,
    },
    {
        "label": "Lift gripper clear",
        "reference": "hoop",
        "offset_xyz": [0.0, 0.0, HOOP_HIGH_CLEARANCE_Z],
        "gripper": "open",
        "duration_steps": 150,
    },
    {
        "label": "Retreat",
        "reference": "hoop",
        "offset_xyz": [HOOP_RETREAT_OFFSET_X, 0.0, HOOP_RETREAT_CLEARANCE_Z],
        "gripper": "open",
        "duration_steps": 200,
    },
]


def _gripper_width(state: str) -> float:
    return GRIPPER_OPEN if state == "open" else GRIPPER_CLOSED


class SlipState(Enum):
    IDLE = 0
    PRE_FAIL = 1
    POST_FAIL = 2


class SlipFailure:
    """Open the gripper a few sim steps after a chosen subgoal."""

    def __init__(
        self,
        enabled: bool,
        arm_after_subgoal: str,
        steps_after_subgoal: int,
        open_width: float,
        keep_open: bool,
    ):
        self.enabled = enabled
        self.arm_after_subgoal = arm_after_subgoal
        self.steps_after_subgoal = steps_after_subgoal
        self.open_width = open_width
        self.keep_open = keep_open
        self.state = SlipState.IDLE
        self.steps_counter = 0

    def on_subgoal_complete(self, label: str):
        if not self.enabled or self.state != SlipState.IDLE:
            return
        if label == self.arm_after_subgoal:
            self.state = SlipState.PRE_FAIL
            self.steps_counter = 0
            print(
                f"[SLIP DEBUG]: Armed after '{label}'. "
                f"Will open gripper in {self.steps_after_subgoal} sim steps."
            )

    def update_target_grip(self, nominal_grip: float) -> float:
        if not self.enabled:
            return nominal_grip

        if self.state == SlipState.PRE_FAIL:
            self.steps_counter += 1
            if self.steps_counter >= self.steps_after_subgoal:
                self.state = SlipState.POST_FAIL
                print("[SLIP DEBUG]: Triggered slip failure. Opening gripper now.")
                return self.open_width

        if self.state == SlipState.POST_FAIL and self.keep_open:
            return self.open_width

        return nominal_grip


def _yaw_quat(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))


def _rim_segment_pose(index: int) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    angle = 2.0 * math.pi * index / HOOP_RIM_SEGMENTS
    pos = (
        HOOP_CENTER_XYZ[0] + HOOP_RIM_RADIUS * math.cos(angle),
        HOOP_CENTER_XYZ[1] + HOOP_RIM_RADIUS * math.sin(angle),
        HOOP_CENTER_XYZ[2],
    )
    return pos, _yaw_quat(angle + math.pi / 2.0)


def _net_cord_pose(index: int) -> tuple[float, float, float]:
    angle = 2.0 * math.pi * index / HOOP_RIM_SEGMENTS
    return (
        HOOP_CENTER_XYZ[0] + HOOP_RIM_RADIUS * math.cos(angle),
        HOOP_CENTER_XYZ[1] + HOOP_RIM_RADIUS * math.sin(angle),
        HOOP_CENTER_XYZ[2] - HOOP_NET_LENGTH / 2.0,
    )


def _make_rim_segment_cfg(index: int) -> RigidObjectCfg:
    pos, rot = _rim_segment_pose(index)
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/HoopRim{index:02d}",
        spawn=sim_utils.CylinderCfg(
            radius=HOOP_RIM_THICKNESS / 2.0,
            height=HOOP_RIM_SEGMENT_LENGTH,
            axis="X",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=False),
            collision_props=CONTACT_PROPS,
            physics_material=HIGH_FRICTION_MATERIAL,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.28, 0.02), roughness=0.75),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
    )


def _make_net_cord_cfg(index: int) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/HoopNetCord{index:02d}",
        spawn=sim_utils.CapsuleCfg(
            radius=HOOP_NET_CORD_RADIUS,
            height=HOOP_NET_LENGTH,
            axis="Z",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.96, 0.96, 0.90), roughness=0.95),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=_net_cord_pose(index)),
    )


def _backboard_center() -> tuple[float, float, float]:
    return (
        HOOP_CENTER_XYZ[0] + HOOP_BACKBOARD_OFFSET_X,
        HOOP_CENTER_XYZ[1],
        HOOP_CENTER_XYZ[2] + HOOP_BACKBOARD_OFFSET_Z,
    )


def _backboard_target_pos(offset_y: float, offset_z: float) -> tuple[float, float, float]:
    center_x, center_y, center_z = _backboard_center()
    return (
        center_x - HOOP_BACKBOARD_SIZE[0] / 2.0 - 0.002,
        center_y + offset_y,
        center_z + offset_z,
    )


# ----------------------------------------------------------------------
# 4. Scene configuration.
# ----------------------------------------------------------------------
@configclass
class FrankaBasketballSceneCfg(InteractiveSceneCfg):
    """Ground plane, light, table, ball, hoop, and Franka."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9)),
    )

    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=False),
            collision_props=CONTACT_PROPS,
            physics_material=HIGH_FRICTION_MATERIAL,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.30, 0.20)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, TABLE_HEIGHT / 2.0)),
    )

    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=BALL_RADIUS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=2,
                max_depenetration_velocity=2.0,
            ),
            collision_props=CONTACT_PROPS,
            physics_material=BALL_MATERIAL,
            mass_props=sim_utils.MassPropertiesCfg(mass=BALL_MASS),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.33, 0.03), roughness=0.9),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(BALL_POS_XYZ)),
    )

    ball_seam_vertical = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Ball/SeamVertical",
        spawn=sim_utils.CuboidCfg(
            size=(0.004, 0.004, BALL_RADIUS * 1.85),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.015, 0.012, 0.010), roughness=0.95),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(BALL_RADIUS * 0.90, BALL_RADIUS * 0.18, 0.0)),
    )

    ball_seam_horizontal = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Ball/SeamHorizontal",
        spawn=sim_utils.CuboidCfg(
            size=(0.004, BALL_RADIUS * 1.85, 0.004),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.015, 0.012, 0.010), roughness=0.95),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(BALL_RADIUS * 0.90, 0.0, BALL_RADIUS * 0.18)),
    )

    ball_seam_left_curve = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Ball/SeamLeftCurve",
        spawn=sim_utils.CuboidCfg(
            size=(0.004, 0.004, BALL_RADIUS * 1.45),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.015, 0.012, 0.010), roughness=0.95),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(BALL_RADIUS * 0.91, -BALL_RADIUS * 0.45, 0.0)),
    )

    ball_seam_right_curve = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Ball/SeamRightCurve",
        spawn=sim_utils.CuboidCfg(
            size=(0.004, 0.004, BALL_RADIUS * 1.45),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.015, 0.012, 0.010), roughness=0.95),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(BALL_RADIUS * 0.91, BALL_RADIUS * 0.45, 0.0)),
    )

    backboard = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/HoopBackboard",
        spawn=sim_utils.CuboidCfg(
            size=HOOP_BACKBOARD_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=False),
            collision_props=CONTACT_PROPS,
            physics_material=HIGH_FRICTION_MATERIAL,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0), roughness=0.35, opacity=1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=_backboard_center()),
    )

    backboard_target_top = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BackboardTargetTop",
        spawn=sim_utils.CuboidCfg(
            size=(0.004, HOOP_BACKBOARD_TARGET_SIZE_YZ[0], HOOP_BACKBOARD_TARGET_LINE_THICKNESS),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.18, 0.02), roughness=0.75),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=_backboard_target_pos(0.0, HOOP_BACKBOARD_TARGET_SIZE_YZ[1] / 2.0),
        ),
    )

    backboard_target_bottom = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BackboardTargetBottom",
        spawn=sim_utils.CuboidCfg(
            size=(0.004, HOOP_BACKBOARD_TARGET_SIZE_YZ[0], HOOP_BACKBOARD_TARGET_LINE_THICKNESS),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.18, 0.02), roughness=0.75),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=_backboard_target_pos(0.0, -HOOP_BACKBOARD_TARGET_SIZE_YZ[1] / 2.0),
        ),
    )

    backboard_target_left = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BackboardTargetLeft",
        spawn=sim_utils.CuboidCfg(
            size=(0.004, HOOP_BACKBOARD_TARGET_LINE_THICKNESS, HOOP_BACKBOARD_TARGET_SIZE_YZ[1]),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.18, 0.02), roughness=0.75),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=_backboard_target_pos(-HOOP_BACKBOARD_TARGET_SIZE_YZ[0] / 2.0, 0.0),
        ),
    )

    backboard_target_right = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BackboardTargetRight",
        spawn=sim_utils.CuboidCfg(
            size=(0.004, HOOP_BACKBOARD_TARGET_LINE_THICKNESS, HOOP_BACKBOARD_TARGET_SIZE_YZ[1]),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.18, 0.02), roughness=0.75),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=_backboard_target_pos(HOOP_BACKBOARD_TARGET_SIZE_YZ[0] / 2.0, 0.0),
        ),
    )

    rim_00 = _make_rim_segment_cfg(0)
    rim_01 = _make_rim_segment_cfg(1)
    rim_02 = _make_rim_segment_cfg(2)
    rim_03 = _make_rim_segment_cfg(3)
    rim_04 = _make_rim_segment_cfg(4)
    rim_05 = _make_rim_segment_cfg(5)
    rim_06 = _make_rim_segment_cfg(6)
    rim_07 = _make_rim_segment_cfg(7)
    rim_08 = _make_rim_segment_cfg(8)
    rim_09 = _make_rim_segment_cfg(9)
    rim_10 = _make_rim_segment_cfg(10)
    rim_11 = _make_rim_segment_cfg(11)

    net_00 = _make_net_cord_cfg(0)
    net_01 = _make_net_cord_cfg(1)
    net_02 = _make_net_cord_cfg(2)
    net_03 = _make_net_cord_cfg(3)
    net_04 = _make_net_cord_cfg(4)
    net_05 = _make_net_cord_cfg(5)
    net_06 = _make_net_cord_cfg(6)
    net_07 = _make_net_cord_cfg(7)
    net_08 = _make_net_cord_cfg(8)
    net_09 = _make_net_cord_cfg(9)
    net_10 = _make_net_cord_cfg(10)
    net_11 = _make_net_cord_cfg(11)

    hoop_post = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/HoopPost",
        spawn=sim_utils.CylinderCfg(
            radius=0.012,
            height=HOOP_SUPPORT_HEIGHT,
            axis="Z",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=False),
            collision_props=CONTACT_PROPS,
            physics_material=HIGH_FRICTION_MATERIAL,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(
                HOOP_CENTER_XYZ[0] + HOOP_BACKBOARD_OFFSET_X,
                HOOP_CENTER_XYZ[1],
                HOOP_CENTER_XYZ[2] - HOOP_SUPPORT_HEIGHT / 2.0,
            ),
        ),
    )

    robot: Articulation = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.spawn.rigid_props.disable_gravity = ROBOT_DISABLE_GRAVITY_FOR_STABLE_IK
    robot.spawn.articulation_props.solver_velocity_iteration_count = 2
    robot.init_state.pos = (0.0, 0.0, TABLE_HEIGHT)
    robot.init_state.joint_pos = {
        "panda_joint1": HOME_JOINTS[0],
        "panda_joint2": HOME_JOINTS[1],
        "panda_joint3": HOME_JOINTS[2],
        "panda_joint4": HOME_JOINTS[3],
        "panda_joint5": HOME_JOINTS[4],
        "panda_joint6": HOME_JOINTS[5],
        "panda_joint7": HOME_JOINTS[6],
        "panda_finger_joint.*": GRIPPER_OPEN,
    }


def _repeat(values: list[float], num_envs: int, device: str) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32, device=device).unsqueeze(0).repeat(num_envs, 1)


def _smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def _quat_slerp_batch(start_quat: torch.Tensor, target_quat: torch.Tensor, t: float) -> torch.Tensor:
    return torch.stack([quat_slerp(q0, q1, t) for q0, q1 in zip(start_quat, target_quat)])


def _make_subgoal_markers() -> VisualizationMarkers:
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/BasketballSubgoals",
        markers={
            "active": sim_utils.SphereCfg(
                radius=0.012,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.35, 1.0)),
            ),
        },
    )
    return VisualizationMarkers(marker_cfg)


def _base_to_world_pos(target_pos_b: torch.Tensor, env_origins: torch.Tensor) -> torch.Tensor:
    target_pos_w = target_pos_b.clone()
    target_pos_w[:, 2] += TABLE_HEIGHT
    return target_pos_w + env_origins


def _object_pos_b(scene: InteractiveScene, object_name: str) -> torch.Tensor:
    object_pos_w = scene[object_name].data.root_pos_w
    object_pos_b = object_pos_w - scene.env_origins
    object_pos_b[:, 2] -= TABLE_HEIGHT
    return object_pos_b


def _hoop_center_b(num_envs: int, device: str) -> torch.Tensor:
    hoop_center_b = torch.tensor(HOOP_CENTER_XYZ, dtype=torch.float32, device=device).unsqueeze(0).repeat(num_envs, 1)
    hoop_center_b[:, 2] -= TABLE_HEIGHT
    return hoop_center_b


def _resolve_subgoal_target(
    params: dict,
    scene: InteractiveScene,
    previous_target_b: torch.Tensor,
    device: str,
) -> torch.Tensor:
    reference = params["reference"]
    if reference == "ball":
        target_pos_b = _object_pos_b(scene, "ball")
        offset = _repeat(params["offset_xyz"], scene.num_envs, device)
        return target_pos_b + offset
    if reference == "hoop":
        target_pos_b = _hoop_center_b(scene.num_envs, device)
        offset = _repeat(params["offset_xyz"], scene.num_envs, device)
        return target_pos_b + offset
    if reference == "previous":
        delta = _repeat(params["delta_xyz"], scene.num_envs, device)
        return previous_target_b + delta
    raise ValueError(f"Unsupported subgoal reference: {reference}")


def _print_subgoals():
    print("[INFO]: Basketball subgoals are resolved relative to objects at runtime:")
    for index, params in enumerate(SUBGOAL_PARAMS):
        gripper_label = params["gripper"]
        if params["reference"] == "previous":
            relation = f"previous + {tuple(params['delta_xyz'])}"
        else:
            relation = f"{params['reference']} + {tuple(params['offset_xyz'])}"
        print(
            f"  {index}: {params['label']}: target={relation}, "
            f"gripper={gripper_label}, steps={params['duration_steps']}"
        )


def _set_gripper(robot: Articulation, gripper_joint_ids: list[int], width: float, num_envs: int):
    target = torch.full((num_envs, len(gripper_joint_ids)), width, dtype=torch.float32, device=robot.device)
    robot.set_joint_position_target(target, joint_ids=gripper_joint_ids)


def _get_ee_pose_b(robot: Articulation, ee_body_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    ee_pose_w = robot.data.body_pose_w[:, ee_body_id]
    root_pose_w = robot.data.root_pose_w
    return subtract_frame_transforms(
        root_pose_w[:, 0:3],
        root_pose_w[:, 3:7],
        ee_pose_w[:, 0:3],
        ee_pose_w[:, 3:7],
    )


def _write_home_state(robot: Articulation):
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = torch.zeros_like(joint_pos)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.set_joint_position_target(joint_pos)
    robot.reset()


def _step(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())


def _apply_gripper_friction():
    material_path = "/World/PhysicsMaterials/GripperHighFriction"
    GRIPPER_FRICTION_MATERIAL.func(material_path, GRIPPER_FRICTION_MATERIAL)

    stage = sim_utils.get_current_stage()
    bound_paths = set()
    bound_count = 0
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        is_finger = "panda_leftfinger" in prim_path or "panda_rightfinger" in prim_path
        looks_like_collision = "collision" in prim_path.lower() or prim.GetName() in {
            "panda_leftfinger",
            "panda_rightfinger",
        }
        if is_finger and (prim.HasAPI(UsdPhysics.CollisionAPI) or looks_like_collision) and prim_path not in bound_paths:
            sim_utils.bind_physics_material(prim_path, material_path)
            bound_paths.add(prim_path)
            bound_count += 1

    print(f"[INFO]: Applied high-friction gripper material to {bound_count} fingertip prims.")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    robot = scene["robot"]
    ball = scene["ball"]
    num_envs = scene.num_envs

    arm_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    arm_entity_cfg.resolve(scene)
    gripper_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"])
    gripper_entity_cfg.resolve(scene)

    ee_jacobi_idx = arm_entity_cfg.body_ids[0] - 1 if robot.is_fixed_base else arm_entity_cfg.body_ids[0]

    diff_ik_controller = DifferentialIKController(
        DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        num_envs=num_envs,
        device=robot.device,
    )

    _write_home_state(robot)
    scene.reset()
    _apply_gripper_friction()
    _set_gripper(robot, gripper_entity_cfg.joint_ids, GRIPPER_OPEN, num_envs)

    for _ in range(60):
        _step(sim, scene)
    diff_ik_controller.reset()

    subgoal_markers = _make_subgoal_markers()
    _print_subgoals()
    print(
        f"[INFO]: Failure injection enabled={FAILURE_ENABLED}, type={FAILURE_TYPE}, "
        f"arm_after='{FAILURE_ARM_AFTER_SUBGOAL}', delay_steps={FAILURE_STEPS_AFTER_SUBGOAL}."
    )

    current_pos, current_quat = _get_ee_pose_b(robot, arm_entity_cfg.body_ids[0])
    current_grip = GRIPPER_OPEN
    hold_arm_joint_pos = robot.data.joint_pos[:, arm_entity_cfg.joint_ids].clone()
    slip_failure = SlipFailure(
        enabled=FAILURE_ENABLED,
        arm_after_subgoal=FAILURE_ARM_AFTER_SUBGOAL,
        steps_after_subgoal=FAILURE_STEPS_AFTER_SUBGOAL,
        open_width=FAILURE_OPEN_GRIPPER,
        keep_open=FAILURE_KEEP_GRIPPER_OPEN,
    )

    for params in SUBGOAL_PARAMS:
        label = params["label"]
        target_grip = _gripper_width(params["gripper"])
        duration = params["duration_steps"]
        start_pos = current_pos.clone()
        start_quat = current_quat.clone()
        target_pos_tensor = _resolve_subgoal_target(params, scene, current_pos, robot.device)
        target_quat_tensor = _repeat(EE_QUAT_DOWN, num_envs, robot.device)
        active_marker_w = _base_to_world_pos(target_pos_tensor, scene.env_origins)
        subgoal_markers.visualize(translations=active_marker_w)
        print(
            f"[INFO]: Starting '{label}' -> hand base xyz="
            f"{tuple(round(v, 3) for v in target_pos_tensor[0].tolist())}"
        )

        for phase_step in range(duration):
            if not simulation_app.is_running():
                return

            t = _smoothstep(min(phase_step / max(duration - 1, 1), 1.0))
            command_pos = start_pos + (target_pos_tensor - start_pos) * t
            command_quat = _quat_slerp_batch(start_quat, target_quat_tensor, t)
            command = torch.cat((command_pos, command_quat), dim=-1)
            diff_ik_controller.set_command(command)

            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_entity_cfg.joint_ids]
            ee_pos_b, ee_quat_b = _get_ee_pose_b(robot, arm_entity_cfg.body_ids[0])
            joint_pos = robot.data.joint_pos[:, arm_entity_cfg.joint_ids]
            joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
            hold_arm_joint_pos = joint_pos_des.clone()

            robot.set_joint_position_target(joint_pos_des, joint_ids=arm_entity_cfg.joint_ids)
            nominal_grip = current_grip + (target_grip - current_grip) * t
            commanded_grip = slip_failure.update_target_grip(nominal_grip)
            _set_gripper(
                robot,
                gripper_entity_cfg.joint_ids,
                commanded_grip,
                num_envs,
            )
            subgoal_markers.visualize(translations=active_marker_w)
            _step(sim, scene)

        print(f"[INFO]: '{label}' complete.")
        current_pos = target_pos_tensor
        current_quat = target_quat_tensor
        current_grip = commanded_grip

        if label == "Lift straight up":
            min_ball_z = TABLE_HEIGHT + BALL_RADIUS + MIN_BALL_LIFT_FOR_HOOP
            ball_z = ball.data.root_pos_w[0, 2].item()
            if ball_z < min_ball_z:
                print(
                    "[WARN]: Ball was not physically grasped "
                    f"(z={ball_z:.3f}, required>{min_ball_z:.3f}). Stopping before hoop motion."
                )
                break

        slip_failure.on_subgoal_complete(label)

    while simulation_app.is_running():
        robot.set_joint_position_target(
            hold_arm_joint_pos,
            joint_ids=arm_entity_cfg.joint_ids,
        )
        _set_gripper(robot, gripper_entity_cfg.joint_ids, current_grip, num_envs)
        _step(sim, scene)


def main():
    sim_cfg = sim_utils.SimulationCfg(
        dt=1.0 / 240.0,
        gravity=GRAVITY_XYZ,
        physics_material=HIGH_FRICTION_MATERIAL,
        physx=sim_utils.PhysxCfg(
            enable_external_forces_every_iteration=True,
            min_velocity_iteration_count=2,
        ),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(2.0, 1.8, 2.1), target=(0.50, -0.05, 1.15))

    scene_cfg = FrankaBasketballSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: Basketball no-failure scene ready. Starting task.")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
