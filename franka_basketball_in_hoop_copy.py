"""
franka_basketball_in_hoop.py

IsaacLab recreation of the AHA/RLBench `basketball_in_hoop` task with a Franka
Emika Panda arm. Pick up the ball and drop it through the hoop.

Run with:
    cd ~/IsaacLab
    ./isaaclab.sh -p scripts/franka_basketball_in_hoop.py

======================================================================
DESIGN: parametrized on the reset, not hardcoded to one scene
======================================================================
RLBench randomizes this task per reset (see the scene report's
`placement_ranges`):
  * the TASK ROOT (basket_boundary_root) is dropped at a random x/y and a
    random YAW (rz in [-0.785, 0.785] rad = +/-45 deg),
  * the BALL is sampled independently within its own x/y range.
Everything else (hoop, backboard, ball_stop, success sensor, and all four
waypoints) is RIGIDLY attached to the task root or the ball with fixed local
offsets, so it rotates/translates with them.

Therefore this script does NOT hardcode world positions. It takes two inputs:
    TASK_ROOT_POSE  (position + yaw)
    BALL_POSE       (position + rpy)
and reconstructs every object and waypoint from the report's reset-invariant
local offsets via  world = parent_pos + R(parent_quat) * local_offset.

The two poses below are filled from the provided `live_scene_after_reset`
report, but you can drop in any other reset's two poses (or sample them within
the documented ranges) and the whole scene + motion follows correctly,
including the +/-45 deg yaw. The local offsets were verified to reproduce the
report's world poses to sub-mm.
"""

import argparse
import math

from isaaclab.app import AppLauncher

# ----------------------------------------------------------------------
# 1. Launch Isaac Sim before importing the rest of Isaac Lab.
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Franka basketball-in-hoop task.")
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
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_apply,
    quat_slerp,
    subtract_frame_transforms,
)
from pxr import UsdPhysics

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG


# ======================================================================
# 3. THE RESET: two random poses define everything else.
# ======================================================================
TASK_ROOT_POS_XYZ = [0.167766, -0.142059, 0.752000]
TASK_ROOT_YAW = -0.312472

BALL_POS_XYZ = [0.036111, -0.063271, 0.781336]
BALL_RPY = [0.029645, 0.015460, -0.312606]

# Documented sampling ranges (for reference / optional randomization).
TASK_ROOT_X_RANGE = (0.075, 0.425)
TASK_ROOT_Y_RANGE = (-0.18, 0.18)
TASK_ROOT_Z_FIXED = 0.752
TASK_ROOT_YAW_RANGE = (-0.785398, 0.785398)
BALL_X_RANGE = (-0.078429, 0.343683)
BALL_Y_RANGE = (-0.261317, 0.310108)
BALL_Z_FIXED = 0.781336


# ======================================================================
# 4. RESET-INVARIANT local offsets (from the report).
# ======================================================================
LOCAL_OFFSETS = {
    "hoop_respondable": {"parent": "root",
        "pos": [0.124750, 0.031119, 0.253726], "rpy": [-3.141533, -0.019338, -3.141577]},
    "ball_stop": {"parent": "root",
        "pos": [-0.149500, 0.034500, 0.004803], "rpy": [0.0, -1.570796, 0.0]},
    "success": {"parent": "root",
        "pos": [0.094618, 0.024090, 0.034339], "rpy": [0.0, 0.0, 0.0]},
    "waypoint3": {"parent": "root",
        "pos": [0.093217, 0.026487, 0.335225], "rpy": [-3.141593, 0.0, 3.141593]},
    "waypoint1": {"parent": "ball",
        "pos": [0.000217, -0.005512, 0.012424], "rpy": [3.141592, 0.0, 3.141593]},
    "waypoint0": {"parent": "waypoint1",
        "pos": [2e-06, -0.0, -0.077], "rpy": [0.0, 0.0, 0.0]},
    "waypoint2": {"parent": "waypoint1",
        "pos": [-0.0, -0.0, -0.355], "rpy": [0.0, 0.0, 0.0]},
}

BALL_RADIUS = 0.030000
HOOP_RESPONDABLE_SIZE_XYZ = (0.190450, 0.423518, 0.467436)
HOOP_VISUAL_SIZE_XYZ = (0.394376, 0.426962, 0.584686)
TABLE_TOP_Z = TASK_ROOT_Z_FIXED

# ======================================================================
# 5. Tunables.
# ======================================================================
BALL_MASS = 0.035

ROBOT_BASE_XY = (-0.308951, 0.000000)
ROBOT_BASE_POS_XYZ = (ROBOT_BASE_XY[0], ROBOT_BASE_XY[1], TABLE_TOP_Z)

TABLE_SIZE = (1.600001, 1.100001, 0.750001)
TABLE_CENTER_XY = (0.300000, 0.000000)
TABLE_CENTER_Z = TABLE_TOP_Z - TABLE_SIZE[2] / 2.0

GRAVITY_XYZ = (0.0, 0.0, -9.81)

HOOP_OPENING = HOOP_RESPONDABLE_SIZE_XYZ[0]
HOOP_RIM_RADIUS = HOOP_OPENING / 2.0
HOOP_RIM_THICKNESS = 0.012
HOOP_RIM_SEGMENTS = 12
HOOP_RIM_SEGMENT_LENGTH = 2.0 * math.pi * HOOP_RIM_RADIUS / HOOP_RIM_SEGMENTS * 0.94
HOOP_BACKBOARD_SIZE = (0.014, HOOP_VISUAL_SIZE_XYZ[1], HOOP_VISUAL_SIZE_XYZ[2])
HOOP_BACKBOARD_OFFSET_X = HOOP_RIM_RADIUS + 0.035
HOOP_BACKBOARD_OFFSET_Z = 0.07
HOOP_NET_LENGTH = 0.09
HOOP_NET_CORD_RADIUS = 0.0018

HOME_JOINTS = [0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.785]
GRIPPER_OPEN = 0.04
GRIPPER_CLOSED = 0.0270
HAND_TO_BALL_Z = BALL_RADIUS + 0.063

PRE_GRASP_UP_Z = 0.077
APPROACH_ABOVE_HOOP_Z = 0.16
RETREAT_BACK_X = -0.16
RETREAT_UP_Z = 0.35
MIN_BALL_LIFT_FOR_HOOP = 0.06
ROBOT_TIME_SCALE = 0.65

EE_QUAT_DOWN = [0.0, 1.0, 0.0, 0.0]

# --- friction (ball<->table raised per request) ----------------------------
TABLE_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    static_friction=2.6, dynamic_friction=2.2, restitution=0.0,
    friction_combine_mode="max", restitution_combine_mode="multiply")
BALL_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    static_friction=4.0, dynamic_friction=3.4, restitution=0.0,
    friction_combine_mode="max", restitution_combine_mode="multiply")
HOOP_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    static_friction=1.2, dynamic_friction=1.0, restitution=0.05,
    friction_combine_mode="average", restitution_combine_mode="multiply")
GRIPPER_FRICTION_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    static_friction=4.0, dynamic_friction=3.2, restitution=0.0,
    friction_combine_mode="max", restitution_combine_mode="multiply")
CONTACT_PROPS = sim_utils.CollisionPropertiesCfg(contact_offset=0.004, rest_offset=0.0)


def _duration_steps(n):
    return max(45, round(n * ROBOT_TIME_SCALE))


# ======================================================================
# 6. Host-side pose math (plain python; wxyz quaternions).
# ======================================================================
def _quat_from_rpy(rpy):
    r, p, y = rpy
    qx = (math.cos(r / 2), math.sin(r / 2), 0.0, 0.0)
    qy = (math.cos(p / 2), 0.0, math.sin(p / 2), 0.0)
    qz = (math.cos(y / 2), 0.0, 0.0, math.sin(y / 2))
    return _qmul(_qmul(qz, qy), qx)


def _qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def _qapply(q, v):
    w, x, y, z = q
    vx, vy, vz = v
    tx = 2 * (y * vz - z * vy)
    ty = 2 * (z * vx - x * vz)
    tz = 2 * (x * vy - y * vx)
    return [vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx)]


def _compose(ppos, pquat, lpos, lquat):
    rp = _qapply(pquat, lpos)
    return [ppos[i] + rp[i] for i in range(3)], _qmul(pquat, lquat)


def _build_world_poses():
    poses = {}
    poses["root"] = (list(TASK_ROOT_POS_XYZ), _quat_from_rpy([0.0, 0.0, TASK_ROOT_YAW]))
    poses["ball"] = (list(BALL_POS_XYZ), _quat_from_rpy(BALL_RPY))
    for name in ["hoop_respondable", "ball_stop", "success", "waypoint3",
                 "waypoint1", "waypoint0", "waypoint2"]:
        spec = LOCAL_OFFSETS[name]
        ppos, pquat = poses[spec["parent"]]
        poses[name] = _compose(ppos, pquat, spec["pos"], _quat_from_rpy(spec["rpy"]))
    return poses


WORLD = _build_world_poses()
HOOP_CENTER_XYZ = WORLD["hoop_respondable"][0]
HOOP_QUAT = WORLD["hoop_respondable"][1]
HOOP_SUPPORT_HEIGHT = HOOP_CENTER_XYZ[2] - TABLE_TOP_Z

# For the VISUAL hoop assembly we use a clean yaw-only frame (a real hoop has a
# vertical backboard and a horizontal rim). The reported hoop orientation
# carries a ~180deg roll/pitch flip that, if used directly, scatters the rim and
# backboard. The task yaw is what visually matters, so build the decoration in a
# yaw-only frame centered at the hoop position.
HOOP_VIS_YAW = TASK_ROOT_YAW
HOOP_VIS_QUAT = _quat_from_rpy([0.0, 0.0, HOOP_VIS_YAW])


# ======================================================================
# 7. Subgoal sequence.
# ======================================================================
SUBGOAL_PARAMS = [
    {"label": "waypoint0 pre-grasp", "reference": "ball",
     "local_offset_xyz": LOCAL_OFFSETS["waypoint1"]["pos"],
     "world_up_z": HAND_TO_BALL_Z + PRE_GRASP_UP_Z, "gripper": "open",
     "duration_steps": _duration_steps(220)},
    {"label": "waypoint1 grasp", "reference": "ball",
     "local_offset_xyz": LOCAL_OFFSETS["waypoint1"]["pos"],
     "world_up_z": HAND_TO_BALL_Z, "gripper": "open",
     "duration_steps": _duration_steps(160)},
    {"label": "Close gripper", "reference": "ball",
     "local_offset_xyz": LOCAL_OFFSETS["waypoint1"]["pos"],
     "world_up_z": HAND_TO_BALL_Z, "gripper": "closed",
     "duration_steps": _duration_steps(140)},
    {"label": "Lift straight up", "reference": "previous",
     "world_delta_xyz": [0.0, 0.0, 0.355], "gripper": "closed",
     "duration_steps": _duration_steps(260)},
    {"label": "Approach above hoop", "reference": "root",
     "local_offset_xyz": LOCAL_OFFSETS["waypoint3"]["pos"],
     "world_up_z": APPROACH_ABOVE_HOOP_Z, "gripper": "closed",
     "duration_steps": _duration_steps(320)},
    {"label": "Over hoop (waypoint3)", "reference": "hoop_from_root",
     "gripper": "closed", "duration_steps": _duration_steps(180)},
    {"label": "Release above hoop", "reference": "hoop_from_root",
     "gripper": "open", "duration_steps": _duration_steps(80)},
    {"label": "Lift gripper clear", "reference": "root",
     "local_offset_xyz": LOCAL_OFFSETS["waypoint3"]["pos"],
     "world_up_z": APPROACH_ABOVE_HOOP_Z, "gripper": "open",
     "duration_steps": _duration_steps(150)},
    {"label": "Retreat", "reference": "root",
     "local_offset_xyz": [LOCAL_OFFSETS["waypoint3"]["pos"][0] + RETREAT_BACK_X,
                          LOCAL_OFFSETS["waypoint3"]["pos"][1],
                          LOCAL_OFFSETS["waypoint3"]["pos"][2]],
     "world_up_z": RETREAT_UP_Z, "gripper": "open",
     "duration_steps": _duration_steps(200)},
]


def _gripper_width(state):
    return GRIPPER_OPEN if state == "open" else GRIPPER_CLOSED


# ----------------------------------------------------------------------
# Hoop visual assembly, placed in the rotated hoop frame.
# ----------------------------------------------------------------------
def _hoop_local_to_world(local_xyz):
    return _compose(HOOP_CENTER_XYZ, HOOP_VIS_QUAT, list(local_xyz), (1.0, 0.0, 0.0, 0.0))[0]


def _rim_segment_pose(i):
    # Horizontal ring in the hoop's XY-plane; each short cylinder is tangent.
    a = 2.0 * math.pi * i / HOOP_RIM_SEGMENTS
    pos = _hoop_local_to_world([HOOP_RIM_RADIUS * math.cos(a), HOOP_RIM_RADIUS * math.sin(a), 0.0])
    # cylinder axis is local X; rotate it tangent (yaw a + 90deg) then by hoop yaw
    quat = _qmul(HOOP_VIS_QUAT, _quat_from_rpy([0.0, 0.0, a + math.pi / 2.0]))
    return tuple(pos), tuple(quat)


def _net_cord_pose(i):
    a = 2.0 * math.pi * i / HOOP_RIM_SEGMENTS
    return tuple(_hoop_local_to_world(
        [HOOP_RIM_RADIUS * math.cos(a), HOOP_RIM_RADIUS * math.sin(a), -HOOP_NET_LENGTH / 2.0]))


def _make_rim_segment_cfg(i):
    pos, rot = _rim_segment_pose(i)
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/HoopRim{i:02d}",
        spawn=sim_utils.CylinderCfg(
            radius=HOOP_RIM_THICKNESS / 2.0, height=HOOP_RIM_SEGMENT_LENGTH, axis="X",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=False),
            collision_props=CONTACT_PROPS, physics_material=HOOP_MATERIAL,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.28, 0.02), roughness=0.75)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot))


def _make_net_cord_cfg(i):
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/HoopNetCord{i:02d}",
        spawn=sim_utils.CapsuleCfg(
            radius=HOOP_NET_CORD_RADIUS, height=HOOP_NET_LENGTH, axis="Z",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.96, 0.96, 0.90), roughness=0.95)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=_net_cord_pose(i)))


def _backboard_pose():
    return tuple(_hoop_local_to_world([HOOP_BACKBOARD_OFFSET_X, 0.0, HOOP_BACKBOARD_OFFSET_Z])), tuple(HOOP_VIS_QUAT)


def _hoop_post_pose():
    return tuple(_hoop_local_to_world([HOOP_BACKBOARD_OFFSET_X, 0.0, -HOOP_SUPPORT_HEIGHT / 2.0])), tuple(HOOP_VIS_QUAT)


# ======================================================================
# 8. Scene configuration.
# ======================================================================
@configclass
class FrankaBasketballSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9)))

    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=False),
            collision_props=CONTACT_PROPS, physics_material=TABLE_MATERIAL,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.30, 0.20))),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(TABLE_CENTER_XY[0], TABLE_CENTER_XY[1], TABLE_CENTER_Z)))

    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=BALL_RADIUS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False, disable_gravity=False,
                solver_position_iteration_count=16, solver_velocity_iteration_count=2,
                max_depenetration_velocity=2.0),
            collision_props=CONTACT_PROPS, physics_material=BALL_MATERIAL,
            mass_props=sim_utils.MassPropertiesCfg(mass=BALL_MASS),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.33, 0.03), roughness=0.9)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(WORLD["ball"][0]), rot=tuple(WORLD["ball"][1])))

    backboard = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/HoopBackboard",
        spawn=sim_utils.CuboidCfg(
            size=HOOP_BACKBOARD_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=False),
            collision_props=CONTACT_PROPS, physics_material=HOOP_MATERIAL,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0), roughness=0.35)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=_backboard_pose()[0], rot=_backboard_pose()[1]))

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
            radius=0.012, height=max(HOOP_SUPPORT_HEIGHT, 0.05), axis="Z",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=False),
            collision_props=CONTACT_PROPS, physics_material=HOOP_MATERIAL,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2))),
        init_state=RigidObjectCfg.InitialStateCfg(pos=_hoop_post_pose()[0]))

    robot: Articulation = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.spawn.articulation_props.solver_velocity_iteration_count = 2
    robot.init_state.pos = ROBOT_BASE_POS_XYZ
    robot.init_state.joint_pos = {
        "panda_joint1": HOME_JOINTS[0], "panda_joint2": HOME_JOINTS[1],
        "panda_joint3": HOME_JOINTS[2], "panda_joint4": HOME_JOINTS[3],
        "panda_joint5": HOME_JOINTS[4], "panda_joint6": HOME_JOINTS[5],
        "panda_joint7": HOME_JOINTS[6], "panda_finger_joint.*": GRIPPER_OPEN}


# ======================================================================
# 9. Runtime helpers.
# ======================================================================
def _repeat(values, num_envs, device):
    return torch.tensor(values, dtype=torch.float32, device=device).unsqueeze(0).repeat(num_envs, 1)


def _smoothstep(t):
    return t * t * (3.0 - 2.0 * t)


def _quat_slerp_batch(start_quat, target_quat, t):
    return torch.stack([quat_slerp(q0, q1, t) for q0, q1 in zip(start_quat, target_quat)])


def _make_subgoal_markers():
    cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/BasketballSubgoals",
        markers={"active": sim_utils.SphereCfg(
            radius=0.012, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.35, 1.0)))})
    return VisualizationMarkers(cfg)


def _world_pos_to_base(target_pos_w, robot):
    rp = robot.data.root_pose_w
    pos_b, _ = subtract_frame_transforms(rp[:, 0:3], rp[:, 3:7], target_pos_w)
    return pos_b


def _base_pos_to_world(target_pos_b, robot):
    rp = robot.data.root_pose_w
    pos_w, _ = combine_frame_transforms(rp[:, 0:3], rp[:, 3:7], target_pos_b)
    return pos_w


def _ball_pose_w(scene):
    return scene["ball"].data.root_pos_w, scene["ball"].data.root_quat_w


def _hoop_pose_w(scene):
    device = scene["robot"].device
    n = scene.num_envs
    pos = _repeat(HOOP_CENTER_XYZ, n, device) + scene.env_origins
    quat = _repeat(HOOP_QUAT, n, device)
    return pos, quat


def _root_pose_w(scene):
    """Task-root world pose (per env). The waypoint3 offset is expressed in the
    ROOT frame, so drop-side targets are resolved against this, not the hoop."""
    device = scene["robot"].device
    n = scene.num_envs
    pos = _repeat(TASK_ROOT_POS_XYZ, n, device) + scene.env_origins
    quat = _repeat(WORLD["root"][1], n, device)
    return pos, quat


def _resolve_subgoal_target_w(params, scene, previous_target_w, device):
    ref = params["reference"]
    n = scene.num_envs
    if ref == "previous":
        return previous_target_w + _repeat(params["world_delta_xyz"], n, device)
    if ref == "hoop_from_root":
        # waypoint3 offset is expressed in the TASK-ROOT frame.
        pos, quat = _root_pose_w(scene)
        local = _repeat(LOCAL_OFFSETS["waypoint3"]["pos"], n, device)
        return pos + quat_apply(quat, local)
    if ref == "ball":
        pos, quat = _ball_pose_w(scene)
    elif ref == "root":
        # drop-side offsets (derived from waypoint3) are root-local
        pos, quat = _root_pose_w(scene)
    else:
        raise ValueError(f"Unsupported reference: {ref}")
    local = _repeat(params["local_offset_xyz"], n, device)
    target = pos + quat_apply(quat, local)
    up = params.get("world_up_z", 0.0)
    if up:
        target = target + _repeat([0.0, 0.0, up], n, device)
    return target


def _print_subgoals():
    print("[INFO]: Subgoals (resolved at runtime against live ball/hoop frames):")
    for i, p in enumerate(SUBGOAL_PARAMS):
        print(f"  {i}: {p['label']}  ref={p['reference']}  grip={p['gripper']}  steps={p['duration_steps']}")


def _set_gripper(robot, gripper_joint_ids, width, num_envs):
    target = torch.full((num_envs, len(gripper_joint_ids)), width, dtype=torch.float32, device=robot.device)
    robot.set_joint_position_target(target, joint_ids=gripper_joint_ids)


def _get_ee_pose_b(robot, ee_body_id):
    ee = robot.data.body_pose_w[:, ee_body_id]
    rp = robot.data.root_pose_w
    return subtract_frame_transforms(rp[:, 0:3], rp[:, 3:7], ee[:, 0:3], ee[:, 3:7])


def _write_home_state(robot):
    jp = robot.data.default_joint_pos.clone()
    robot.write_joint_state_to_sim(jp, torch.zeros_like(jp))
    robot.set_joint_position_target(jp)
    robot.write_data_to_sim()


def _step(sim, scene):
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())


def _collision_prims_under(stage, root_path):
    found = []
    rp = stage.GetPrimAtPath(root_path)
    if not rp or not rp.IsValid():
        return found
    from pxr import Usd
    for prim in Usd.PrimRange(rp):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            found.append(str(prim.GetPath()))
    return found


def _apply_gripper_friction(robot):
    mp = "/World/PhysicsMaterials/GripperHighFriction"
    GRIPPER_FRICTION_MATERIAL.func(mp, GRIPPER_FRICTION_MATERIAL)
    stage = sim_utils.get_current_stage()
    roots = robot.root_physx_view.prim_paths
    bound = set()
    # Search the ENTIRE robot subtree for collision prims whose path mentions a
    # finger. The collider is often nested under .../panda_*finger/collisions or
    # /geometry rather than directly under the body prim, so a subtree scan with
    # a name filter is more robust than looking only one level under the body.
    for root_path in roots:
        for cp in _collision_prims_under(stage, root_path):
            if "finger" in cp.lower() and cp not in bound:
                sim_utils.bind_physics_material(cp, mp)
                bound.add(cp)
    print(f"[INFO]: Applied high-friction gripper material to {len(bound)} fingertip collision prims.")
    if not bound:
        print("[WARN]: No fingertip collision prims found; default materials remain.")
        print("[DEBUG]: ALL collision prims under robot root(s):")
        for root_path in roots:
            for cp in _collision_prims_under(stage, root_path):
                print("        ", cp)


def run_simulator(sim, scene):
    robot = scene["robot"]
    ball = scene["ball"]
    num_envs = scene.num_envs

    arm_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    arm_cfg.resolve(scene)
    grip_cfg = SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"])
    grip_cfg.resolve(scene)

    assert isinstance(arm_cfg.body_ids, list)
    ee_body_id = arm_cfg.body_ids[0]
    ee_jacobi_idx = ee_body_id - 1 if robot.is_fixed_base else ee_body_id

    ik = DifferentialIKController(
        DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        num_envs=num_envs, device=robot.device)

    scene.reset()
    _write_home_state(robot)
    robot.reset()
    _apply_gripper_friction(robot)
    _set_gripper(robot, grip_cfg.joint_ids, GRIPPER_OPEN, num_envs)

    for _ in range(60):
        robot.set_joint_position_target(
            robot.data.default_joint_pos[:, arm_cfg.joint_ids], joint_ids=arm_cfg.joint_ids)
        _set_gripper(robot, grip_cfg.joint_ids, GRIPPER_OPEN, num_envs)
        _step(sim, scene)
    ik.reset()

    markers = _make_subgoal_markers()
    _print_subgoals()

    ee_pos_b0, _ = _get_ee_pose_b(robot, ee_body_id)
    jac0 = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_cfg.joint_ids]
    print("[DIAG]: |J| sum =", float(jac0.abs().sum().item()), " jac shape =", tuple(jac0.shape))
    print("[DIAG]: hand world @home =", tuple(round(v, 3) for v in _base_pos_to_world(ee_pos_b0, robot)[0].tolist()))
    print("[DIAG]: ball world =", tuple(round(v, 3) for v in ball.data.root_pos_w[0].tolist()))
    print("[DIAG]: hoop world =", tuple(round(v, 3) for v in HOOP_CENTER_XYZ))

    current_pos, current_quat = _get_ee_pose_b(robot, ee_body_id)
    current_grip = GRIPPER_OPEN
    hold = robot.data.joint_pos[:, arm_cfg.joint_ids].clone()
    previous_target_w = _base_pos_to_world(current_pos, robot)

    for params in SUBGOAL_PARAMS:
        label = params["label"]
        target_grip = _gripper_width(params["gripper"])
        duration = params["duration_steps"]
        start_pos = current_pos.clone()
        start_quat = current_quat.clone()

        target_pos_w = _resolve_subgoal_target_w(params, scene, previous_target_w, robot.device)
        target_pos_b = _world_pos_to_base(target_pos_w, robot)
        target_quat = _repeat(EE_QUAT_DOWN, num_envs, robot.device)

        # Set the IK command ONCE to the final target for this subgoal, then let
        # the DLS controller drive the arm to it over `duration` steps. (Feeding
        # a re-interpolated command every step makes the per-step error tiny and
        # the arm barely moves -- that was the previous bug.)
        ik.set_command(torch.cat((target_pos_b, target_quat), dim=-1))

        markers.visualize(translations=target_pos_w)
        print(f"[INFO]: '{label}' -> world {tuple(round(v,3) for v in target_pos_w[0].tolist())}")

        for step in range(duration):
            if not simulation_app.is_running():
                return
            t = min(step / max(duration - 1, 1), 1.0)

            jac = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_cfg.joint_ids]
            ee_pos_b, ee_quat_b = _get_ee_pose_b(robot, ee_body_id)
            jpos = robot.data.joint_pos[:, arm_cfg.joint_ids]
            jdes = ik.compute(ee_pos_b, ee_quat_b, jac, jpos)
            hold = jdes.clone()

            if params is SUBGOAL_PARAMS[0] and step < 3:
                dq = (jdes - jpos).abs().max().item()
                err = (target_pos_b - ee_pos_b).norm(dim=-1).max().item()
                print(f"[DIAG]: step {step}: max|dq|={dq:.5f} pos_err={err:.4f}")

            robot.set_joint_position_target(jdes, joint_ids=arm_cfg.joint_ids)
            _set_gripper(robot, grip_cfg.joint_ids, current_grip + (target_grip - current_grip) * t, num_envs)
            markers.visualize(translations=target_pos_w)
            _step(sim, scene)

        print(f"[INFO]: '{label}' complete.")
        current_pos, current_quat = _get_ee_pose_b(robot, ee_body_id)
        current_grip = target_grip
        previous_target_w = target_pos_w

        if label == "Lift straight up":
            min_z = TABLE_TOP_Z + BALL_RADIUS + MIN_BALL_LIFT_FOR_HOOP
            bz = ball.data.root_pos_w[:, 2]
            if bz.min().item() < min_z:
                failed = int((bz < min_z).sum().item())
                print(f"[WARN]: Ball not grasped in {failed}/{num_envs} env(s) "
                      f"(min z={bz.min().item():.3f}, need>{min_z:.3f}). Stopping.")
                break

    while simulation_app.is_running():
        robot.set_joint_position_target(hold, joint_ids=arm_cfg.joint_ids)
        _set_gripper(robot, grip_cfg.joint_ids, current_grip, num_envs)
        _step(sim, scene)


def main():
    sim_cfg = sim_utils.SimulationCfg(
        dt=1.0 / 240.0, gravity=GRAVITY_XYZ, physics_material=TABLE_MATERIAL,
        physx=sim_utils.PhysxCfg(enable_external_forces_every_iteration=True, min_velocity_iteration_count=2))
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(2.0, 1.8, 2.1), target=(0.30, -0.10, 1.05))
    scene = InteractiveScene(FrankaBasketballSceneCfg(num_envs=1, env_spacing=2.5))
    sim.reset()
    print("[INFO]: Basketball scene ready. Starting task.")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()