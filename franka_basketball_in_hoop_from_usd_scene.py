"""
franka_basketball_in_hoop_from_usd_scene.py

Builds a Franka basketball-in-hoop Isaac Lab task from:
    /home/ramtin/AHA/portable_scene_reports/basketball_in_hoop.scene_context.md
and visible USD assets from:
    /home/ramtin/Downloads/basketball_in_hoop_usd-20260529T125644Z-3-001/basketball_in_hoop_usd

The scene report is read at startup. USD meshes are used for the visible ball,
hoop, and ball stop. Simple Isaac primitives remain as invisible collision
proxies so the Franka grasp/drop task is physically stable.

Run with:
    cd ~/IsaacLab
    ./isaaclab.sh -p scripts/franka_basketball_in_hoop_from_usd_scene.py
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

ISAACLAB_ROOT = Path(__file__).resolve().parents[1]
for package_dir in ("isaaclab", "isaaclab_assets", "isaaclab_tasks"):
    source_path = ISAACLAB_ROOT / "source" / package_dir
    if source_path.is_dir() and str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

from isaaclab.app import AppLauncher


DEFAULT_SCENE_CONTEXT = Path("/home/ramtin/AHA/portable_scene_reports/basketball_in_hoop.scene_context.md")
DEFAULT_USD_DIR = Path(
    "/home/ramtin/Downloads/basketball_in_hoop_usd-20260529T125644Z-3-001/basketball_in_hoop_usd"
)


def _load_scene_context(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if match is None:
        raise RuntimeError(f"No fenced JSON scene data found in {path}")
    return json.loads(match.group(1))


def _scene_objects(scene_data: dict) -> dict[str, dict]:
    return {obj["name"]: obj for obj in scene_data["objects"]}


def _scene_waypoints(scene_data: dict) -> dict[str, dict]:
    return {waypoint["name"]: waypoint for waypoint in scene_data["waypoints"]}


def _pose_from_location(location: dict) -> tuple[list[float], tuple[float, float, float, float]]:
    pos = [float(v) for v in location["position_xyz_m"]]
    qx, qy, qz, qw = [float(v) for v in location["quaternion_xyzw"]]
    return pos, (qw, qx, qy, qz)


def _world_pose(entry: dict) -> tuple[list[float], tuple[float, float, float, float]]:
    return _pose_from_location(entry["world_location"])


def _world_rpy(entry: dict) -> list[float]:
    return [float(v) for v in entry["world_location"]["orientation_rpy_rad"]]


def _find_usd_paths(usd_dir: Path) -> dict[str, str]:
    names = {
        "basket_boundary_root": "basketball_in_hoop_basket_boundary_root.usd",
        "basket_ball_hoop_respondable": "basketball_in_hoop_basket_ball_hoop_respondable.usd",
        "basket_ball_hoop_visual": "basketball_in_hoop_basket_ball_hoop_visual.usd",
        "ball_stop": "basketball_in_hoop_ball_stop.usd",
        "ball": "basketball_in_hoop_ball.usd",
    }
    paths = {key: usd_dir / filename for key, filename in names.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing USD file(s):\n  " + "\n  ".join(missing))
    return {key: str(path) for key, path in paths.items()}


# ----------------------------------------------------------------------
# 1. Parse paths and launch Isaac Sim before importing the rest of Isaac Lab.
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Franka basketball-in-hoop task from AHA scene report + USD assets.")
parser.add_argument("--scene-context", type=Path, default=DEFAULT_SCENE_CONTEXT)
parser.add_argument("--usd-dir", type=Path, default=DEFAULT_USD_DIR)
parser.add_argument(
    "--show-proxy-geometry",
    action="store_true",
    help="Show the simple primitive collision proxies used under the USD visuals.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

SCENE_DATA = _load_scene_context(args_cli.scene_context)
OBJECTS = _scene_objects(SCENE_DATA)
WAYPOINTS = _scene_waypoints(SCENE_DATA)
USD_PATHS = _find_usd_paths(args_cli.usd_dir)
SHOW_PROXY_GEOMETRY = bool(args_cli.show_proxy_geometry)

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
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms
from pxr import Usd, UsdPhysics

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG


# ----------------------------------------------------------------------
# 3. Values read from the report and a few Panda/task tunables.
# ----------------------------------------------------------------------
LOCAL_FRANKA_USD = ISAACLAB_ROOT / "source/isaaclab_assets/data/Robots/FrankaEmika/panda_instanceable.usd"

ROOT_POS_XYZ, ROOT_QUAT_WXYZ = _world_pose(OBJECTS["basket_boundary_root"])
ROOT_RPY = _world_rpy(OBJECTS["basket_boundary_root"])
TASK_ROOT_YAW = ROOT_RPY[2]

BALL_POS_XYZ, BALL_QUAT_WXYZ = _world_pose(OBJECTS["ball"])
BALL_STOP_POS_XYZ, BALL_STOP_QUAT_WXYZ = _world_pose(OBJECTS["ball_stop"])
HOOP_RESPONDABLE_POS_XYZ, HOOP_RESPONDABLE_QUAT_WXYZ = _world_pose(OBJECTS["basket_ball_hoop_respondable"])
HOOP_VISUAL_POS_XYZ, HOOP_VISUAL_QUAT_WXYZ = _world_pose(OBJECTS["basket_ball_hoop_visual"])

WAYPOINT3_POS_XYZ, _ = _world_pose(WAYPOINTS["waypoint3"])

TABLE_TOP_Z = ROOT_POS_XYZ[2]
TABLE_SIZE = (1.600001, 1.100001, 0.750001)
TABLE_CENTER_XY = (0.300000, 0.000000)
TABLE_CENTER_Z = TABLE_TOP_Z - TABLE_SIZE[2] / 2.0
GRAVITY_XYZ = (0.0, 0.0, -9.81)

ROBOT_BASE_POS_XYZ = (-0.308951, 0.0, TABLE_TOP_Z)
HOME_JOINTS = [0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.785]
ROBOT_DISABLE_GRAVITY_FOR_STABLE_IK = True
ROBOT_TIME_SCALE = 0.65

BALL_RADIUS = 0.030
BALL_MASS = 0.035
GRIPPER_OPEN = 0.040
GRIPPER_CLOSED = 0.027
HAND_TO_BALL_Z = BALL_RADIUS + 0.063
MIN_BALL_LIFT_FOR_HOOP = 0.06

WAYPOINT1_FROM_BALL_XY = [
    WAYPOINTS["waypoint1"]["relative_to_nearest_object"]["location_in_reference_frame"]["position_xyz_m"][0],
    WAYPOINTS["waypoint1"]["relative_to_nearest_object"]["location_in_reference_frame"]["position_xyz_m"][1],
]
WAYPOINT0_UP_FROM_WAYPOINT1_Z = abs(WAYPOINTS["waypoint0"]["fixed_offset_xyz_m"][2])
WAYPOINT2_UP_FROM_WAYPOINT1_Z = abs(WAYPOINTS["waypoint2"]["fixed_offset_xyz_m"][2])
WAYPOINT3_DELTA_FROM_HOOP_XYZ = [
    WAYPOINT3_POS_XYZ[i] - HOOP_RESPONDABLE_POS_XYZ[i] for i in range(3)
]
APPROACH_ABOVE_HOOP_Z = 0.16
RETREAT_BACK_X = -0.16
RETREAT_UP_Z = 0.35

# Hand points down toward the table/ball.
EE_QUAT_DOWN = [0.0, 1.0, 0.0, 0.0]

# Collision proxies around the visible USD hoop.
HOOP_OPENING = 0.190450
HOOP_RIM_RADIUS = HOOP_OPENING / 2.0
HOOP_RIM_THICKNESS = 0.012
HOOP_RIM_SEGMENTS = 12
HOOP_RIM_SEGMENT_LENGTH = 2.0 * math.pi * HOOP_RIM_RADIUS / HOOP_RIM_SEGMENTS * 0.94
HOOP_BACKBOARD_SIZE = (0.014, 0.426962, 0.584686)
HOOP_BACKBOARD_OFFSET_X = HOOP_RIM_RADIUS + 0.035
HOOP_BACKBOARD_OFFSET_Z = 0.07
HOOP_SUPPORT_HEIGHT = max(HOOP_RESPONDABLE_POS_XYZ[2] - TABLE_TOP_Z, 0.05)

TABLE_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    static_friction=2.6,
    dynamic_friction=2.2,
    restitution=0.0,
    friction_combine_mode="max",
    restitution_combine_mode="multiply",
)
BALL_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    static_friction=4.0,
    dynamic_friction=3.4,
    restitution=0.0,
    friction_combine_mode="max",
    restitution_combine_mode="multiply",
)
HOOP_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    static_friction=1.2,
    dynamic_friction=1.0,
    restitution=0.05,
    friction_combine_mode="average",
    restitution_combine_mode="multiply",
)
GRIPPER_FRICTION_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    static_friction=4.0,
    dynamic_friction=3.2,
    restitution=0.0,
    friction_combine_mode="max",
    restitution_combine_mode="multiply",
)
CONTACT_PROPS = sim_utils.CollisionPropertiesCfg(contact_offset=0.004, rest_offset=0.0)


def _duration_steps(n: int) -> int:
    return max(45, round(n * ROBOT_TIME_SCALE))


def _quat_from_rpy(rpy: list[float]) -> tuple[float, float, float, float]:
    r, p, y = rpy
    qx = (math.cos(r / 2), math.sin(r / 2), 0.0, 0.0)
    qy = (math.cos(p / 2), 0.0, math.sin(p / 2), 0.0)
    qz = (math.cos(y / 2), 0.0, 0.0, math.sin(y / 2))
    return _qmul(_qmul(qz, qy), qx)


def _qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _qapply(q, v):
    w, x, y, z = q
    vx, vy, vz = v
    tx = 2 * (y * vz - z * vy)
    ty = 2 * (z * vx - x * vz)
    tz = 2 * (x * vy - y * vx)
    return [
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ]


def _compose(ppos, pquat, lpos, lquat):
    rp = _qapply(pquat, lpos)
    return [ppos[i] + rp[i] for i in range(3)], _qmul(pquat, lquat)


HOOP_VIS_QUAT = _quat_from_rpy([0.0, 0.0, TASK_ROOT_YAW])


def _hoop_local_to_world(local_xyz):
    return _compose(HOOP_RESPONDABLE_POS_XYZ, HOOP_VIS_QUAT, list(local_xyz), (1.0, 0.0, 0.0, 0.0))[0]


def _rim_segment_pose(index: int):
    angle = 2.0 * math.pi * index / HOOP_RIM_SEGMENTS
    pos = _hoop_local_to_world([HOOP_RIM_RADIUS * math.cos(angle), HOOP_RIM_RADIUS * math.sin(angle), 0.0])
    quat = _qmul(HOOP_VIS_QUAT, _quat_from_rpy([0.0, 0.0, angle + math.pi / 2.0]))
    return tuple(pos), tuple(quat)


def _make_rim_segment_cfg(index: int) -> RigidObjectCfg:
    pos, rot = _rim_segment_pose(index)
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/HoopRimProxy{index:02d}",
        spawn=sim_utils.CylinderCfg(
            radius=HOOP_RIM_THICKNESS / 2.0,
            height=HOOP_RIM_SEGMENT_LENGTH,
            axis="X",
            visible=SHOW_PROXY_GEOMETRY,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=False),
            collision_props=CONTACT_PROPS,
            physics_material=HOOP_MATERIAL,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.28, 0.02), roughness=0.75),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
    )


def _backboard_pose():
    return tuple(_hoop_local_to_world([HOOP_BACKBOARD_OFFSET_X, 0.0, HOOP_BACKBOARD_OFFSET_Z])), tuple(HOOP_VIS_QUAT)


def _hoop_post_pose():
    return tuple(_hoop_local_to_world([HOOP_BACKBOARD_OFFSET_X, 0.0, -HOOP_SUPPORT_HEIGHT / 2.0])), tuple(HOOP_VIS_QUAT)


SUBGOAL_PARAMS = [
    {
        "label": "waypoint0 pre-grasp",
        "reference": "ball_world",
        "offset_xyz": [WAYPOINT1_FROM_BALL_XY[0], WAYPOINT1_FROM_BALL_XY[1], HAND_TO_BALL_Z + WAYPOINT0_UP_FROM_WAYPOINT1_Z],
        "gripper": "open",
        "duration_steps": _duration_steps(220),
    },
    {
        "label": "waypoint1 grasp",
        "reference": "ball_world",
        "offset_xyz": [WAYPOINT1_FROM_BALL_XY[0], WAYPOINT1_FROM_BALL_XY[1], HAND_TO_BALL_Z],
        "gripper": "open",
        "duration_steps": _duration_steps(160),
    },
    {
        "label": "Close gripper",
        "reference": "ball_world",
        "offset_xyz": [WAYPOINT1_FROM_BALL_XY[0], WAYPOINT1_FROM_BALL_XY[1], HAND_TO_BALL_Z],
        "gripper": "closed",
        "duration_steps": _duration_steps(140),
    },
    {
        "label": "Lift straight up",
        "reference": "previous",
        "world_delta_xyz": [0.0, 0.0, WAYPOINT2_UP_FROM_WAYPOINT1_Z],
        "gripper": "closed",
        "duration_steps": _duration_steps(260),
    },
    {
        "label": "Approach above hoop",
        "reference": "hoop",
        "offset_xyz": [
            WAYPOINT3_DELTA_FROM_HOOP_XYZ[0],
            WAYPOINT3_DELTA_FROM_HOOP_XYZ[1],
            WAYPOINT3_DELTA_FROM_HOOP_XYZ[2] + APPROACH_ABOVE_HOOP_Z,
        ],
        "gripper": "closed",
        "duration_steps": _duration_steps(320),
    },
    {
        "label": "waypoint3 over hoop",
        "reference": "hoop",
        "offset_xyz": WAYPOINT3_DELTA_FROM_HOOP_XYZ,
        "gripper": "closed",
        "duration_steps": _duration_steps(180),
    },
    {
        "label": "Release above hoop",
        "reference": "hoop",
        "offset_xyz": WAYPOINT3_DELTA_FROM_HOOP_XYZ,
        "gripper": "open",
        "duration_steps": _duration_steps(80),
    },
    {
        "label": "Lift gripper clear",
        "reference": "hoop",
        "offset_xyz": [
            WAYPOINT3_DELTA_FROM_HOOP_XYZ[0],
            WAYPOINT3_DELTA_FROM_HOOP_XYZ[1],
            WAYPOINT3_DELTA_FROM_HOOP_XYZ[2] + APPROACH_ABOVE_HOOP_Z,
        ],
        "gripper": "open",
        "duration_steps": _duration_steps(150),
    },
    {
        "label": "Retreat",
        "reference": "hoop",
        "offset_xyz": [
            WAYPOINT3_DELTA_FROM_HOOP_XYZ[0] + RETREAT_BACK_X,
            WAYPOINT3_DELTA_FROM_HOOP_XYZ[1],
            WAYPOINT3_DELTA_FROM_HOOP_XYZ[2] + RETREAT_UP_Z,
        ],
        "gripper": "open",
        "duration_steps": _duration_steps(200),
    },
]


def _gripper_width(state: str) -> float:
    return GRIPPER_OPEN if state == "open" else GRIPPER_CLOSED


# ----------------------------------------------------------------------
# 4. Scene configuration.
# ----------------------------------------------------------------------
@configclass
class FrankaBasketballUsdSceneCfg(InteractiveSceneCfg):
    ground = RigidObjectCfg(
        prim_path="/World/ground",
        spawn=sim_utils.CuboidCfg(
            size=(20.0, 20.0, 0.02),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=False),
            collision_props=CONTACT_PROPS,
            physics_material=TABLE_MATERIAL,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.08, 0.08, 0.08), roughness=0.9),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -0.01)),
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
            physics_material=TABLE_MATERIAL,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.30, 0.20)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(TABLE_CENTER_XY[0], TABLE_CENTER_XY[1], TABLE_CENTER_Z)),
    )

    basket_boundary_root_usd = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BasketBoundaryRootUSD",
        spawn=sim_utils.UsdFileCfg(usd_path=USD_PATHS["basket_boundary_root"], visible=False),
        init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(ROOT_POS_XYZ), rot=ROOT_QUAT_WXYZ),
    )

    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=BALL_RADIUS,
            visible=SHOW_PROXY_GEOMETRY,
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(BALL_POS_XYZ), rot=BALL_QUAT_WXYZ),
    )

    ball_usd_visual = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Ball/UsdVisual",
        spawn=sim_utils.UsdFileCfg(usd_path=USD_PATHS["ball"]),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    ball_stop_usd_visual = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BallStopUSD",
        spawn=sim_utils.UsdFileCfg(usd_path=USD_PATHS["ball_stop"]),
        init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(BALL_STOP_POS_XYZ), rot=BALL_STOP_QUAT_WXYZ),
    )

    ball_stop_proxy = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BallStopProxy",
        spawn=sim_utils.CuboidCfg(
            size=(0.018, 0.018, 0.055),
            visible=SHOW_PROXY_GEOMETRY,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=False),
            collision_props=CONTACT_PROPS,
            physics_material=HOOP_MATERIAL,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.1), roughness=0.8),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(BALL_STOP_POS_XYZ), rot=BALL_STOP_QUAT_WXYZ),
    )

    hoop_respondable_usd_visual = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/HoopRespondableUSD",
        spawn=sim_utils.UsdFileCfg(usd_path=USD_PATHS["basket_ball_hoop_respondable"]),
        init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(HOOP_RESPONDABLE_POS_XYZ), rot=HOOP_RESPONDABLE_QUAT_WXYZ),
    )

    hoop_visual_usd = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/HoopVisualUSD",
        spawn=sim_utils.UsdFileCfg(usd_path=USD_PATHS["basket_ball_hoop_visual"]),
        init_state=AssetBaseCfg.InitialStateCfg(pos=tuple(HOOP_VISUAL_POS_XYZ), rot=HOOP_VISUAL_QUAT_WXYZ),
    )

    backboard_proxy = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/HoopBackboardProxy",
        spawn=sim_utils.CuboidCfg(
            size=HOOP_BACKBOARD_SIZE,
            visible=SHOW_PROXY_GEOMETRY,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=False),
            collision_props=CONTACT_PROPS,
            physics_material=HOOP_MATERIAL,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0), roughness=0.35),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=_backboard_pose()[0], rot=_backboard_pose()[1]),
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

    hoop_post_proxy = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/HoopPostProxy",
        spawn=sim_utils.CylinderCfg(
            radius=0.012,
            height=HOOP_SUPPORT_HEIGHT,
            axis="Z",
            visible=SHOW_PROXY_GEOMETRY,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=False),
            collision_props=CONTACT_PROPS,
            physics_material=HOOP_MATERIAL,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=_hoop_post_pose()[0]),
    )

    robot: Articulation = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    if LOCAL_FRANKA_USD.is_file():
        robot.spawn.usd_path = str(LOCAL_FRANKA_USD)
    robot.spawn.rigid_props.disable_gravity = ROBOT_DISABLE_GRAVITY_FOR_STABLE_IK
    robot.spawn.articulation_props.solver_velocity_iteration_count = 2
    robot.init_state.pos = ROBOT_BASE_POS_XYZ
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


# ----------------------------------------------------------------------
# 5. Runtime helpers.
# ----------------------------------------------------------------------
def _repeat(values, num_envs: int, device: str):
    return torch.tensor(values, dtype=torch.float32, device=device).unsqueeze(0).repeat(num_envs, 1)


def _make_subgoal_markers():
    cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/BasketballSubgoals",
        markers={
            "active": sim_utils.SphereCfg(
                radius=0.012,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.35, 1.0)),
            )
        },
    )
    return VisualizationMarkers(cfg)


def _world_pos_to_base(target_pos_w, robot):
    root_pose = robot.data.root_pose_w
    pos_b, _ = subtract_frame_transforms(root_pose[:, 0:3], root_pose[:, 3:7], target_pos_w)
    return pos_b


def _base_pos_to_world(target_pos_b, robot):
    root_pose = robot.data.root_pose_w
    pos_w, _ = combine_frame_transforms(root_pose[:, 0:3], root_pose[:, 3:7], target_pos_b)
    return pos_w


def _resolve_subgoal_target_w(params, scene, previous_target_w, device):
    num_envs = scene.num_envs
    reference = params["reference"]
    if reference == "previous":
        return previous_target_w + _repeat(params["world_delta_xyz"], num_envs, device)
    if reference == "ball_world":
        return scene["ball"].data.root_pos_w + _repeat(params["offset_xyz"], num_envs, device)
    if reference == "hoop":
        hoop_pos_w = _repeat(HOOP_RESPONDABLE_POS_XYZ, num_envs, device) + scene.env_origins
        return hoop_pos_w + _repeat(params["offset_xyz"], num_envs, device)
    raise ValueError(f"Unsupported subgoal reference: {reference}")


def _print_loaded_scene():
    print(f"[INFO]: Loaded scene report: {args_cli.scene_context}")
    print(f"[INFO]: Loaded USD directory: {args_cli.usd_dir}")
    for name, path in USD_PATHS.items():
        print(f"  {name}: {path}")
    print("[INFO]: Scene anchors from report:")
    print(f"  ball={tuple(round(v, 4) for v in BALL_POS_XYZ)}")
    print(f"  hoop_respondable={tuple(round(v, 4) for v in HOOP_RESPONDABLE_POS_XYZ)}")
    print(f"  waypoint3={tuple(round(v, 4) for v in WAYPOINT3_POS_XYZ)}")


def _print_subgoals():
    print("[INFO]: Subgoals:")
    for index, params in enumerate(SUBGOAL_PARAMS):
        print(
            f"  {index}: {params['label']}  ref={params['reference']}  "
            f"grip={params['gripper']}  steps={params['duration_steps']}"
        )


def _set_gripper(robot, gripper_joint_ids, width: float, num_envs: int):
    target = torch.full((num_envs, len(gripper_joint_ids)), width, dtype=torch.float32, device=robot.device)
    robot.set_joint_position_target(target, joint_ids=gripper_joint_ids)


def _get_ee_pose_b(robot, ee_body_id):
    ee_pose = robot.data.body_pose_w[:, ee_body_id]
    root_pose = robot.data.root_pose_w
    return subtract_frame_transforms(root_pose[:, 0:3], root_pose[:, 3:7], ee_pose[:, 0:3], ee_pose[:, 3:7])


def _write_home_state(robot):
    joint_pos = robot.data.default_joint_pos.clone()
    robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))
    robot.set_joint_position_target(joint_pos)
    robot.write_data_to_sim()


def _step(sim, scene):
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())


def _collision_prims_under(stage, root_path):
    found = []
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim or not root_prim.IsValid():
        return found
    for prim in Usd.PrimRange(root_prim):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            found.append(str(prim.GetPath()))
    return found


def _apply_gripper_friction(robot):
    material_path = "/World/PhysicsMaterials/GripperHighFriction"
    GRIPPER_FRICTION_MATERIAL.func(material_path, GRIPPER_FRICTION_MATERIAL)
    stage = sim_utils.get_current_stage()
    bound = set()
    for root_path in robot.root_physx_view.prim_paths:
        for collision_path in _collision_prims_under(stage, root_path):
            if "finger" in collision_path.lower() and collision_path not in bound:
                sim_utils.bind_physics_material(collision_path, material_path)
                bound.add(collision_path)
    print(f"[INFO]: Applied high-friction gripper material to {len(bound)} fingertip collision prims.")
    if not bound:
        print("[WARN]: No fingertip collision prims found; default materials remain.")


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
        num_envs=num_envs,
        device=robot.device,
    )

    scene.reset()
    _write_home_state(robot)
    robot.reset()
    _apply_gripper_friction(robot)
    _set_gripper(robot, grip_cfg.joint_ids, GRIPPER_OPEN, num_envs)

    for _ in range(60):
        robot.set_joint_position_target(robot.data.default_joint_pos[:, arm_cfg.joint_ids], joint_ids=arm_cfg.joint_ids)
        _set_gripper(robot, grip_cfg.joint_ids, GRIPPER_OPEN, num_envs)
        _step(sim, scene)
    ik.reset()

    markers = _make_subgoal_markers()
    _print_loaded_scene()
    _print_subgoals()

    ee_pos_b0, _ = _get_ee_pose_b(robot, ee_body_id)
    jac0 = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_cfg.joint_ids]
    print("[DIAG]: |J| sum =", float(jac0.abs().sum().item()), " jac shape =", tuple(jac0.shape))
    print("[DIAG]: hand world @home =", tuple(round(v, 3) for v in _base_pos_to_world(ee_pos_b0, robot)[0].tolist()))
    print("[DIAG]: ball world =", tuple(round(v, 3) for v in ball.data.root_pos_w[0].tolist()))
    print("[DIAG]: hoop world =", tuple(round(v, 3) for v in HOOP_RESPONDABLE_POS_XYZ))

    current_pos, _ = _get_ee_pose_b(robot, ee_body_id)
    current_grip = GRIPPER_OPEN
    hold = robot.data.joint_pos[:, arm_cfg.joint_ids].clone()
    previous_target_w = _base_pos_to_world(current_pos, robot)

    for params in SUBGOAL_PARAMS:
        label = params["label"]
        target_grip = _gripper_width(params["gripper"])
        duration = params["duration_steps"]

        target_pos_w = _resolve_subgoal_target_w(params, scene, previous_target_w, robot.device)
        target_pos_b = _world_pos_to_base(target_pos_w, robot)
        target_quat = _repeat(EE_QUAT_DOWN, num_envs, robot.device)
        ik.set_command(torch.cat((target_pos_b, target_quat), dim=-1))

        markers.visualize(translations=target_pos_w)
        print(f"[INFO]: '{label}' -> world {tuple(round(v, 3) for v in target_pos_w[0].tolist())}")

        start_grip = current_grip
        for step in range(duration):
            if not simulation_app.is_running():
                return
            t = min(step / max(duration - 1, 1), 1.0)

            jac = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_cfg.joint_ids]
            ee_pos_b, ee_quat_b = _get_ee_pose_b(robot, ee_body_id)
            joint_pos = robot.data.joint_pos[:, arm_cfg.joint_ids]
            joint_des = ik.compute(ee_pos_b, ee_quat_b, jac, joint_pos)
            hold = joint_des.clone()

            robot.set_joint_position_target(joint_des, joint_ids=arm_cfg.joint_ids)
            _set_gripper(robot, grip_cfg.joint_ids, start_grip + (target_grip - start_grip) * t, num_envs)
            markers.visualize(translations=target_pos_w)
            _step(sim, scene)

        print(f"[INFO]: '{label}' complete.")
        current_pos, _ = _get_ee_pose_b(robot, ee_body_id)
        current_grip = target_grip
        previous_target_w = target_pos_w

        if label == "Lift straight up":
            min_z = TABLE_TOP_Z + BALL_RADIUS + MIN_BALL_LIFT_FOR_HOOP
            ball_z = ball.data.root_pos_w[:, 2]
            if ball_z.min().item() < min_z:
                failed = int((ball_z < min_z).sum().item())
                print(
                    f"[WARN]: Ball not grasped in {failed}/{num_envs} env(s) "
                    f"(min z={ball_z.min().item():.3f}, need>{min_z:.3f}). Stopping."
                )
                break

    while simulation_app.is_running():
        robot.set_joint_position_target(hold, joint_ids=arm_cfg.joint_ids)
        _set_gripper(robot, grip_cfg.joint_ids, current_grip, num_envs)
        _step(sim, scene)


def main():
    sim_cfg = sim_utils.SimulationCfg(
        device=args_cli.device,
        dt=1.0 / 240.0,
        gravity=GRAVITY_XYZ,
        physics_material=TABLE_MATERIAL,
        physx=sim_utils.PhysxCfg(enable_external_forces_every_iteration=True, min_velocity_iteration_count=2),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(2.0, 1.8, 2.1), target=(0.30, -0.10, 1.05))
    scene = InteractiveScene(FrankaBasketballUsdSceneCfg(num_envs=1, env_spacing=2.5))
    sim.reset()
    print("[INFO]: Basketball USD scene ready. Starting task.")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
