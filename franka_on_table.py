"""
franka_on_table.py  (offline version)

Spawns a Franka Emika Panda arm on a wider table with a small red cube,
then executes a simple IK pick-and-place sequence.

Run with:
    cd ~/IsaacLab
    ./isaaclab.sh -p scripts/franka_on_table.py
"""

import argparse

from isaaclab.app import AppLauncher

# ----------------------------------------------------------------------
# 1. Launch the Isaac Sim app FIRST, before any other isaaclab imports.
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Franka pick-and-place on a table (offline).")
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

from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG

# ----------------------------------------------------------------------
# 3. Scene constants.
# ----------------------------------------------------------------------
TABLE_HEIGHT = 1.0
TABLE_SIZE = (1.6, 1.9, TABLE_HEIGHT)
GRAVITY_XYZ = (0.0, 0.0, -9.81)
ROBOT_DISABLE_GRAVITY_FOR_STABLE_IK = True

CUBE_SIZE = 0.04
CUBE_MASS = 0.1
CUBE_POS_XYZ = [0.45, 0.0, TABLE_HEIGHT + CUBE_SIZE / 2.0]
PLACE_POS_XY = [0.30, 0.35]

# Finger joints are prismatic (metres).
GRIPPER_OPEN = 0.04
GRIPPER_CLOSED = 0.0

# The IK controller tracks the panda_hand frame. The finger center is about
# 10.7 cm along the hand's local z-axis, so with the hand pointing down the
# hand frame should stay above the cube by roughly this amount.
HAND_TO_FINGER_CENTER = 0.107
GRASP_HAND_Z = CUBE_SIZE / 2.0 + HAND_TO_FINGER_CENTER + 0.015
ABOVE_PICK_Z = GRASP_HAND_Z + 0.18
LIFT_Z = GRASP_HAND_Z + 0.28
RETREAT_Z = ABOVE_PICK_Z

HIGH_FRICTION_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    static_friction=2.4,
    dynamic_friction=1.8,
    restitution=0.0,
    friction_combine_mode="max",
    restitution_combine_mode="multiply",
)
CUBE_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    static_friction=3.0,
    dynamic_friction=2.4,
    restitution=0.0,
    friction_combine_mode="max",
    restitution_combine_mode="multiply",
)
CONTACT_PROPS = sim_utils.CollisionPropertiesCfg(
    contact_offset=0.004,
    rest_offset=0.0,
)

# Quaternion wxyz: hand local z points down toward the table.
EE_QUAT_DOWN = [0.0, 1.0, 0.0, 0.0]

# ----------------------------------------------------------------------
# 4. Home joint preset [j1 .. j7] (radians).
# ----------------------------------------------------------------------
HOME_JOINTS = [0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.785]

# Subgoal parameters mirror the basketball task style: targets are relative to
# the live cube pose, the place location, or the previous robot target.
SUBGOAL_PARAMS = [
    {
        "label": "Move above cube",
        "reference": "cube",
        "offset_xyz": [0.0, 0.0, ABOVE_PICK_Z - CUBE_SIZE / 2.0],
        "gripper": "open",
        "duration_steps": 180,
    },
    {
        "label": "Lower to cube",
        "reference": "cube",
        "offset_xyz": [0.0, 0.0, GRASP_HAND_Z - CUBE_SIZE / 2.0],
        "gripper": "open",
        "duration_steps": 120,
    },
    {
        "label": "Close gripper",
        "reference": "cube",
        "offset_xyz": [0.0, 0.0, GRASP_HAND_Z - CUBE_SIZE / 2.0],
        "gripper": "closed",
        "duration_steps": 100,
    },
    {
        "label": "Lift cube",
        "reference": "previous",
        "delta_xyz": [0.0, 0.0, LIFT_Z - GRASP_HAND_Z],
        "gripper": "closed",
        "duration_steps": 160,
    },
    {
        "label": "Move to place",
        "reference": "place",
        "offset_xyz": [0.0, 0.0, LIFT_Z],
        "gripper": "closed",
        "duration_steps": 200,
    },
    {
        "label": "Lower to place",
        "reference": "place",
        "offset_xyz": [0.0, 0.0, GRASP_HAND_Z],
        "gripper": "closed",
        "duration_steps": 140,
    },
    {
        "label": "Open gripper",
        "reference": "place",
        "offset_xyz": [0.0, 0.0, GRASP_HAND_Z],
        "gripper": "open",
        "duration_steps": 100,
    },
    {
        "label": "Retreat",
        "reference": "place",
        "offset_xyz": [0.0, 0.0, RETREAT_Z],
        "gripper": "open",
        "duration_steps": 140,
    },
]


# ----------------------------------------------------------------------
# 5. Scene configuration.
# ----------------------------------------------------------------------
@configclass
class FrankaTableSceneCfg(InteractiveSceneCfg):
    """Ground plane, light, wider table, red cube, and Franka."""

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

    # Red cube sitting on the table at the pick location.
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=2,
                max_depenetration_velocity=2.0,
            ),
            collision_props=CONTACT_PROPS,
            physics_material=CUBE_MATERIAL,
            mass_props=sim_utils.MassPropertiesCfg(mass=CUBE_MASS),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.15, 0.15)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(CUBE_POS_XYZ)),
    )

    # Use the high-PD Franka config for IK joint target tracking. The default
    # Franka config is intentionally softer and can visibly sag/oscillate under gravity.
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


def _gripper_width(state: str) -> float:
    return GRIPPER_OPEN if state == "open" else GRIPPER_CLOSED


def _make_subgoal_markers() -> VisualizationMarkers:
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/TablePickPlaceSubgoals",
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


def _place_pos_b(num_envs: int, device: str) -> torch.Tensor:
    return torch.tensor(
        [PLACE_POS_XY[0], PLACE_POS_XY[1], 0.0],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0).repeat(num_envs, 1)


def _resolve_subgoal_target(
    params: dict,
    scene: InteractiveScene,
    previous_target_b: torch.Tensor,
    device: str,
) -> torch.Tensor:
    reference = params["reference"]
    if reference == "cube":
        return _object_pos_b(scene, "cube") + _repeat(params["offset_xyz"], scene.num_envs, device)
    if reference == "place":
        return _place_pos_b(scene.num_envs, device) + _repeat(params["offset_xyz"], scene.num_envs, device)
    if reference == "previous":
        return previous_target_b + _repeat(params["delta_xyz"], scene.num_envs, device)
    raise ValueError(f"Unsupported subgoal reference: {reference}")


def _print_subgoals():
    print("[INFO]: Table pick-and-place subgoals are resolved relative to objects at runtime:")
    for index, params in enumerate(SUBGOAL_PARAMS):
        if params["reference"] == "previous":
            relation = f"previous + {tuple(params['delta_xyz'])}"
        else:
            relation = f"{params['reference']} + {tuple(params['offset_xyz'])}"
        print(
            f"  {index}: {params['label']}: target={relation}, "
            f"gripper={params['gripper']}, steps={params['duration_steps']}"
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


# ----------------------------------------------------------------------
# 7. Simulation loop.
# ----------------------------------------------------------------------
def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    robot    = scene["robot"]
    num_envs = scene.num_envs

    arm_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    arm_entity_cfg.resolve(scene)
    gripper_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"])
    gripper_entity_cfg.resolve(scene)

    if robot.is_fixed_base:
        ee_jacobi_idx = arm_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = arm_entity_cfg.body_ids[0]

    diff_ik_controller = DifferentialIKController(
        DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        num_envs=num_envs,
        device=robot.device,
    )

    # Start with matching simulated state and drive targets, then let contacts settle.
    _write_home_state(robot)
    scene.reset()
    _set_gripper(robot, gripper_entity_cfg.joint_ids, GRIPPER_OPEN, num_envs)
    for _ in range(60):
        _step(sim, scene)
    diff_ik_controller.reset()

    subgoal_markers = _make_subgoal_markers()
    _print_subgoals()

    current_pos, current_quat = _get_ee_pose_b(robot, arm_entity_cfg.body_ids[0])
    current_grip = GRIPPER_OPEN
    hold_arm_joint_pos = robot.data.joint_pos[:, arm_entity_cfg.joint_ids].clone()

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
            command = torch.cat(
                (
                    start_pos + (target_pos_tensor - start_pos) * t,
                    _quat_slerp_batch(start_quat, target_quat_tensor, t),
                ),
                dim=-1,
            )
            diff_ik_controller.set_command(command)

            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_entity_cfg.joint_ids]
            ee_pos_b, ee_quat_b = _get_ee_pose_b(robot, arm_entity_cfg.body_ids[0])
            joint_pos = robot.data.joint_pos[:, arm_entity_cfg.joint_ids]
            joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
            hold_arm_joint_pos = joint_pos_des.clone()

            robot.set_joint_position_target(joint_pos_des, joint_ids=arm_entity_cfg.joint_ids)
            _set_gripper(
                robot,
                gripper_entity_cfg.joint_ids,
                current_grip + (target_grip - current_grip) * t,
                num_envs,
            )
            subgoal_markers.visualize(translations=active_marker_w)
            _step(sim, scene)

        print(f"[INFO]: '{label}' complete.")
        current_pos = target_pos_tensor
        current_quat = target_quat_tensor
        current_grip = target_grip

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
    sim.set_camera_view(eye=(2.0, 2.0, 2.5), target=(0.0, 0.0, 1.0))

    scene_cfg = FrankaTableSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: Scene ready. Starting pick-and-place sequence.")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
