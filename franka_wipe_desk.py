# """
# franka_wipe_desk.py

# IsaacLab recreation of an RLBench-style wipe_desk task with a Franka Emika
# Panda arm. The robot grasps a sponge, closes the gripper on it, and wipes over
# the dirt region using scripted IK subgoals.

# Run with:
#     cd ~/IsaacLab
#     ./isaaclab.sh -p scripts/franka_wipe_desk.py
# """

# import argparse
# import math

# from isaaclab.app import AppLauncher

# # ----------------------------------------------------------------------
# # 1. Launch Isaac Sim before importing the rest of Isaac Lab.
# # ----------------------------------------------------------------------
# parser = argparse.ArgumentParser(description="Franka wipe-desk task.")
# AppLauncher.add_app_launcher_args(parser)
# args_cli = parser.parse_args()

# app_launcher = AppLauncher(args_cli)
# simulation_app = app_launcher.app

# # ----------------------------------------------------------------------
# # 2. Safe to import the rest now.
# # ----------------------------------------------------------------------
# import torch

# import isaaclab.sim as sim_utils
# from isaaclab.assets import Articulation, AssetBaseCfg, RigidObjectCfg
# from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
# from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
# from isaaclab.managers import SceneEntityCfg
# from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
# from isaaclab.utils import configclass
# from isaaclab.utils.math import quat_slerp, subtract_frame_transforms
# from pxr import UsdPhysics

# from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG

# # ----------------------------------------------------------------------
# # 3. Task parameters.
# # ----------------------------------------------------------------------
# TABLE_HEIGHT = 1.0
# TABLE_SIZE = (1.6, 1.9, TABLE_HEIGHT)
# GRAVITY_XYZ = (0.0, 0.0, -9.81)
# ROBOT_DISABLE_GRAVITY_FOR_STABLE_IK = True

# SPONGE_SIZE = (0.080, 0.042, 0.028)
# SPONGE_MASS = 0.045
# SPONGE_POS_XYZ = [0.42, -0.14, TABLE_HEIGHT + SPONGE_SIZE[2] / 2.0]
# SPONGE_YAW = -1.717265

# # These x/y deltas come from the RLBench wipe_desk TTM inspection. The original
# # world z values sit on the RLBench table, so this script places them on the
# # IsaacLab table while preserving the sponge-relative dirt layout.
# DIRT_BOUNDARY_DELTA_XY = (0.026102, 0.198311)
# DIRT_WORLD_DELTAS_XY = [
#     (-0.034243, 0.105306),
#     (-0.073154, 0.288558),
#     (0.084825, 0.224747),
#     (0.053136, 0.307493),
#     (0.187654, 0.290303),
#     (-0.007299, 0.293905),
#     (0.149427, 0.075790),
#     (0.138809, 0.037485),
#     (0.014554, 0.286638),
#     (-0.016617, 0.191742),
#     (0.097604, 0.252807),
#     (-0.087565, 0.147470),
#     (0.028953, 0.116021),
#     (0.069776, 0.115449),
#     (0.147600, 0.267318),
#     (0.033979, 0.082064),
#     (0.110339, 0.301118),
#     (0.135764, 0.285152),
#     (-0.061519, 0.322318),
#     (-0.027579, 0.343364),
#     (0.118626, 0.146184),
#     (0.155602, 0.155682),
#     (-0.062563, 0.097114),
#     (0.081882, 0.275862),
#     (-0.043900, 0.238250),
#     (-0.014108, 0.225279),
#     (0.124685, 0.312943),
#     (-0.014441, 0.112572),
#     (-0.058792, 0.353938),
#     (0.035238, 0.095136),
#     (0.085822, 0.318023),
#     (0.110750, 0.177731),
#     (-0.073475, 0.234120),
#     (-0.023924, 0.239751),
#     (0.009773, 0.304937),
#     (-0.077827, 0.271118),
#     (0.122886, 0.285072),
#     (0.075188, 0.145963),
#     (-0.011852, 0.128142),
#     (-0.136078, 0.101618),
#     (-0.107583, 0.156380),
#     (0.119947, 0.088043),
#     (-0.100220, 0.229658),
#     (0.074463, 0.331001),
#     (-0.060108, 0.296502),
#     (0.103707, 0.082465),
#     (0.030275, 0.191439),
#     (-0.042117, 0.060308),
#     (0.045034, 0.252353),
#     (0.079216, 0.122156),
# ]

# DIRT_PIECE_SIZE = (0.018, 0.018, 0.004)
# DIRT_PIECE_MASS = 0.002
# DIRT_CENTER_XY = [
#     SPONGE_POS_XYZ[0] + DIRT_BOUNDARY_DELTA_XY[0],
#     SPONGE_POS_XYZ[1] + DIRT_BOUNDARY_DELTA_XY[1],
# ]
# DIRT_X_MIN = min(SPONGE_POS_XYZ[0] + delta[0] for delta in DIRT_WORLD_DELTAS_XY) - 0.035
# DIRT_X_MAX = max(SPONGE_POS_XYZ[0] + delta[0] for delta in DIRT_WORLD_DELTAS_XY) + 0.035
# DIRT_Y_MIN = min(SPONGE_POS_XYZ[1] + delta[1] for delta in DIRT_WORLD_DELTAS_XY) - 0.035
# DIRT_Y_MAX = max(SPONGE_POS_XYZ[1] + delta[1] for delta in DIRT_WORLD_DELTAS_XY) + 0.035
# DIRT_BOUNDARY_SIZE = (
#     max(DIRT_X_MAX - DIRT_X_MIN, 0.32),
#     max(DIRT_Y_MAX - DIRT_Y_MIN, 0.32),
#     0.002,
# )

# HOME_JOINTS = [0.0, -0.620, 0.0, -2.720, 0.0, 2.960, 0.785]
# GRIPPER_OPEN = 0.04
# GRIPPER_CLOSED = 0.015
# HAND_TO_SPONGE_CENTER_Z = 0.108
# SPONGE_PREGRASP_OFFSET_Z = 0.22
# SPONGE_GRASP_OFFSET_Z = HAND_TO_SPONGE_CENTER_Z
# LIFT_AFTER_GRASP_DELTA_Z = 0.18
# MIN_SPONGE_LIFT_FOR_WIPE = 0.04
# WIPE_CONTACT_Z = HAND_TO_SPONGE_CENTER_Z + 0.010
# WIPE_HIGH_Z = 0.28
# RETREAT_Z = 0.36

# # Quaternion wxyz: hand local z points down toward the table/sponge.
# EE_QUAT_DOWN = [0.0, 1.0, 0.0, 0.0]

# HIGH_FRICTION_MATERIAL = sim_utils.RigidBodyMaterialCfg(
#     static_friction=3.0,
#     dynamic_friction=2.4,
#     restitution=0.0,
#     friction_combine_mode="max",
#     restitution_combine_mode="multiply",
# )
# SPONGE_MATERIAL = sim_utils.RigidBodyMaterialCfg(
#     static_friction=4.0,
#     dynamic_friction=3.2,
#     restitution=0.0,
#     friction_combine_mode="max",
#     restitution_combine_mode="multiply",
# )
# DIRT_MATERIAL = sim_utils.RigidBodyMaterialCfg(
#     static_friction=0.8,
#     dynamic_friction=0.5,
#     restitution=0.0,
#     friction_combine_mode="average",
#     restitution_combine_mode="multiply",
# )
# GRIPPER_FRICTION_MATERIAL = sim_utils.RigidBodyMaterialCfg(
#     static_friction=4.5,
#     dynamic_friction=3.5,
#     restitution=0.0,
#     friction_combine_mode="max",
#     restitution_combine_mode="multiply",
# )
# CONTACT_PROPS = sim_utils.CollisionPropertiesCfg(
#     contact_offset=0.004,
#     rest_offset=0.0,
# )

# SUBGOAL_PARAMS = [
#     {
#         "label": "Approach sponge",
#         "reference": "sponge",
#         "offset_xyz": [0.0, 0.0, SPONGE_PREGRASP_OFFSET_Z],
#         "gripper": "open",
#         "duration_steps": 220,
#     },
#     {
#         "label": "Lower to sponge",
#         "reference": "sponge",
#         "offset_xyz": [0.0, 0.0, SPONGE_GRASP_OFFSET_Z],
#         "gripper": "open",
#         "duration_steps": 150,
#     },
#     {
#         "label": "Close gripper on sponge",
#         "reference": "sponge",
#         "offset_xyz": [0.0, 0.0, SPONGE_GRASP_OFFSET_Z],
#         "gripper": "closed",
#         "duration_steps": 140,
#     },
#     {
#         "label": "Lift sponge",
#         "reference": "previous",
#         "delta_xyz": [0.0, 0.0, LIFT_AFTER_GRASP_DELTA_Z],
#         "gripper": "closed",
#         "duration_steps": 220,
#     },
#     {
#         "label": "Move above dirt start",
#         "reference": "point",
#         "point_xyz": [DIRT_X_MIN, DIRT_Y_MIN, WIPE_HIGH_Z],
#         "gripper": "closed",
#         "duration_steps": 260,
#     },
#     {
#         "label": "Lower sponge to desk",
#         "reference": "point",
#         "point_xyz": [DIRT_X_MIN, DIRT_Y_MIN, WIPE_CONTACT_Z],
#         "gripper": "closed",
#         "duration_steps": 160,
#     },
#     {
#         "label": "Wipe stroke right",
#         "reference": "point",
#         "point_xyz": [DIRT_X_MAX, DIRT_Y_MIN, WIPE_CONTACT_Z],
#         "gripper": "closed",
#         "duration_steps": 300,
#     },
#     {
#         "label": "Wipe stroke forward",
#         "reference": "point",
#         "point_xyz": [DIRT_X_MAX, DIRT_Y_MAX, WIPE_CONTACT_Z],
#         "gripper": "closed",
#         "duration_steps": 260,
#     },
#     {
#         "label": "Wipe stroke left",
#         "reference": "point",
#         "point_xyz": [DIRT_X_MIN, DIRT_Y_MAX, WIPE_CONTACT_Z],
#         "gripper": "closed",
#         "duration_steps": 300,
#     },
#     {
#         "label": "Second pass center",
#         "reference": "point",
#         "point_xyz": [(DIRT_X_MIN + DIRT_X_MAX) / 2.0, DIRT_CENTER_XY[1], WIPE_CONTACT_Z],
#         "gripper": "closed",
#         "duration_steps": 240,
#     },
#     {
#         "label": "Lift sponge clear",
#         "reference": "point",
#         "point_xyz": [(DIRT_X_MIN + DIRT_X_MAX) / 2.0, DIRT_CENTER_XY[1], WIPE_HIGH_Z],
#         "gripper": "closed",
#         "duration_steps": 160,
#     },
#     {
#         "label": "Open gripper",
#         "reference": "point",
#         "point_xyz": [(DIRT_X_MIN + DIRT_X_MAX) / 2.0, DIRT_CENTER_XY[1], WIPE_HIGH_Z],
#         "gripper": "open",
#         "duration_steps": 100,
#     },
#     {
#         "label": "Retreat",
#         "reference": "point",
#         "point_xyz": [DIRT_X_MIN - 0.08, DIRT_Y_MIN - 0.04, RETREAT_Z],
#         "gripper": "open",
#         "duration_steps": 180,
#     },
# ]


# def _yaw_quat(yaw: float) -> tuple[float, float, float, float]:
#     return (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))


# def _gripper_width(state: str) -> float:
#     return GRIPPER_OPEN if state == "open" else GRIPPER_CLOSED


# # ----------------------------------------------------------------------
# # 4. Scene configuration.
# # ----------------------------------------------------------------------
# @configclass
# class FrankaWipeDeskSceneCfg(InteractiveSceneCfg):
#     """Ground plane, light, desk, sponge, dirt region, and Franka."""

#     ground = AssetBaseCfg(
#         prim_path="/World/ground",
#         spawn=sim_utils.GroundPlaneCfg(),
#     )

#     dome_light = AssetBaseCfg(
#         prim_path="/World/Light",
#         spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9)),
#     )

#     table = RigidObjectCfg(
#         prim_path="{ENV_REGEX_NS}/Table",
#         spawn=sim_utils.CuboidCfg(
#             size=TABLE_SIZE,
#             rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=False),
#             collision_props=CONTACT_PROPS,
#             physics_material=HIGH_FRICTION_MATERIAL,
#             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.30, 0.20)),
#         ),
#         init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, TABLE_HEIGHT / 2.0)),
#     )

#     dirt_boundary = AssetBaseCfg(
#         prim_path="{ENV_REGEX_NS}/DirtBoundary",
#         spawn=sim_utils.CuboidCfg(
#             size=DIRT_BOUNDARY_SIZE,
#             visual_material=sim_utils.PreviewSurfaceCfg(
#                 diffuse_color=(0.12, 0.10, 0.08),
#                 opacity=0.22,
#                 roughness=0.95,
#             ),
#         ),
#         init_state=AssetBaseCfg.InitialStateCfg(
#             pos=(DIRT_CENTER_XY[0], DIRT_CENTER_XY[1], TABLE_HEIGHT + 0.001),
#         ),
#     )

#     sponge = RigidObjectCfg(
#         prim_path="{ENV_REGEX_NS}/Sponge",
#         spawn=sim_utils.CuboidCfg(
#             size=SPONGE_SIZE,
#             rigid_props=sim_utils.RigidBodyPropertiesCfg(
#                 kinematic_enabled=False,
#                 disable_gravity=False,
#                 solver_position_iteration_count=24,
#                 solver_velocity_iteration_count=4,
#                 max_depenetration_velocity=1.5,
#             ),
#             collision_props=CONTACT_PROPS,
#             physics_material=SPONGE_MATERIAL,
#             mass_props=sim_utils.MassPropertiesCfg(mass=SPONGE_MASS),
#             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.84, 0.25), roughness=0.92),
#         ),
#         init_state=RigidObjectCfg.InitialStateCfg(
#             pos=tuple(SPONGE_POS_XYZ),
#             rot=_yaw_quat(SPONGE_YAW),
#         ),
#     )

#     robot: Articulation = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
#     robot.spawn.rigid_props.disable_gravity = ROBOT_DISABLE_GRAVITY_FOR_STABLE_IK
#     robot.spawn.articulation_props.solver_velocity_iteration_count = 2
#     robot.init_state.pos = (0.0, 0.0, TABLE_HEIGHT)
#     robot.init_state.joint_pos = {
#         "panda_joint1": HOME_JOINTS[0],
#         "panda_joint2": HOME_JOINTS[1],
#         "panda_joint3": HOME_JOINTS[2],
#         "panda_joint4": HOME_JOINTS[3],
#         "panda_joint5": HOME_JOINTS[4],
#         "panda_joint6": HOME_JOINTS[5],
#         "panda_joint7": HOME_JOINTS[6],
#         "panda_finger_joint.*": GRIPPER_OPEN,
#     }


# def _spawn_dirt_pieces(scene: InteractiveScene):
#     """Spawn small dynamic dirt blocks from the TTM sponge-relative layout."""
#     dirt_cfg = sim_utils.CuboidCfg(
#         size=DIRT_PIECE_SIZE,
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             kinematic_enabled=False,
#             disable_gravity=False,
#             solver_position_iteration_count=8,
#             solver_velocity_iteration_count=2,
#             max_depenetration_velocity=1.0,
#         ),
#         collision_props=CONTACT_PROPS,
#         physics_material=DIRT_MATERIAL,
#         mass_props=sim_utils.MassPropertiesCfg(mass=DIRT_PIECE_MASS),
#         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.08, 0.065, 0.045), roughness=0.95),
#     )

#     for env_index in range(scene.num_envs):
#         env_path = f"/World/envs/env_{env_index}"
#         for dirt_index, (delta_x, delta_y) in enumerate(DIRT_WORLD_DELTAS_XY):
#             dirt_cfg.func(
#                 f"{env_path}/DirtPiece{dirt_index:02d}",
#                 dirt_cfg,
#                 translation=(
#                     SPONGE_POS_XYZ[0] + delta_x,
#                     SPONGE_POS_XYZ[1] + delta_y,
#                     TABLE_HEIGHT + DIRT_PIECE_SIZE[2] / 2.0,
#                 ),
#             )

#     print(f"[INFO]: Spawned {len(DIRT_WORLD_DELTAS_XY)} dynamic dirt pieces per environment.")


# def _repeat(values: list[float], num_envs: int, device: str) -> torch.Tensor:
#     return torch.tensor(values, dtype=torch.float32, device=device).unsqueeze(0).repeat(num_envs, 1)


# def _smoothstep(t: float) -> float:
#     return t * t * (3.0 - 2.0 * t)


# def _quat_slerp_batch(start_quat: torch.Tensor, target_quat: torch.Tensor, t: float) -> torch.Tensor:
#     return torch.stack([quat_slerp(q0, q1, t) for q0, q1 in zip(start_quat, target_quat)])


# def _make_subgoal_markers() -> VisualizationMarkers:
#     marker_cfg = VisualizationMarkersCfg(
#         prim_path="/Visuals/WipeDeskSubgoals",
#         markers={
#             "active": sim_utils.SphereCfg(
#                 radius=0.012,
#                 visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.35, 1.0)),
#             ),
#         },
#     )
#     return VisualizationMarkers(marker_cfg)


# def _base_to_world_pos(target_pos_b: torch.Tensor, env_origins: torch.Tensor) -> torch.Tensor:
#     target_pos_w = target_pos_b.clone()
#     target_pos_w[:, 2] += TABLE_HEIGHT
#     return target_pos_w + env_origins


# def _object_pos_b(scene: InteractiveScene, object_name: str) -> torch.Tensor:
#     object_pos_w = scene[object_name].data.root_pos_w
#     object_pos_b = object_pos_w - scene.env_origins
#     object_pos_b[:, 2] -= TABLE_HEIGHT
#     return object_pos_b


# def _point_b(params: dict, num_envs: int, device: str) -> torch.Tensor:
#     return _repeat(params["point_xyz"], num_envs, device)


# def _resolve_subgoal_target(
#     params: dict,
#     scene: InteractiveScene,
#     previous_target_b: torch.Tensor,
#     device: str,
# ) -> torch.Tensor:
#     reference = params["reference"]
#     if reference == "sponge":
#         target_pos_b = _object_pos_b(scene, "sponge")
#         offset = _repeat(params["offset_xyz"], scene.num_envs, device)
#         return target_pos_b + offset
#     if reference == "point":
#         return _point_b(params, scene.num_envs, device)
#     if reference == "previous":
#         delta = _repeat(params["delta_xyz"], scene.num_envs, device)
#         return previous_target_b + delta
#     raise ValueError(f"Unsupported subgoal reference: {reference}")


# def _print_subgoals():
#     print("[INFO]: Wipe-desk subgoals:")
#     for index, params in enumerate(SUBGOAL_PARAMS):
#         if params["reference"] == "sponge":
#             relation = f"sponge + {tuple(params['offset_xyz'])}"
#         elif params["reference"] == "previous":
#             relation = f"previous + {tuple(params['delta_xyz'])}"
#         else:
#             relation = f"point {tuple(round(v, 3) for v in params['point_xyz'])}"
#         print(
#             f"  {index}: {params['label']}: target={relation}, "
#             f"gripper={params['gripper']}, steps={params['duration_steps']}"
#         )


# def _set_gripper(robot: Articulation, gripper_joint_ids: list[int], width: float, num_envs: int):
#     target = torch.full((num_envs, len(gripper_joint_ids)), width, dtype=torch.float32, device=robot.device)
#     robot.set_joint_position_target(target, joint_ids=gripper_joint_ids)


# def _get_ee_pose_b(robot: Articulation, ee_body_id: int) -> tuple[torch.Tensor, torch.Tensor]:
#     ee_pose_w = robot.data.body_pose_w[:, ee_body_id]
#     root_pose_w = robot.data.root_pose_w
#     return subtract_frame_transforms(
#         root_pose_w[:, 0:3],
#         root_pose_w[:, 3:7],
#         ee_pose_w[:, 0:3],
#         ee_pose_w[:, 3:7],
#     )


# def _write_home_state(robot: Articulation):
#     joint_pos = robot.data.default_joint_pos.clone()
#     joint_vel = torch.zeros_like(joint_pos)
#     robot.write_joint_state_to_sim(joint_pos, joint_vel)
#     robot.set_joint_position_target(joint_pos)
#     robot.reset()


# def _step(sim: sim_utils.SimulationContext, scene: InteractiveScene):
#     scene.write_data_to_sim()
#     sim.step()
#     scene.update(sim.get_physics_dt())


# def _apply_gripper_friction():
#     material_path = "/World/PhysicsMaterials/GripperHighFriction"
#     GRIPPER_FRICTION_MATERIAL.func(material_path, GRIPPER_FRICTION_MATERIAL)

#     stage = sim_utils.get_current_stage()
#     bound_paths = set()
#     bound_count = 0
#     for prim in stage.Traverse():
#         prim_path = str(prim.GetPath())
#         is_finger = "panda_leftfinger" in prim_path or "panda_rightfinger" in prim_path
#         looks_like_collision = "collision" in prim_path.lower() or prim.GetName() in {
#             "panda_leftfinger",
#             "panda_rightfinger",
#         }
#         if is_finger and (prim.HasAPI(UsdPhysics.CollisionAPI) or looks_like_collision) and prim_path not in bound_paths:
#             sim_utils.bind_physics_material(prim_path, material_path)
#             bound_paths.add(prim_path)
#             bound_count += 1

#     print(f"[INFO]: Applied high-friction gripper material to {bound_count} fingertip prims.")


# def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
#     robot = scene["robot"]
#     sponge = scene["sponge"]
#     num_envs = scene.num_envs

#     arm_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
#     arm_entity_cfg.resolve(scene)
#     gripper_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"])
#     gripper_entity_cfg.resolve(scene)

#     ee_jacobi_idx = arm_entity_cfg.body_ids[0] - 1 if robot.is_fixed_base else arm_entity_cfg.body_ids[0]

#     diff_ik_controller = DifferentialIKController(
#         DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
#         num_envs=num_envs,
#         device=robot.device,
#     )

#     _write_home_state(robot)
#     scene.reset()
#     _apply_gripper_friction()
#     _set_gripper(robot, gripper_entity_cfg.joint_ids, GRIPPER_OPEN, num_envs)

#     for _ in range(60):
#         _step(sim, scene)
#     diff_ik_controller.reset()

#     subgoal_markers = _make_subgoal_markers()
#     _print_subgoals()
#     print(
#         "[INFO]: Sponge and dirt are physical rigid bodies. "
#         "The sponge is moved only by gripper contact/friction once grasped."
#     )

#     current_pos, current_quat = _get_ee_pose_b(robot, arm_entity_cfg.body_ids[0])
#     current_grip = GRIPPER_OPEN
#     hold_arm_joint_pos = robot.data.joint_pos[:, arm_entity_cfg.joint_ids].clone()

#     for params in SUBGOAL_PARAMS:
#         label = params["label"]
#         target_grip = _gripper_width(params["gripper"])
#         duration = params["duration_steps"]
#         start_pos = current_pos.clone()
#         start_quat = current_quat.clone()
#         target_pos_tensor = _resolve_subgoal_target(params, scene, current_pos, robot.device)
#         target_quat_tensor = _repeat(EE_QUAT_DOWN, num_envs, robot.device)
#         active_marker_w = _base_to_world_pos(target_pos_tensor, scene.env_origins)
#         subgoal_markers.visualize(translations=active_marker_w)
#         print(
#             f"[INFO]: Starting '{label}' -> hand base xyz="
#             f"{tuple(round(v, 3) for v in target_pos_tensor[0].tolist())}"
#         )

#         for phase_step in range(duration):
#             if not simulation_app.is_running():
#                 return

#             t = _smoothstep(min(phase_step / max(duration - 1, 1), 1.0))
#             command_pos = start_pos + (target_pos_tensor - start_pos) * t
#             command_quat = _quat_slerp_batch(start_quat, target_quat_tensor, t)
#             command = torch.cat((command_pos, command_quat), dim=-1)
#             diff_ik_controller.set_command(command)

#             jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_entity_cfg.joint_ids]
#             ee_pos_b, ee_quat_b = _get_ee_pose_b(robot, arm_entity_cfg.body_ids[0])
#             joint_pos = robot.data.joint_pos[:, arm_entity_cfg.joint_ids]
#             joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
#             hold_arm_joint_pos = joint_pos_des.clone()

#             robot.set_joint_position_target(joint_pos_des, joint_ids=arm_entity_cfg.joint_ids)
#             _set_gripper(
#                 robot,
#                 gripper_entity_cfg.joint_ids,
#                 current_grip + (target_grip - current_grip) * t,
#                 num_envs,
#             )
#             subgoal_markers.visualize(translations=active_marker_w)
#             _step(sim, scene)

#         print(f"[INFO]: '{label}' complete.")
#         current_pos = target_pos_tensor
#         current_quat = target_quat_tensor
#         current_grip = target_grip

#         if label == "Lift sponge":
#             min_sponge_z = TABLE_HEIGHT + SPONGE_SIZE[2] / 2.0 + MIN_SPONGE_LIFT_FOR_WIPE
#             sponge_z = sponge.data.root_pos_w[0, 2].item()
#             if sponge_z < min_sponge_z:
#                 print(
#                     "[WARN]: Sponge was not physically grasped "
#                     f"(z={sponge_z:.3f}, required>{min_sponge_z:.3f}). Stopping before wipe motion."
#                 )
#                 break

#     sponge_xyz = sponge.data.root_pos_w[0].tolist()
#     print(f"[INFO]: Wipe sequence complete. Sponge world xyz={tuple(round(v, 3) for v in sponge_xyz)}")

#     while simulation_app.is_running():
#         robot.set_joint_position_target(
#             hold_arm_joint_pos,
#             joint_ids=arm_entity_cfg.joint_ids,
#         )
#         _set_gripper(robot, gripper_entity_cfg.joint_ids, current_grip, num_envs)
#         _step(sim, scene)


# def main():
#     sim_cfg = sim_utils.SimulationCfg(
#         device=args_cli.device,
#         dt=1.0 / 240.0,
#         gravity=GRAVITY_XYZ,
#         physics_material=HIGH_FRICTION_MATERIAL,
#         physx=sim_utils.PhysxCfg(
#             enable_external_forces_every_iteration=True,
#             min_velocity_iteration_count=2,
#         ),
#     )
#     sim = sim_utils.SimulationContext(sim_cfg)
#     sim.set_camera_view(eye=(1.9, 1.8, 2.25), target=(0.42, 0.02, 1.08))

#     scene_cfg = FrankaWipeDeskSceneCfg(num_envs=1, env_spacing=2.0)
#     scene = InteractiveScene(scene_cfg)
#     _spawn_dirt_pieces(scene)

#     sim.reset()
#     print("[INFO]: Wipe-desk scene ready. Starting task.")
#     run_simulator(sim, scene)


# if __name__ == "__main__":
#     main()
#     simulation_app.close()
