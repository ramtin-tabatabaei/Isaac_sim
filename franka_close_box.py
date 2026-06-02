# """
# franka_close_box_dynamic_hinge_outside.py

# Franka Emika Panda closes a box with a real physical hinged lid.

# This version uses the older, simpler setup:
#   * the box base is a kinematic RigidObject
#   * the lid is a dynamic RigidObject
#   * a UsdPhysics.RevoluteJoint is created after the scene spawns
#   * the lid is spawned outside/back from the box, not inside the cavity

# Run with:
#     cd ~/IsaacLab
#     ./isaaclab.sh -p scripts/franka_close_box.py
# """

# import argparse
# import math

# from isaaclab.app import AppLauncher

# # ----------------------------------------------------------------------
# # 1. Launch Isaac Sim before importing the rest of Isaac Lab.
# # ----------------------------------------------------------------------
# parser = argparse.ArgumentParser(description="Franka close-box task with dynamic hinged lid.")
# AppLauncher.add_app_launcher_args(parser)
# args_cli = parser.parse_args()

# app_launcher = AppLauncher(args_cli)
# simulation_app = app_launcher.app

# # ----------------------------------------------------------------------
# # 2. Safe to import Isaac Lab / USD after the app is launched.
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
# from pxr import Gf, Sdf, UsdPhysics, PhysxSchema

# from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG

# # ----------------------------------------------------------------------
# # 3. Scene constants.
# # ----------------------------------------------------------------------
# TABLE_HEIGHT = 1.0
# TABLE_SIZE = (1.6, 1.9, TABLE_HEIGHT)
# GRAVITY_XYZ = (0.0, 0.0, -9.81)
# ROBOT_DISABLE_GRAVITY_FOR_STABLE_IK = False

# BOX_X, BOX_Y = 0.55, 0.0
# BOX_LENGTH = 0.34
# BOX_WIDTH = 0.28
# BOX_HEIGHT = 0.10
# WALL_THICKNESS = 0.025
# LID_THICKNESS = 0.025
# LID_MASS = 0.12

# # Hinge runs along the y-axis at the rear/top edge of the box.
# # The rear edge is the side closer to the robot.
# HINGE_X = BOX_X - BOX_LENGTH / 2.0
# HINGE_Y = BOX_Y
# HINGE_Z = TABLE_HEIGHT + BOX_HEIGHT + LID_THICKNESS / 2.0

# # 0 rad means closed and flat; 90 deg is straight up.  This signed offset is
# # measured from vertical, where positive leans toward the robot and negative
# # leans away from it.  -30 deg gives a physical hinge angle of 60 deg.
# LID_OPEN_SIGNED_FROM_VERTICAL = math.radians(-30.0)
# LID_OPEN_ANGLE = math.radians(90.0) + LID_OPEN_SIGNED_FROM_VERTICAL
# LID_CLOSED_ANGLE = math.radians(1.0)
# LID_CLOSED_SUCCESS_ANGLE = math.radians(8.0)

# # Joint limits in the same convention used by this script.
# LID_JOINT_LOWER = math.radians(0.0)
# LID_JOINT_UPPER = math.radians(125.0)

# # A passive spring-damper holds the lid open while the robot approaches, then we
# # reduce it to a weak closed latch during the push.  Values are specified in
# # N*m/rad and N*m*s/rad, then converted for the USD angular drive.
# LID_OPEN_HOLD_STIFFNESS = 2.8
# LID_OPEN_HOLD_DAMPING = 0.30
# LID_LATCH_STIFFNESS = 0.04
# LID_LATCH_DAMPING = 0.03
# LID_MIN_SETTLED_OPEN_ANGLE = math.radians(45.0)

# LID_CONTACT_FRACTION = 0.88
# HAND_APPROACH_CLEARANCE_Z = 0.105
# HAND_CONTACT_CLEARANCE_Z = 0.040
# HAND_PUSH_CLEARANCE_Z = 0.055
# HAND_PRESS_CLEARANCE_Z = 0.075

# HIGH_FRICTION_MATERIAL = sim_utils.RigidBodyMaterialCfg(
#     static_friction=4.0,
#     dynamic_friction=3.0,
#     restitution=0.0,
#     friction_combine_mode="max",
#     restitution_combine_mode="multiply",
# )
# STICKY_CONTACT_PROPS = sim_utils.CollisionPropertiesCfg(
#     contact_offset=0.005,
#     rest_offset=0.0,
#     torsional_patch_radius=0.03,
#     min_torsional_patch_radius=0.01,
# )

# SUBGOAL_PARAMS = [
#     {
#         "label": "Approach open lid",
#         "lid_angle": LID_OPEN_ANGLE,
#         "offset_xyz": [-0.120, 0.0, HAND_APPROACH_CLEARANCE_Z],
#         "gripper": "open",
#         "duration_steps": 240,
#     },
#     {
#         "label": "Close gripper into a pushing tool",
#         "lid_angle": LID_OPEN_ANGLE,
#         "offset_xyz": [-0.120, 0.0, HAND_APPROACH_CLEARANCE_Z],
#         "gripper": "closed",
#         "duration_steps": 100,
#     },
#     {
#         "label": "Make contact with lid",
#         "lid_angle": LID_OPEN_ANGLE,
#         "offset_xyz": [-0.020, 0.0, HAND_CONTACT_CLEARANCE_Z],
#         "gripper": "closed",
#         "duration_steps": 260,
#     },
#     {
#         "label": "Push lid closed",
#         "lid_angle": LID_CLOSED_ANGLE,
#         "offset_xyz": [-0.028, 0.0, HAND_PUSH_CLEARANCE_Z],
#         "gripper": "closed",
#         "duration_steps": 520,
#     },
#     {
#         "label": "Press closed",
#         "lid_angle": LID_CLOSED_ANGLE,
#         "offset_xyz": [-0.002, 0.0, HAND_PRESS_CLEARANCE_Z],
#         "gripper": "closed",
#         "duration_steps": 160,
#     },
#     {
#         "label": "Open gripper",
#         "lid_angle": LID_CLOSED_ANGLE,
#         "offset_xyz": [-0.002, 0.0, HAND_PRESS_CLEARANCE_Z + 0.020],
#         "gripper": "open",
#         "duration_steps": 90,
#     },
#     {
#         "label": "Retreat",
#         "lid_angle": LID_CLOSED_ANGLE,
#         "offset_xyz": [-0.006, 0.0, 0.220],
#         "gripper": "open",
#         "duration_steps": 180,
#     },
# ]

# # Slightly taller rest posture so the arm starts clear of the box/lid.
# HOME_JOINTS = [0.0, -0.720, 0.0, -2.580, 0.0, 2.850, 0.785]
# GRIPPER_OPEN = 0.04
# GRIPPER_CLOSED = 0.002


# def _lid_center_world(angle_rad: float) -> list[float]:
#     """Lid COM in world frame for a hinge rotating about the y-axis."""
#     center_x = HINGE_X + math.cos(angle_rad) * (BOX_LENGTH / 2.0)
#     center_z = HINGE_Z + math.sin(angle_rad) * (BOX_LENGTH / 2.0)
#     return [center_x, HINGE_Y, center_z]


# def _lid_quat_wxyz(angle_rad: float) -> list[float]:
#     """Lid orientation in world frame, wxyz.

#     Positive angle opens the lid upward/backward outside the box.
#     """
#     # In USD/PhysX, the matching body rotation for this script's positive
#     # opening angle is negative about Y; this keeps the hinge anchor fixed.
#     half = -0.5 * angle_rad
#     return [math.cos(half), 0.0, math.sin(half), 0.0]


# def _usd_lid_joint_degrees(script_angle_rad: float) -> float:
#     """Convert script lid angle to USD joint degrees."""
#     return -math.degrees(script_angle_rad)


# def _usd_angular_drive_gain(value_rad_units: float) -> float:
#     """Convert angular drive gains from per-radian to per-degree USD units."""
#     return value_rad_units * math.pi / 180.0


# def _hand_quat_for_lid(angle_rad: float) -> list[float]:
#     """Orient panda_hand so its local z-axis approaches along the lid normal."""
#     half = -0.5 * angle_rad
#     c, s = math.cos(half), math.sin(half)
#     return [0.0, c, 0.0, -s]


# def _lid_contact_point_base(angle_rad: float) -> list[float]:
#     """Lid contact point in robot-base frame, before user offsets."""
#     contact_distance = BOX_LENGTH * LID_CONTACT_FRACTION
#     front_x = HINGE_X + math.cos(angle_rad) * contact_distance
#     front_z_world = HINGE_Z + math.sin(angle_rad) * contact_distance
#     return [front_x, BOX_Y, front_z_world - TABLE_HEIGHT]


# def _hand_push_point_base(angle_rad: float, offset_xyz: list[float]) -> list[float]:
#     """Hand target in robot-base frame after applying xyz offset."""
#     contact_point = _lid_contact_point_base(angle_rad)
#     return [
#         contact_point[0] + offset_xyz[0],
#         contact_point[1] + offset_xyz[1],
#         contact_point[2] + offset_xyz[2],
#     ]


# def _gripper_width(state: str) -> float:
#     return GRIPPER_OPEN if state == "open" else GRIPPER_CLOSED


# PHASES = [
#     (
#         params["label"],
#         params["lid_angle"],
#         params["offset_xyz"],
#         _gripper_width(params["gripper"]),
#         params["duration_steps"],
#     )
#     for params in SUBGOAL_PARAMS
# ]

# # ----------------------------------------------------------------------
# # 4. Scene configuration.
# # ----------------------------------------------------------------------
# @configclass
# class FrankaCloseBoxSceneCfg(InteractiveSceneCfg):
#     """Ground, table, kinematic box base, dynamic lid, and Franka."""

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
#             collision_props=STICKY_CONTACT_PROPS,
#             physics_material=HIGH_FRICTION_MATERIAL,
#             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.30, 0.20)),
#         ),
#         init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, TABLE_HEIGHT / 2.0)),
#     )

#     box_base = RigidObjectCfg(
#         prim_path="{ENV_REGEX_NS}/BoxBase",
#         spawn=sim_utils.CuboidCfg(
#             size=(BOX_LENGTH, BOX_WIDTH, BOX_HEIGHT),
#             rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=False),
#             collision_props=STICKY_CONTACT_PROPS,
#             physics_material=HIGH_FRICTION_MATERIAL,
#             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.24, 0.18, 0.12)),
#         ),
#         init_state=RigidObjectCfg.InitialStateCfg(
#             pos=(BOX_X, BOX_Y, TABLE_HEIGHT + BOX_HEIGHT / 2.0),
#         ),
#     )

#     # Visual-only dark cavity so the box looks hollow.
#     box_cavity = AssetBaseCfg(
#         prim_path="{ENV_REGEX_NS}/BoxCavity",
#         spawn=sim_utils.CuboidCfg(
#             size=(BOX_LENGTH - 2.0 * WALL_THICKNESS, BOX_WIDTH - 2.0 * WALL_THICKNESS, BOX_HEIGHT + 0.004),
#             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.04, 0.035)),
#         ),
#         init_state=AssetBaseCfg.InitialStateCfg(
#             pos=(BOX_X, BOX_Y, TABLE_HEIGHT + BOX_HEIGHT / 2.0 + 0.003),
#         ),
#     )

#     lid = RigidObjectCfg(
#         prim_path="{ENV_REGEX_NS}/BoxLid",
#         spawn=sim_utils.CuboidCfg(
#             size=(BOX_LENGTH, BOX_WIDTH, LID_THICKNESS),
#             rigid_props=sim_utils.RigidBodyPropertiesCfg(
#                 kinematic_enabled=False,
#                 disable_gravity=False,
#                 solver_position_iteration_count=32,
#                 solver_velocity_iteration_count=4,
#                 max_depenetration_velocity=0.5,
#             ),
#             mass_props=sim_utils.MassPropertiesCfg(mass=LID_MASS),
#             collision_props=STICKY_CONTACT_PROPS,
#             physics_material=HIGH_FRICTION_MATERIAL,
#             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.60, 0.36, 0.18)),
#         ),
#         init_state=RigidObjectCfg.InitialStateCfg(
#             pos=tuple(_lid_center_world(LID_OPEN_ANGLE)),
#             rot=tuple(_lid_quat_wxyz(LID_OPEN_ANGLE)),
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

# # ----------------------------------------------------------------------
# # 5. Hinge construction and helpers.
# # ----------------------------------------------------------------------
# def _add_lid_hinge_joint(scene: InteractiveScene) -> list[str]:
#     """Create one UsdPhysics.RevoluteJoint per environment.

#     The joint is created after InteractiveScene has spawned BoxBase and BoxLid,
#     but before sim.reset(), so PhysX picks it up while parsing the scene.
#     """
#     stage = sim_utils.get_current_stage()
#     joint_paths = []

#     for env_index in range(scene.num_envs):
#         env_origin = scene.env_origins[env_index].tolist()
#         env_path = f"/World/envs/env_{env_index}"
#         base_path = f"{env_path}/BoxBase"
#         lid_path = f"{env_path}/BoxLid"
#         joint_path = f"{base_path}/LidHinge"

#         hinge_w = Gf.Vec3f(
#             HINGE_X + env_origin[0],
#             HINGE_Y + env_origin[1],
#             HINGE_Z + env_origin[2],
#         )
#         base_center_w = Gf.Vec3f(
#             BOX_X + env_origin[0],
#             BOX_Y + env_origin[1],
#             TABLE_HEIGHT + BOX_HEIGHT / 2.0 + env_origin[2],
#         )
#         anchor_in_base = hinge_w - base_center_w

#         # Hinge edge in the lid's local frame.  This is independent of angle.
#         anchor_in_lid = Gf.Vec3f(-BOX_LENGTH / 2.0, 0.0, 0.0)

#         joint = UsdPhysics.RevoluteJoint.Define(stage, Sdf.Path(joint_path))
#         joint.CreateBody0Rel().SetTargets([Sdf.Path(base_path)])
#         joint.CreateBody1Rel().SetTargets([Sdf.Path(lid_path)])
#         joint.CreateAxisAttr("Y")
#         joint.CreateLocalPos0Attr(anchor_in_base)
#         joint.CreateLocalPos1Attr(anchor_in_lid)
#         joint.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
#         joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
#         joint_limits = sorted(
#             (
#                 _usd_lid_joint_degrees(LID_JOINT_LOWER),
#                 _usd_lid_joint_degrees(LID_JOINT_UPPER),
#             )
#         )
#         joint.CreateLowerLimitAttr(joint_limits[0])
#         joint.CreateUpperLimitAttr(joint_limits[1])

#         drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
#         drive.CreateTypeAttr("force")
#         drive.CreateStiffnessAttr(_usd_angular_drive_gain(LID_OPEN_HOLD_STIFFNESS))
#         drive.CreateDampingAttr(_usd_angular_drive_gain(LID_OPEN_HOLD_DAMPING))
#         drive.CreateTargetPositionAttr(_usd_lid_joint_degrees(LID_OPEN_ANGLE))

#         physx_joint = PhysxSchema.PhysxJointAPI.Apply(joint.GetPrim())
#         physx_joint.CreateJointFrictionAttr(0.08)

#         joint_paths.append(joint_path)

#     print(f"[INFO]: Created {len(joint_paths)} physical lid hinge joint(s).")
#     return joint_paths


# def _set_lid_drive(joint_paths: list[str], target_angle: float, stiffness: float, damping: float):
#     stage = sim_utils.get_current_stage()
#     for joint_path in joint_paths:
#         joint_prim = stage.GetPrimAtPath(joint_path)
#         drive = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
#         drive.GetTargetPositionAttr().Set(_usd_lid_joint_degrees(target_angle))
#         drive.GetStiffnessAttr().Set(_usd_angular_drive_gain(stiffness))
#         drive.GetDampingAttr().Set(_usd_angular_drive_gain(damping))


# def _read_lid_angle(scene: InteractiveScene) -> torch.Tensor:
#     """Recover lid angle from measured lid orientation.

#     This is an approximate readout for reporting/task success.  The lid itself
#     is physical and is not controlled by this function.
#     """
#     quat_w = scene["lid"].data.root_quat_w  # (num_envs, 4), wxyz
#     qw = quat_w[:, 0]
#     qy = quat_w[:, 2]
#     return -2.0 * torch.atan2(qy, qw)


# def _lid_angle_signed_from_vertical(lid_angle: torch.Tensor) -> torch.Tensor:
#     return lid_angle - math.radians(90.0)


# def _repeat(values: list[float], num_envs: int, device: str) -> torch.Tensor:
#     return torch.tensor(values, dtype=torch.float32, device=device).unsqueeze(0).repeat(num_envs, 1)


# def _smoothstep(t: float) -> float:
#     return t * t * (3.0 - 2.0 * t)


# def _lerp_xyz(start_xyz: list[float], target_xyz: list[float], t: float) -> list[float]:
#     return [start + (target - start) * t for start, target in zip(start_xyz, target_xyz)]


# def _quat_slerp_batch(start_quat: torch.Tensor, target_quat: torch.Tensor, t: float) -> torch.Tensor:
#     return torch.stack([quat_slerp(q0, q1, t) for q0, q1 in zip(start_quat, target_quat)])


# def _make_subgoal_markers() -> VisualizationMarkers:
#     marker_cfg = VisualizationMarkersCfg(
#         prim_path="/Visuals/CloseBoxSubgoals",
#         markers={
#             "active": sim_utils.SphereCfg(
#                 radius=0.012,
#                 visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.35, 1.0)),
#             ),
#         },
#     )
#     return VisualizationMarkers(marker_cfg)


# def _subgoal_position_world(
#     lid_angle: float, offset_xyz: list[float], device: str, env_origin: torch.Tensor
# ) -> torch.Tensor:
#     position = torch.tensor(_hand_push_point_base(lid_angle, offset_xyz), dtype=torch.float32, device=device)
#     position[2] += TABLE_HEIGHT
#     return position.unsqueeze(0) + env_origin.unsqueeze(0)


# def _print_subgoals(env_origin: torch.Tensor):
#     print("[INFO]: Close-box subgoal coordinates:")
#     for index, (label, lid_angle, offset_xyz, gripper, duration) in enumerate(PHASES):
#         contact_b = _lid_contact_point_base(lid_angle)
#         pos_b = _hand_push_point_base(lid_angle, offset_xyz)
#         gripper_label = "open" if gripper > GRIPPER_CLOSED else "closed"
#         print(
#             f"  {index}: {label}: contact base xyz={tuple(round(v, 3) for v in contact_b)}, "
#             f"offset xyz={tuple(round(v, 3) for v in offset_xyz)}, "
#             f"hand base xyz={tuple(round(v, 3) for v in pos_b)}, "
#             f"gripper={gripper_label}, steps={duration}"
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


# def _write_lid_open_state(scene: InteractiveScene):
#     lid = scene["lid"]
#     root_state = torch.zeros((scene.num_envs, 13), dtype=torch.float32, device=lid.device)
#     root_state[:, 0:3] = _repeat(_lid_center_world(LID_OPEN_ANGLE), scene.num_envs, lid.device) + scene.env_origins
#     root_state[:, 3:7] = _repeat(_lid_quat_wxyz(LID_OPEN_ANGLE), scene.num_envs, lid.device)
#     lid.write_root_state_to_sim(root_state)
#     lid.reset()


# def _step(sim: sim_utils.SimulationContext, scene: InteractiveScene):
#     scene.write_data_to_sim()
#     sim.step()
#     scene.update(sim.get_physics_dt())


# def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, hinge_joint_paths: list[str]):
#     robot = scene["robot"]
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
#     _write_lid_open_state(scene)
#     _set_lid_drive(
#         hinge_joint_paths,
#         target_angle=LID_OPEN_ANGLE,
#         stiffness=LID_OPEN_HOLD_STIFFNESS,
#         damping=LID_OPEN_HOLD_DAMPING,
#     )
#     _set_gripper(robot, gripper_entity_cfg.joint_ids, GRIPPER_OPEN, num_envs)

#     for _ in range(60):
#         _step(sim, scene)
#     diff_ik_controller.reset()

#     subgoal_markers = _make_subgoal_markers()
#     _print_subgoals(scene.env_origins[0])

#     start_angle = _read_lid_angle(scene)
#     start_signed_angle = _lid_angle_signed_from_vertical(start_angle)
#     print(
#         f"[INFO]: Lid angle after settling: hinge={math.degrees(start_angle.mean().item()):.1f} deg, "
#         f"signed-from-vertical={math.degrees(start_signed_angle.mean().item()):.1f} deg"
#     )
#     if start_angle.min().item() < LID_MIN_SETTLED_OPEN_ANGLE:
#         print(
#             "[WARN]: Lid sagged below the expected open angle during settling. "
#             "Increase LID_OPEN_HOLD_STIFFNESS/LID_OPEN_HOLD_DAMPING if it still drops before contact."
#         )

#     current_pos, current_quat = _get_ee_pose_b(robot, arm_entity_cfg.body_ids[0])
#     current_offset_xyz = SUBGOAL_PARAMS[0]["offset_xyz"]
#     current_grip = GRIPPER_OPEN
#     hold_arm_joint_pos = robot.data.joint_pos[:, arm_entity_cfg.joint_ids].clone()
#     guide_lid_angle = LID_OPEN_ANGLE

#     for label, target_lid_angle, offset_xyz, target_grip, duration in PHASES:
#         if label == "Push lid closed":
#             _set_lid_drive(
#                 hinge_joint_paths,
#                 target_angle=LID_CLOSED_ANGLE,
#                 stiffness=LID_LATCH_STIFFNESS,
#                 damping=LID_LATCH_DAMPING,
#             )
#             print("[INFO]: Released open hold; weak closed latch is active.")

#         start_pos = current_pos.clone()
#         start_quat = current_quat.clone()
#         start_guide_angle = guide_lid_angle
#         start_offset_xyz = current_offset_xyz
#         target_pos_tensor = _repeat(_hand_push_point_base(target_lid_angle, offset_xyz), num_envs, robot.device)
#         target_quat_tensor = _repeat(_hand_quat_for_lid(target_lid_angle), num_envs, robot.device)
#         follows_lid_surface = abs(target_lid_angle - start_guide_angle) > 1e-5
#         active_marker_w = _subgoal_position_world(target_lid_angle, offset_xyz, robot.device, scene.env_origins[0])
#         subgoal_markers.visualize(translations=active_marker_w)

#         for phase_step in range(duration):
#             if not simulation_app.is_running():
#                 return

#             t_raw = min(phase_step / max(duration - 1, 1), 1.0)
#             t = _smoothstep(t_raw)
#             guide_angle = start_guide_angle + (target_lid_angle - start_guide_angle) * t
#             command_offset_xyz = _lerp_xyz(start_offset_xyz, offset_xyz, t)

#             if follows_lid_surface:
#                 command_pos = _repeat(
#                     _hand_push_point_base(guide_angle, command_offset_xyz), num_envs, robot.device
#                 )
#                 command_quat = _repeat(_hand_quat_for_lid(guide_angle), num_envs, robot.device)
#             else:
#                 command_pos = start_pos + (target_pos_tensor - start_pos) * t
#                 command_quat = _quat_slerp_batch(start_quat, target_quat_tensor, t)

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

#         measured = _read_lid_angle(scene)
#         print(
#             f"[INFO]: '{label}' complete. "
#             f"Lid angle now {math.degrees(measured.mean().item()):.1f} deg."
#         )
#         current_pos = target_pos_tensor
#         current_quat = target_quat_tensor
#         guide_lid_angle = target_lid_angle
#         current_offset_xyz = offset_xyz
#         current_grip = target_grip

#     final_angle = _read_lid_angle(scene)
#     worst = final_angle.max().item()
#     if worst <= LID_CLOSED_SUCCESS_ANGLE:
#         print(
#             f"[SUCCESS]: Box closed. Max lid angle "
#             f"{math.degrees(worst):.1f} deg <= {math.degrees(LID_CLOSED_SUCCESS_ANGLE):.1f} deg."
#         )
#     else:
#         failed = int((final_angle > LID_CLOSED_SUCCESS_ANGLE).sum().item())
#         print(
#             f"[WARN]: Lid not fully closed in {failed}/{num_envs} env(s). "
#             f"Max angle {math.degrees(worst):.1f} deg "
#             f"needed <= {math.degrees(LID_CLOSED_SUCCESS_ANGLE):.1f} deg."
#         )

#     while simulation_app.is_running():
#         robot.set_joint_position_target(hold_arm_joint_pos, joint_ids=arm_entity_cfg.joint_ids)
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
#     sim.set_camera_view(eye=(2.0, 1.9, 2.2), target=(0.48, 0.0, 1.08))

#     scene_cfg = FrankaCloseBoxSceneCfg(num_envs=1, env_spacing=2.0)
#     scene = InteractiveScene(scene_cfg)

#     # Create the hinge after prims exist, before sim.reset().
#     hinge_joint_paths = _add_lid_hinge_joint(scene)

#     sim.reset()
#     print("[INFO]: Close-box scene ready with dynamic outside hinge lid. Starting task.")
#     run_simulator(sim, scene, hinge_joint_paths)


# if __name__ == "__main__":
#     main()
#     simulation_app.close()
