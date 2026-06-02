"""
robot_controller.py

Generic differential-IK waypoint controller for a single Franka arm spawned in a
*standalone* Isaac Lab scene (no ``InteractiveScene``). The caller hands it an
``Articulation`` and a list of end-effector ``Waypoint`` s expressed in the
world frame; the controller interpolates between them and drives the arm with
PhysX-Jacobian differential IK, opening/closing the gripper as requested.

The arm *motion* (the waypoint list) is meant to be scripted/hard-coded per task
by the caller. This module never reads or writes any scene-object poses, so the
placement of the manipulated objects stays entirely data-driven.

Like ``robot_arm``, this imports Isaac Lab modules at import time and must only
be imported after ``AppLauncher`` has started the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import quat_slerp, subtract_frame_transforms
from pxr import UsdPhysics

# End-effector pointing straight down at the table (wxyz).
EE_QUAT_DOWN = (0.0, 1.0, 0.0, 0.0)

# Fixed geometric tool offset (m). The controller drives the ``panda_hand`` body,
# but task waypoints describe the gripper *tip*. With the gripper pointing
# straight down (EE_QUAT_DOWN) the hand sits this far above the tip in world +z,
# so the controller adds it to every commanded target to make the tip land on
# the waypoint. This is robot geometry, not a per-task tuning knob (measured
# panda_hand->fingertip ~0.112, ->grasp-center ~0.085).
TIP_TO_HAND_Z = 0.10


@dataclass
class Waypoint:
    """A single scripted end-effector goal, in the world frame.

    If ``via_points_w`` is set, the end-effector sweeps continuously through that
    ordered list of world points (one smooth, eased motion over ``duration_steps``
    with no stop at each point) instead of a single straight segment to ``pos_w``.
    """

    label: str
    pos_w: tuple[float, float, float]
    quat_w: tuple[float, float, float, float] = EE_QUAT_DOWN
    gripper: str = "open"  # "open" or "closed"
    duration_steps: int = 180
    via_points_w: list[tuple[float, float, float]] | None = None


def _smoothstep(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _point_along_polyline(points: list[tuple[float, float, float]], s: float) -> tuple[float, float, float]:
    """Return the world point at arc-length fraction ``s`` (0..1) along a polyline."""
    if len(points) == 1:
        return points[0]
    seg_lengths = [
        sum((points[i + 1][k] - points[i][k]) ** 2 for k in range(3)) ** 0.5 for i in range(len(points) - 1)
    ]
    total = sum(seg_lengths)
    if total < 1.0e-9:
        return points[-1]
    target = min(max(s, 0.0), 1.0) * total
    travelled = 0.0
    for i, seg_len in enumerate(seg_lengths):
        if travelled + seg_len >= target or i == len(seg_lengths) - 1:
            local = 0.0 if seg_len < 1.0e-9 else (target - travelled) / seg_len
            local = min(max(local, 0.0), 1.0)
            return tuple(points[i][k] + (points[i + 1][k] - points[i][k]) * local for k in range(3))
        travelled += seg_len
    return points[-1]


class FrankaWaypointController:
    """Drives a standalone Franka ``Articulation`` through world-frame waypoints."""

    def __init__(
        self,
        robot: Articulation,
        sim: SimulationContext,
        simulation_app,
        gripper_open: float = 0.04,
        gripper_closed: float = 0.01,
        settle_steps: int = 30,
        tip_offset_z: float = TIP_TO_HAND_Z,
        planner=None,
    ):
        self.robot = robot
        self.sim = sim
        self.simulation_app = simulation_app
        self.gripper_open = gripper_open
        self.gripper_closed = gripper_closed
        self.settle_steps = settle_steps
        self.tip_offset_z = tip_offset_z
        # Optional external planner (e.g. RMPFlow). When set, follow() drives the
        # arm with it instead of differential IK; the gripper schedule is unchanged.
        self.planner = planner
        self.device = robot.device
        self.num_envs = robot.num_instances

        # Resolve the arm / gripper joints and the end-effector body directly
        # from the articulation (no SceneEntityCfg, which needs a scene).
        self.arm_joint_ids, self._arm_names = robot.find_joints("panda_joint.*")
        self.gripper_joint_ids, self._gripper_names = robot.find_joints("panda_finger_joint.*")
        self.ee_body_id = robot.find_bodies("panda_hand")[0][0]

        # Map the planner's joint-output order to our arm joint order (by name).
        self._planner_cols = None
        if planner is not None:
            self._planner_cols = [planner.dof_names.index(name) for name in self._arm_names]

        # For a fixed-base robot the Jacobian index is one less than the body
        # index, because the root body is not part of the returned Jacobians.
        self.ee_jacobi_idx = self.ee_body_id - 1 if robot.is_fixed_base else self.ee_body_id

        self.controller = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            num_envs=self.num_envs,
            device=self.device,
        )

        self._current_grip = gripper_open
        self._hold_arm_target = robot.data.joint_pos[:, self.arm_joint_ids].clone()
        self._nan_warned = False

    # ------------------------------------------------------------------
    # Low-level helpers.
    # ------------------------------------------------------------------
    def _step(self):
        self.robot.write_data_to_sim()
        self.sim.step()
        self.robot.update(self.sim.get_physics_dt())
        # If the physics state goes NaN, the whole articulation (incl. the arm)
        # is corrupted - usually an unstable collider on some object. Warn once.
        if not self._nan_warned and torch.isnan(self.robot.data.joint_pos).any():
            print("[ERROR]: NaN in robot joint state -> PhysX is unstable (a bad/penetrating "
                  "collider, e.g. an object). The arm will not move correctly. Check object physics.")
            self._nan_warned = True

    def _grip_width(self, state: str) -> float:
        return self.gripper_open if state == "open" else self.gripper_closed

    def _set_gripper(self, width: float):
        target = torch.full(
            (self.num_envs, len(self.gripper_joint_ids)), width, dtype=torch.float32, device=self.device
        )
        self.robot.set_joint_position_target(target, joint_ids=self.gripper_joint_ids)
        self._current_grip = width

    def _ee_pose_b(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pose_w = self.robot.data.body_pose_w[:, self.ee_body_id]
        root_pose_w = self.robot.data.root_pose_w
        return subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )

    def _hand_target(self, tip_w, quat_w=EE_QUAT_DOWN) -> tuple[float, float, float]:
        """Lift a gripper-tip world point to the panda_hand target it implies.

        The tip sits ``tip_offset_z`` along the tool's local +z (approach axis) from
        the hand, so the hand target is the tip minus that vector rotated into world.
        For a top-down gripper (EE_QUAT_DOWN) this reduces to a +z shift."""
        from scene_context import _qapply

        off = _qapply(tuple(float(v) for v in quat_w), (0.0, 0.0, self.tip_offset_z))
        return (float(tip_w[0]) - off[0], float(tip_w[1]) - off[1], float(tip_w[2]) - off[2])

    def _world_to_base(self, pos_w, quat_w) -> tuple[torch.Tensor, torch.Tensor]:
        pos_w_t = torch.tensor([pos_w], dtype=torch.float32, device=self.device).repeat(self.num_envs, 1)
        quat_w_t = torch.tensor([quat_w], dtype=torch.float32, device=self.device).repeat(self.num_envs, 1)
        root_pose_w = self.robot.data.root_pose_w
        return subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], pos_w_t, quat_w_t
        )

    def _slerp_batch(self, q0: torch.Tensor, q1: torch.Tensor, t: float) -> torch.Tensor:
        return torch.stack([quat_slerp(a, b, t) for a, b in zip(q0, q1)])

    # ------------------------------------------------------------------
    # Public API.
    # ------------------------------------------------------------------
    def apply_gripper_friction(self, static_friction: float = 4.0, dynamic_friction: float = 3.2):
        """Bind a high-friction physics material to the fingertip colliders.

        Needed for a stable friction grip on a rigid object; call once after the
        robot has been spawned (the finger prims must exist on the stage).
        """
        material_path = "/World/PhysicsMaterials/GripperHighFriction"
        material_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=static_friction,
            dynamic_friction=dynamic_friction,
            restitution=0.0,
            friction_combine_mode="max",
            restitution_combine_mode="multiply",
        )
        material_cfg.func(material_path, material_cfg)

        stage = sim_utils.get_current_stage()
        bound = 0
        for prim in stage.Traverse():
            prim_path = str(prim.GetPath())
            is_finger = "panda_leftfinger" in prim_path or "panda_rightfinger" in prim_path
            # The CollisionAPI often lives on a child mesh, so also match the
            # finger link by name / a "collision" path and let the binding (which
            # is applied nested) propagate down to the actual collider prims.
            looks_like_collision = (
                prim.HasAPI(UsdPhysics.CollisionAPI)
                or "collision" in prim_path.lower()
                or prim.GetName() in {"panda_leftfinger", "panda_rightfinger"}
            )
            if is_finger and looks_like_collision:
                sim_utils.bind_physics_material(prim_path, material_path)
                bound += 1
        print(f"[INFO]: Applied high-friction gripper material to {bound} fingertip prims.")

    def reset_to_home(self):
        """Snap the arm to its default posture, settle physics, reset the IK."""
        robot = self.robot
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = torch.zeros_like(joint_pos)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        robot.set_joint_position_target(joint_pos)
        robot.reset()
        self._set_gripper(self.gripper_open)
        for _ in range(self.settle_steps):
            self._step()
        self.controller.reset()
        if self.planner is not None:
            self.planner.reset()
        self._hold_arm_target = robot.data.joint_pos[:, self.arm_joint_ids].clone()

    def _apply_planner_targets(self, target_w, quat_w):
        """One RMPFlow step toward (target_w, quat_w); writes arm joint targets."""
        self.planner.set_target(target_w, quat_w)
        joint_pos = self.planner.compute_joint_targets()  # (1, n_dof) in planner order
        arm_target = joint_pos[:, self._planner_cols].to(self.device)
        self.robot.set_joint_position_target(arm_target, joint_ids=self.arm_joint_ids)
        self._hold_arm_target = arm_target.clone()

    def _ee_pos_w(self):
        return self.robot.data.body_pose_w[:, self.ee_body_id][0, 0:3].tolist()

    def _follow_with_planner(self, waypoints: list[Waypoint]):
        """Drive the arm through waypoints with the external planner (RMPFlow)."""
        current_grip = self._current_grip
        for wp in waypoints:
            targets = wp.via_points_w if wp.via_points_w else [wp.pos_w]
            target_grip = self._grip_width(wp.gripper)
            steps_each = max(wp.duration_steps // len(targets), 1)
            start_ee = self._ee_pos_w()
            print(
                f"[INFO]: [RMPFlow] '{wp.label}': target={tuple(round(v, 3) for v in targets[-1])} "
                f"ee_now={tuple(round(v, 3) for v in start_ee)} ({len(targets)} target(s), {steps_each} steps each)"
            )
            for tgt in targets:
                for phase_step in range(steps_each):
                    if not self.simulation_app.is_running():
                        return
                    t = _smoothstep((phase_step + 1) / steps_each)
                    self._apply_planner_targets(tgt, wp.quat_w)
                    self._set_gripper(current_grip + (target_grip - current_grip) * t)
                    self._step()
            end_ee = self._ee_pos_w()
            moved = sum((end_ee[i] - start_ee[i]) ** 2 for i in range(3)) ** 0.5
            print(f"[INFO]: [RMPFlow] '{wp.label}' done: ee_now={tuple(round(v, 3) for v in end_ee)} (moved {moved:.3f} m)")
            current_grip = target_grip
        print("[INFO]: [RMPFlow] motion complete.")

    def _execute_arm_trajectory(self, positions, traj_names, current_grip, target_grip, duration_steps):
        """Play back a planned joint trajectory, stretched over ``duration_steps``
        physics steps so the arm tracks it smoothly (cuRobo's raw plan is short -
        firing one point per step would make the arm lurch). We resample the
        trajectory by linear interpolation between its rows."""
        cols = [traj_names.index(name) for name in self._arm_names]
        traj = positions[:, cols].to(self.device)  # (horizon, 7)
        horizon = traj.shape[0]
        steps = max(int(duration_steps), horizon)
        for step_index in range(steps):
            if not self.simulation_app.is_running():
                return False
            # Position along the planned trajectory for this physics step (eased).
            u = _smoothstep(step_index / max(steps - 1, 1)) * (horizon - 1)
            i0 = int(u)
            i1 = min(i0 + 1, horizon - 1)
            alpha = u - i0
            q = (traj[i0] * (1.0 - alpha) + traj[i1] * alpha).unsqueeze(0)
            self.robot.set_joint_position_target(q, joint_ids=self.arm_joint_ids)
            self._hold_arm_target = q.clone()
            frac = (step_index + 1) / steps
            self._set_gripper(current_grip + (target_grip - current_grip) * frac)
            self._step()
        return True

    def _follow_with_batch_planner(self, waypoints: list[Waypoint]):
        """Two-phase execution for the cuRobo planner.

        Phase A (approach + grasp): differential IK straight through the waypoints up to
        and including the grasp. This follows the report's recorded side-approach to the
        object (e.g. wp0 -> wp1, coming in along the handle), which clears protruding
        parts like the wand's ring - unlike a free planner path, which descends through
        the ring on its way down to the grasp.

        Phase B (carry): cuRobo plans the remaining waypoints collision-free around the
        cuboid, starting from the ACTUAL post-grasp configuration, so the carried object
        is moved without the arm driving into the base. Planning from the real current
        joints (not a chained plan) avoids any jump between the two phases.

        A carry segment cuRobo cannot plan falls back to straight differential IK.
        """
        grasp_dwell = next((i for i, wp in enumerate(waypoints) if wp.gripper == "closed"), None)
        approach = waypoints if grasp_dwell is None else waypoints[: grasp_dwell + 1]
        carry = [] if grasp_dwell is None else waypoints[grasp_dwell + 1:]

        # Phase A: deterministic safe approach + grasp via differential IK.
        current_pos_b, current_quat_b = self._ee_pose_b()
        current_grip = self._current_grip
        for wp in approach:
            print(f"[INFO]: [approach/diffik] '{wp.label}' -> world xyz={tuple(round(v, 3) for v in wp.pos_w)}")
            result = self._diffik_segment(wp, current_pos_b.clone(), current_quat_b.clone(), current_grip)
            if result is None:
                return
            current_pos_b, current_quat_b, current_grip = result

        if not carry:
            print("[INFO]: [cuRobo] no carry phase; motion complete.")
            return

        # Phase B: plan the carry with cuRobo from the ACTUAL current configuration.
        start_positions = self.robot.data.joint_pos[:, self.arm_joint_ids][0].detach().cpu().tolist()
        start_names = list(self._arm_names)
        # Waypoints are gripper-TIP targets; cuRobo solves for panda_hand, so lift each
        # goal by the tool offset (matching the diff-IK path).
        goals = [(self._hand_target(wp.pos_w, wp.quat_w), wp.quat_w) for wp in carry]
        print(f"[INFO]: [cuRobo] planning {len(goals)} carry waypoint(s) around the cuboid...")
        segments = self.planner.plan_all(start_positions, start_names, goals)
        for wp, segment in zip(carry, segments):
            target_grip = self._grip_width(wp.gripper)
            if segment is None:
                print(f"[WARN]: [cuRobo] no plan for '{wp.label}'; falling back to differential IK.")
                start_pos_b, start_quat_b = self._ee_pose_b()
                result = self._diffik_segment(wp, start_pos_b, start_quat_b, current_grip)
                if result is None:
                    return
                current_grip = result[2]
                continue
            positions, traj_names = segment
            print(f"[INFO]: [cuRobo] '{wp.label}': executing {positions.shape[0]}-point trajectory.")
            if not self._execute_arm_trajectory(positions, traj_names, current_grip, target_grip, wp.duration_steps):
                return
            current_grip = target_grip
        print("[INFO]: [cuRobo] carry complete.")

    def _diffik_segment(self, wp: Waypoint, start_pos_b, start_quat_b, current_grip: float):
        """Drive one waypoint with differential IK (shared by the default path and
        the cuRobo fallback). Returns (target_pos_b, target_quat_b, target_grip), or
        None if the app stopped mid-segment."""
        robot = self.robot
        # Waypoints are gripper-tip targets; command the panda_hand above them.
        target_pos_b, target_quat_b = self._world_to_base(self._hand_target(wp.pos_w, wp.quat_w), wp.quat_w)
        target_grip = self._grip_width(wp.gripper)
        duration = max(wp.duration_steps, 1)

        # For a continuous path, sweep through the current EE pose + via points as
        # one polyline (computed in world, converted to base each step). The current
        # hand pose needs no offset; the via points are tips, so do.
        polyline = None
        if wp.via_points_w:
            ee_w = robot.data.body_pose_w[:, self.ee_body_id][0, 0:3].tolist()
            polyline = [tuple(float(c) for c in ee_w)] + [self._hand_target(p, wp.quat_w) for p in wp.via_points_w]

        for phase_step in range(duration):
            if not self.simulation_app.is_running():
                return None
            t = _smoothstep(phase_step / max(duration - 1, 1))
            if polyline is not None:
                command_pos, _ = self._world_to_base(_point_along_polyline(polyline, t), wp.quat_w)
            else:
                command_pos = start_pos_b + (target_pos_b - start_pos_b) * t
            command_quat = self._slerp_batch(start_quat_b, target_quat_b, t)
            self.controller.set_command(torch.cat((command_pos, command_quat), dim=-1))

            jacobian = robot.root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, self.arm_joint_ids]
            ee_pos_b, ee_quat_b = self._ee_pose_b()
            joint_pos = robot.data.joint_pos[:, self.arm_joint_ids]
            joint_pos_des = self.controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
            self._hold_arm_target = joint_pos_des.clone()

            robot.set_joint_position_target(joint_pos_des, joint_ids=self.arm_joint_ids)
            self._set_gripper(current_grip + (target_grip - current_grip) * t)
            self._step()
        return target_pos_b, target_quat_b, target_grip

    def follow(self, waypoints: list[Waypoint]):
        """Move the end-effector through ``waypoints`` (world frame), in order."""
        if self.planner is not None:
            if hasattr(self.planner, "plan_all"):
                self._follow_with_batch_planner(waypoints)  # cuRobo (out-of-process)
            else:
                self._follow_with_planner(waypoints)  # reactive (RMPFlow)
            return
        current_pos_b, current_quat_b = self._ee_pose_b()
        current_grip = self._current_grip

        for wp in waypoints:
            print(f"[INFO]: Arm waypoint '{wp.label}' -> world xyz={tuple(round(v, 3) for v in wp.pos_w)}")
            result = self._diffik_segment(wp, current_pos_b.clone(), current_quat_b.clone(), current_grip)
            if result is None:
                return
            current_pos_b, current_quat_b, current_grip = result
            print(f"[INFO]: Arm waypoint '{wp.label}' complete.")

    def hold(self):
        """Keep the last commanded arm/gripper target until the app closes."""
        while self.simulation_app.is_running():
            self.robot.set_joint_position_target(self._hold_arm_target, joint_ids=self.arm_joint_ids)
            self._set_gripper(self._current_grip)
            self._step()
