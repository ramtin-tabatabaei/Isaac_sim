"""
rmpflow_planner.py

EXPERIMENTAL collision-avoiding arm driver using Isaac Lab's RMPFlow controller
(Lula). It drives the Franka end-effector toward each waypoint while avoiding the
scene's static objects, which are registered as (bounding-box) obstacles.

RMPFlow is a *reactive* policy, not a global planner: it avoids obstacles and
respects joint limits / self-collision, but can get stuck in local minima and
will NOT reliably thread a loop or insert a peg. Obstacles are approximated by
their world bounding box, so concave features (e.g. the hole of a ring) are
treated as solid.

Imports Isaac Lab / Isaac Sim at load, so import only after AppLauncher starts.
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.controllers.config.rmp_flow import FRANKA_RMPFLOW_CFG
from isaaclab.controllers.rmp_flow import RmpFlowController
from pxr import Usd, UsdGeom


class RmpFlowPlanner:
    """Thin wrapper around ``RmpFlowController`` for a single Franka arm."""

    def __init__(self, robot_prim_path: str, device: str):
        self.controller = RmpFlowController(FRANKA_RMPFLOW_CFG, device)
        self.controller.initialize(robot_prim_path)
        self.device = device
        self.dof_names = list(self.controller.active_dof_names)
        self._obstacle_count = 0
        print(f"[INFO]: RMPFlow ready (arm dofs: {self.dof_names}).")

    def add_box_obstacles(self, prim_paths: list[str]):
        """Register each prim's world bounding box as a static RMPFlow obstacle."""
        try:
            from isaacsim.core.api.objects import VisualCuboid
        except Exception as exc:  # pragma: no cover - depends on Isaac Sim build
            print(f"[WARN]: Could not import VisualCuboid; skipping obstacles ({exc}).")
            return

        stage = sim_utils.get_current_stage()
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], useExtentsHint=True
        )
        motion_policy = self.controller.articulation_policies[0].get_motion_policy()
        for path in prim_paths:
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                continue
            box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
            lo, hi = box.GetMin(), box.GetMax()
            size = [max(float(hi[i] - lo[i]), 0.01) for i in range(3)]
            center = [float((hi[i] + lo[i]) / 2.0) for i in range(3)]
            obs_path = f"/World/Obstacles/Obs_{self._obstacle_count}"
            try:
                cuboid = VisualCuboid(prim_path=obs_path, position=center, scale=size, visible=False)
                motion_policy.add_obstacle(cuboid)
                self._obstacle_count += 1
            except Exception as exc:  # keep going; one bad obstacle should not abort
                print(f"[WARN]: RMPFlow could not add obstacle for {path}: {exc}")
        print(f"[INFO]: RMPFlow registered {self._obstacle_count} obstacle(s).")

    def set_base_pose(self, position, orientation_wxyz):
        """Tell RMPFlow where the robot base is in the world (it plans in base frame,
        but we feed world-frame targets)."""
        import numpy as np

        motion_policy = self.controller.articulation_policies[0].get_motion_policy()
        try:
            motion_policy.set_robot_base_pose(
                robot_position=np.asarray(position, dtype=float),
                robot_orientation=np.asarray(orientation_wxyz, dtype=float),
            )
            print(f"[INFO]: RMPFlow base pose set to {tuple(round(float(v), 3) for v in position)}.")
        except Exception as exc:
            print(f"[WARN]: RMPFlow set_robot_base_pose failed: {exc}")

    def reset(self):
        self.controller.reset_idx()

    def set_target(self, pos_w, quat_w):
        command = torch.tensor([[*pos_w, *quat_w]], dtype=torch.float32, device=self.device)
        self.controller.set_command(command)

    def compute_joint_targets(self) -> torch.Tensor:
        """Return desired arm joint positions, shape (1, len(dof_names))."""
        joint_pos, _ = self.controller.compute()
        return joint_pos
