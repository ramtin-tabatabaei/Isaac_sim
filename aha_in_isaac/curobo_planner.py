"""
curobo_planner.py

Isaac-side CLIENT for cuRobo collision-free planning. cuRobo's warp-lang cannot
coexist in the same process as Isaac Sim's older warp, so the actual planning runs
in a SEPARATE process (``curobo_plan_worker.py``). This client:

  * builds the obstacle set (bounding boxes of the scene's static objects), in the
    robot base frame, using Isaac's USD;
  * transforms the world-frame waypoint goals into the base frame;
  * writes a request, runs the worker subprocess (a plain Python with cuRobo, no
    Isaac), and reads back the planned joint trajectories.

The controller then executes those trajectories. Set the worker's Python via the
``AHA_CUROBO_PYTHON`` env var if ``sys.executable`` is not the cuRobo env.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import torch

import isaaclab.sim as sim_utils
from pxr import Gf, Usd, UsdGeom

from scene_context import _qapply, _qinv, _qmul

_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "curobo_plan_worker.py")


class CuroboPlanner:
    # The controller builds its joint mapping from this; the real per-segment order
    # comes back with each plan, so this only needs the arm joints in cspace order.
    dof_names = [f"panda_joint{i}" for i in range(1, 8)]

    # A goal inside (or within this margin of) an obstacle's bounding box can never
    # be planned to - the box swallows the target. Such an obstacle is one the arm
    # intentionally reaches into (place into a hoop, thread a loop), so we drop it
    # from the planning scene and let the diff-IK fallback do that final reach.
    GOAL_MARGIN_M = 0.05

    def __init__(self, robot_base_pos, robot_base_quat, device: str, obstacle_prim_paths,
                 robot_file: str = "franka.yml", obstacle_mode: str = "mesh",
                 approach_only_prim_paths=None, safety_margin: float = 0.01, use_graph: bool = False):
        self.base_pos = tuple(float(v) for v in robot_base_pos)
        self.base_quat = tuple(float(v) for v in robot_base_quat)  # (w, x, y, z)
        self.robot_file = robot_file
        self.obstacle_mode = obstacle_mode  # "mesh" (accurate) or "bbox" (bounding boxes)
        # Safety knobs (see cli.py): clearance kept from obstacles, and whether to route
        # the path with cuRobo's sampling-based PRM graph planner (collision-free through
        # free space) instead of pure local trajectory optimization (which hugs obstacles).
        self.safety_margin = float(safety_margin)
        self.use_graph = bool(use_graph)
        self.python = os.environ.get("AHA_CUROBO_PYTHON", sys.executable)
        self._obstacles = self._build_obstacles(obstacle_prim_paths)
        # Permanent obstacles (kept for the whole motion) and "approach-only" obstacles
        # (the grasped object's protruding parts, e.g. the wand ring) that the worker
        # drops once the gripper grasps, since they are then carried with the arm.
        self._meshes = self._build_mesh_obstacles(obstacle_prim_paths, "obs_mesh") if obstacle_mode == "mesh" else {}
        self._approach_meshes = (
            self._build_mesh_obstacles(approach_only_prim_paths or [], "approach_mesh")
            if obstacle_mode == "mesh" else {}
        )
        print(f"[INFO]: cuRobo (out-of-process) ready. obstacle_mode={obstacle_mode} "
              f"graph_planner(PRM)={'on' if self.use_graph else 'off'} safety_margin={self.safety_margin} m "
              f"approach_only={list(self._approach_meshes)} worker python={self.python}")

    # ------------------------------------------------------------------
    def _to_base(self, pos_w, quat_w):
        rel = tuple(pos_w[i] - self.base_pos[i] for i in range(3))
        return _qapply(_qinv(self.base_quat), rel), _qmul(_qinv(self.base_quat), quat_w)

    def _build_obstacles(self, prim_paths) -> dict:
        stage = sim_utils.get_current_stage()
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], useExtentsHint=True
        )
        obstacles: dict = {}
        for index, path in enumerate(prim_paths):
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                continue
            box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
            lo, hi = box.GetMin(), box.GetMax()
            dims = [max(float(hi[i] - lo[i]), 0.01) for i in range(3)]
            center_w = [float((hi[i] + lo[i]) / 2.0) for i in range(3)]
            center_b, _ = self._to_base(center_w, (1.0, 0.0, 0.0, 0.0))
            obstacles[f"obs_{index}"] = {"dims": dims, "pose": [*center_b, 1.0, 0.0, 0.0, 0.0]}
        print(f"[INFO]: cuRobo scene has {len(obstacles)} obstacle cuboid(s).")
        return obstacles

    def _extract_world_mesh(self, prim_path):
        """Collect (vertices_world, faces) for every Mesh under ``prim_path``, with
        each prim's world transform baked into the vertices and faces triangulated."""
        stage = sim_utils.get_current_stage()
        root = stage.GetPrimAtPath(prim_path)
        verts: list = []
        faces: list = []
        if not root or not root.IsValid():
            return verts, faces
        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get()
            counts = mesh.GetFaceVertexCountsAttr().Get()
            indices = mesh.GetFaceVertexIndicesAttr().Get()
            if not points or not counts or not indices:
                continue
            xform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            base = len(verts)
            for p in points:
                wp = xform.Transform(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
                verts.append([wp[0], wp[1], wp[2]])
            offset = 0
            for count in counts:
                face = [int(indices[offset + k]) + base for k in range(count)]
                for k in range(1, count - 1):  # fan-triangulate polygons
                    faces.append([face[0], face[k], face[k + 1]])
                offset += count
        return verts, faces

    def _build_mesh_obstacles(self, prim_paths, prefix: str = "mesh") -> dict:
        """Each obstacle's real triangle mesh, vertices expressed in the robot base
        frame (pose left at identity). Concave geometry (a thin rod) is preserved, so
        a goal beside it is reachable - unlike a bounding box that swallows the goal.
        ``prefix`` names the entries so the worker can tell permanent from approach-only."""
        meshes: dict = {}
        for index, path in enumerate(prim_paths):
            verts_w, faces = self._extract_world_mesh(path)
            if not verts_w or not faces:
                continue
            verts_b = [self._to_base(v, (1.0, 0.0, 0.0, 0.0))[0] for v in verts_w]
            meshes[f"{prefix}_{index}"] = {
                "vertices": verts_b,
                "faces": faces,
                "pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            }
            print(f"[INFO]: cuRobo mesh obstacle '{path}' -> {len(verts_b)} verts, {len(faces)} tris ({prefix}).")
        return meshes

    def reset(self):
        pass

    def _box_encloses(self, obs, point) -> bool:
        """True if ``point`` (base frame) is inside ``obs``'s axis-aligned box + margin."""
        cx, cy, cz = obs["pose"][0], obs["pose"][1], obs["pose"][2]
        dx, dy, dz = obs["dims"]
        m = self.GOAL_MARGIN_M
        return (
            abs(point[0] - cx) <= dx / 2.0 + m
            and abs(point[1] - cy) <= dy / 2.0 + m
            and abs(point[2] - cz) <= dz / 2.0 + m
        )

    def _active_obstacles(self, goals_base) -> dict:
        """Drop obstacles whose bounding box encloses a goal (unreachable as a hard
        obstacle); keep the rest as the planning scene."""
        active, dropped = {}, []
        for name, obs in self._obstacles.items():
            if any(self._box_encloses(obs, g[:3]) for g in goals_base):
                dropped.append(name)
            else:
                active[name] = obs
        if dropped:
            print(f"[INFO]: cuRobo dropping {len(dropped)} obstacle(s) enclosing a goal "
                  f"(diff-IK will do that reach): {dropped}")
        return active

    def plan_all(self, start_positions, start_joint_names, goals_world, grasp_goal_index=None):
        """Plan a collision-free trajectory to each world goal pose, in sequence.

        ``grasp_goal_index`` is the goal at which the gripper grasps; the worker drops
        the approach-only obstacles (the carried ring) from that goal onward.

        Returns a list (one per goal) of (position_tensor (horizon, dof), joint_names),
        or None for goals that could not be planned.
        """
        goals_base = []
        for pos_w, quat_w in goals_world:
            pos_b, quat_b = self._to_base(tuple(float(v) for v in pos_w), tuple(float(v) for v in quat_w))
            goals_base.append([*pos_b, *quat_b])

        request = {
            "robot_file": self.robot_file,
            "start_positions": list(start_positions),
            "start_joint_names": list(start_joint_names),
            "goals": goals_base,
            "max_attempts": 5,
            # Clearance the trajopt keeps from obstacles (cuRobo's collision activation distance).
            "collision_activation_distance": self.safety_margin,
            # graph_attempt=0 -> route every plan with the PRM roadmap from the first attempt
            # (safer); 1 -> trajopt-only first, PRM only as a retry (the old behaviour).
            "graph_attempt": 0 if self.use_graph else 1,
        }
        # Mesh mode keeps the obstacle's true (thin) shape so a goal beside it stays
        # reachable; bbox mode drops any box that swallows a goal (the old behaviour).
        if self.obstacle_mode == "mesh" and (self._meshes or self._approach_meshes):
            request["meshes"] = {**self._meshes, **self._approach_meshes}
            # Approach-only obstacles (the ring) are present until the grasp, then the
            # worker removes them via update_world (they are carried from then on).
            if self._approach_meshes and grasp_goal_index is not None:
                request["approach_only_mesh_names"] = list(self._approach_meshes.keys())
                request["grasp_goal_index"] = int(grasp_goal_index)
        else:
            request["obstacles"] = self._active_obstacles(goals_base)

        with tempfile.TemporaryDirectory() as tmp:
            req_path = os.path.join(tmp, "request.json")
            resp_path = os.path.join(tmp, "response.json")
            with open(req_path, "w", encoding="utf-8") as handle:
                json.dump(request, handle)

            print(f"[INFO]: cuRobo planning {len(goals_base)} goal(s) in a worker process...")
            proc = subprocess.run([self.python, _WORKER, req_path, resp_path], capture_output=True, text=True)
            if proc.stdout:
                print(proc.stdout.rstrip())
            if proc.returncode != 0 or not os.path.exists(resp_path):
                print(f"[ERROR]: cuRobo worker failed (code {proc.returncode}).\n{proc.stderr.rstrip()}")
                return [None] * len(goals_base)
            response = json.loads(open(resp_path, encoding="utf-8").read())

        results = []
        for segment in response["segments"]:
            if not segment.get("success"):
                results.append(None)
                continue
            positions = torch.tensor(segment["positions"], dtype=torch.float32)
            results.append((positions, segment["joint_names"]))
        return results
