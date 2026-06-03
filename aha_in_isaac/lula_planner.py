"""
lula_planner.py

Isaac-side CLIENT for the Lula RRT planner. The ``lula`` pybind module is bundled
inside the Isaac Sim extensions (not on the default path) and its shared libs need
``LD_LIBRARY_PATH``, so - like the cuRobo client - the actual planning runs in a
SEPARATE process (``lula_plan_worker.py``) launched with the right environment.

This client only converts a single end-effector goal into a worker request and
returns the planned c-space trajectory. It is the "curve to that point" half of
the linear-first policy: the controller tries a straight line, and only when that
line collides does it ask this planner to route around the collision (modelled as
small spheres) using RRT.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import torch

_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lula_plan_worker.py")

# Where the bundled ``lula`` pybind + its shared libs live (override with env vars
# if the Isaac Sim install moves). The worker is launched with these on the path.
LULA_PREBUNDLE_DEFAULT = (
    "/home/ramtin/miniconda3/envs/env_isaacsim51/lib/python3.11/site-packages/"
    "isaacsim/exts/isaacsim.robot_motion.lula/pip_prebundle"
)


class LulaRrtPlanner:
    """Out-of-process Lula RRT client. ``plan_segment`` marks this as the
    linear-first/RRT planner the controller drives (vs cuRobo's ``plan_all``)."""

    dof_names = [f"panda_joint{i}" for i in range(1, 8)]

    def __init__(self, robot_base_pos, robot_base_quat, device: str, max_iterations: int = 80000):
        self.base_pos = [float(v) for v in robot_base_pos]
        self.base_quat = [float(v) for v in robot_base_quat]  # (w, x, y, z)
        self.device = device
        self.max_iterations = int(max_iterations)
        self.python = os.environ.get("AHA_LULA_PYTHON", sys.executable)
        self.prebundle = os.environ.get("AHA_LULA_PREBUNDLE", LULA_PREBUNDLE_DEFAULT)
        self.libs = os.path.join(self.prebundle, "_lula_libs")
        print(f"[INFO]: Lula RRT (out-of-process) ready. worker_python={self.python} "
              f"prebundle={self.prebundle} max_iterations={self.max_iterations}")

    def reset(self):
        pass

    def _worker_env(self) -> dict:
        env = dict(os.environ)
        env["PYTHONPATH"] = self.prebundle + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["LD_LIBRARY_PATH"] = self.libs + (os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        return env

    def plan_segment(
        self,
        start_joints,
        joint_names,
        target_pos_w,
        target_quat_w,
        sphere_centers_w,
        radius: float,
        ee_frame: str = "panda_hand",
    ):
        """Plan a collision-free c-space path to a single world EE goal, routing the
        robot around the given collision spheres. Returns ``(positions_tensor
        (horizon, 7), joint_names)`` or ``None`` if no plan was found."""
        request = {
            "base_pos": self.base_pos,
            "base_quat": self.base_quat,
            "start_joints": [float(v) for v in start_joints],
            "joint_names": list(joint_names),
            "ee_frame": ee_frame,
            "target_pos_w": [float(v) for v in target_pos_w],
            "target_quat_w": None if target_quat_w is None else [float(v) for v in target_quat_w],
            "obstacle_spheres_w": [
                {"center": [float(c) for c in center], "radius": float(radius)} for center in sphere_centers_w
            ],
            "max_iterations": self.max_iterations,
            "seeds": [12345, 7, 999],
        }
        with tempfile.TemporaryDirectory() as tmp:
            req_path = os.path.join(tmp, "request.json")
            resp_path = os.path.join(tmp, "response.json")
            with open(req_path, "w", encoding="utf-8") as handle:
                json.dump(request, handle)
            print(f"[INFO]: [Lula RRT] planning around {len(request['obstacle_spheres_w'])} sphere(s) "
                  f"in a worker process...")
            proc = subprocess.run(
                [self.python, _WORKER, req_path, resp_path], capture_output=True, text=True, env=self._worker_env()
            )
            if proc.stdout:
                print(proc.stdout.rstrip())
            if proc.returncode != 0 or not os.path.exists(resp_path):
                print(f"[ERROR]: Lula RRT worker failed (code {proc.returncode}).\n{proc.stderr.rstrip()}")
                return None
            response = json.loads(open(resp_path, encoding="utf-8").read())

        if not response.get("success"):
            return None
        positions = torch.tensor(response["positions"], dtype=torch.float32)
        return positions, response["joint_names"]
