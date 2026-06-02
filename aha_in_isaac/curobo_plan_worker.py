"""
curobo_plan_worker.py

Standalone cuRobo planning worker. Runs in its OWN Python process (NO Isaac Sim),
so cuRobo's required warp-lang does not clash with the older warp that Isaac Sim
loads. Communicates with the Isaac-side client via JSON files.

    python curobo_plan_worker.py <request.json> <response.json>

Request JSON (all poses already in the robot BASE frame):
    {
      "robot_file": "franka.yml",
      "obstacles": { "obs_0": {"dims":[dx,dy,dz], "pose":[x,y,z,qw,qx,qy,qz]}, ... },
      "start_positions": [...],            # current joint angles
      "start_joint_names": [...],          # names for the above
      "goals": [[x,y,z,qw,qx,qy,qz], ...], # one EE pose per waypoint (base frame)
      "max_attempts": 5
    }

Response JSON:
    {
      "segments": [
        {"success": true, "positions": [[...dof], ...], "joint_names": [...]},
        {"success": false, "positions": [], "joint_names": []},
        ...
      ]
    }

Each segment chains from the previous one's final configuration.
"""

from __future__ import annotations

import json
import sys

import torch

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import GoalToolPose, JointState, Pose


def _status_str(result) -> str:
    """Best-effort human-readable reason a plan failed (fields vary by cuRobo build)."""
    bits = []
    for attr in ("status", "valid_query"):
        if hasattr(result, attr):
            bits.append(f"{attr}={getattr(result, attr)}")
    return ", ".join(bits) if bits else "no status fields"


def _drop_approach_obstacles(planner, obstacles, meshes, approach_only):
    """Remove the approach-only obstacles (e.g. the now-grasped ring) from the planning
    world, so the grasp and carry are planned against the permanent obstacles only.
    Uses update_world (no re-warmup); the cache was pre-allocated for the full scene."""
    from curobo._src.geom.types import SceneCfg

    carry_meshes = {k: v for k, v in meshes.items() if k not in approach_only}
    carry_scene = {}
    if obstacles:
        carry_scene["cuboid"] = obstacles
    if carry_meshes:
        carry_scene["mesh"] = carry_meshes
    try:
        planner.update_world(SceneCfg.create(carry_scene))
        print(f"[worker]: dropped approach-only obstacles {sorted(approach_only)} for grasp+carry; "
              f"world now meshes={list(carry_meshes)} cuboids={list(obstacles)}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[worker]: update_world failed ({exc!r}); keeping all obstacles", flush=True)


def main(request_path: str, response_path: str):
    import os

    request = json.loads(open(request_path, encoding="utf-8").read())
    obstacles = request.get("obstacles", {})        # cuboids: {name: {dims, pose}}
    meshes = request.get("meshes", {})              # meshes:  {name: {vertices, faces, pose}}
    # Approach-only obstacles (e.g. the wand ring): present until the grasp, then dropped
    # via update_world so the carried object's start pose doesn't block the carry.
    approach_only = set(request.get("approach_only_mesh_names", []))
    grasp_goal_index = request.get("grasp_goal_index")
    if os.environ.get("AHA_CUROBO_NO_OBSTACLES"):
        print("[worker]: AHA_CUROBO_NO_OBSTACLES set -> ignoring obstacles for this run.", flush=True)
        obstacles, meshes, approach_only = {}, {}, set()
    max_attempts = int(request.get("max_attempts", 5))
    # Safety knobs from the client. collision_activation_distance is the clearance the
    # trajopt keeps from obstacles; graph_attempt=0 routes the plan with the sampling-based
    # PRM graph planner from the first attempt (collision-free through free space) instead of
    # the corner-cutting trajopt-only first attempt (graph_attempt=1, the old behaviour).
    collision_activation_distance = float(request.get("collision_activation_distance", 0.01))
    graph_attempt = int(request.get("graph_attempt", 1))

    # Mesh obstacles give an accurate concave shape (e.g. a thin rod) instead of a
    # bounding box that would swallow a nearby goal, so the arm can plan around it.
    scene = {}
    if obstacles:
        scene["cuboid"] = obstacles
    if meshes:
        scene["mesh"] = meshes

    config = MotionPlannerCfg.create(
        robot=request.get("robot_file", "franka.yml"),
        scene_model=scene if scene else None,
        optimizer_collision_activation_distance=collision_activation_distance,
    )
    planner = MotionPlanner(config)
    print("[worker]: warming up cuRobo...", flush=True)
    planner.warmup(enable_graph=True, num_warmup_iterations=5)
    device = config.device_cfg.device
    tool_frame = planner.tool_frames[0]
    # Planning cspace is the 7 arm joints (fingers are locked, not in trajopt).
    cspace_names = list(planner.joint_names)
    print(f"[worker]: tool_frame={tool_frame} cspace={cspace_names} "
          f"cuboids={list(obstacles)} meshes={list(meshes)} "
          f"graph(PRM)={'on' if graph_attempt == 0 else 'off'} "
          f"collision_activation_distance={collision_activation_distance} m", flush=True)

    start = JointState.from_position(
        torch.tensor([request["start_positions"]], dtype=torch.float32, device=device),
        joint_names=list(request["start_joint_names"]),
    )

    segments = []
    for index, goal_list in enumerate(request["goals"]):
        # At the grasp goal, drop the approach-only obstacles (the ring is now carried).
        if approach_only and grasp_goal_index is not None and index == grasp_goal_index:
            _drop_approach_obstacles(planner, obstacles, meshes, approach_only)
        gl = [float(v) for v in goal_list]
        print(f"[worker]: goal {index} pose(base) pos={gl[:3]} quat(wxyz)={gl[3:]}", flush=True)
        goal_pose = Pose.from_list(gl, config.device_cfg)
        goal = GoalToolPose.from_poses({tool_frame: goal_pose}, num_goalset=1)
        result = planner.plan_pose(
            goal, start, max_attempts=max_attempts, enable_graph_attempt=graph_attempt
        )
        # If the PRM roadmap could not connect start->goal, fall back to pure trajopt
        # (still clearance-aware) so we don't regress to "no plan" (which would hand the
        # segment to the uncollision-checked diff-IK fallback on the Isaac side).
        if graph_attempt == 0 and (result is None or not bool(result.success.any())):
            fallback = planner.plan_pose(
                goal, start, max_attempts=max_attempts, enable_graph_attempt=max_attempts + 1
            )
            if fallback is not None and bool(fallback.success.any()):
                print(f"[worker]: goal {index} PRM could not connect; used pure-trajopt fallback", flush=True)
                result = fallback

        if result is None or not bool(result.success.any()):
            reason = "result is None" if result is None else _status_str(result)
            print(f"[worker]: goal {index} FAILED ({reason})", flush=True)
            segments.append({"success": False, "positions": [], "joint_names": []})
            continue

        interp = result.get_interpolated_plan()
        position = interp.position
        while position.dim() > 2:
            position = position[0]
        names = list(interp.joint_names)
        segments.append({"success": True, "positions": position.detach().cpu().tolist(), "joint_names": names})
        print(f"[worker]: goal {index} OK ({position.shape[0]} pts)", flush=True)

        # Chain: next segment starts from this one's final config, restricted to the
        # planning cspace (the 7 arm joints), since the start state must be cspace-sized.
        cols = [names.index(n) for n in cspace_names]
        last_cspace = position[-1][cols]
        start = JointState.from_position(last_cspace.unsqueeze(0).to(device), joint_names=cspace_names)

    with open(response_path, "w", encoding="utf-8") as handle:
        json.dump({"segments": segments}, handle)
    print(f"[worker]: wrote {len(segments)} segment(s) -> {response_path}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
