"""
lula_plan_worker.py

Standalone **RRT-Connect** planning worker. Runs in its OWN Python process so the
``lula`` pybind module (bundled inside the Isaac Sim extensions, not on the default
path) can be imported with the right PYTHONPATH / LD_LIBRARY_PATH set by the client,
without disturbing the Isaac Sim process. Communicates via JSON files.

    python lula_plan_worker.py <request.json> <response.json>

This implements the SAME algorithm AHA/RLBench uses (OMPL's ``RRTConnect``): a
bidirectional RRT that grows a tree from the start config and another from the goal
config and connects them. OMPL is not installed here and Lula only ships a single-tree
``JtRRT``, so the planner itself is implemented here; Lula is used only for the pieces
that need the robot model:

  * inverse kinematics for the goal EE pose (``lula.compute_ik_ccd``),
  * robot-vs-world collision checking (``lula.RobotWorldInspector``), whose robot
    collision spheres come from the descriptor (densified at the fingers, see
    ``_densified_robot_descriptor``),
  * joint limits + the c-space coordinate order (``lula.Kinematics``).

Request JSON (world frame; the worker transforms to the robot base frame):
    {
      "base_pos": [x, y, z],
      "base_quat": [w, x, y, z],
      "start_joints": [j1..j7],         # current arm joints
      "joint_names": ["panda_joint1", ...],   # order of start_joints / returned path
      "ee_frame": "panda_hand",         # frame the goal pose is expressed for
      "target_pos_w": [x, y, z],        # world EE target
      "target_quat_w": [w, x, y, z] | null,   # world EE orientation (null -> ignore)
      "obstacle_spheres_w": [{"center": [x, y, z], "radius": r}, ...],
      "max_iterations": 80000,
      "seeds": [12345, 7, 999]
    }

Response JSON:
    {"success": bool, "positions": [[j1..j7], ...], "joint_names": [...]}
The positions are the dense (interpolated) c-space path in ``joint_names`` order.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

# Pure-Python quaternion helpers shared with the Isaac side (no Isaac imports).
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
from scene_context import _qinv, _qmul, _qapply  # noqa: E402

# Bundled Franka assets that ship with the motion_generation extension. Override
# the base dir with AHA_LULA_MG if the Isaac install lives elsewhere.
MG_DEFAULT = (
    "/home/ramtin/miniconda3/envs/env_isaacsim51/lib/python3.11/site-packages/"
    "isaacsim/exts/isaacsim.robot_motion.motion_generation"
)
MG = os.environ.get("AHA_LULA_MG", MG_DEFAULT)
ROBOT_DESC = os.path.join(MG, "motion_policy_configs/franka/rmpflow/robot_descriptor.yaml")
URDF = os.path.join(MG, "motion_policy_configs/franka/lula_franka_gen.urdf")

# RRT-Connect parameters. STEP is the c-space extension distance (rad); RES is the
# edge-checking resolution (rad). ESCAPE lets the tree leave a start/goal config that
# the conservative collision spheres flag as touching (e.g. the hand by the grasp):
# collision is ignored within ESCAPE rad of a tree root so the arm can back out.
STEP = 0.15
RES = 0.05
ESCAPE = 0.30
GOAL_BIAS = 0.1
SHORTCUT_ITERS = 200


def _to_base(base_pos, base_quat, pos_w):
    """World point -> robot base frame (the frame the planner works in)."""
    rel = tuple(float(pos_w[i]) - float(base_pos[i]) for i in range(3))
    return _qapply(_qinv(base_quat), rel)


def _quat_to_base(base_quat, quat_w):
    return _qmul(_qinv(base_quat), quat_w)


def _densified_robot_descriptor():
    """Path to a robot descriptor whose FINGER collision spheres are densified.

    The stock Franka descriptor covers each finger with too few/too-small spheres, so a
    plan can pass the discrete collision check while the real fingertip clips a thin
    obstacle. We add spheres along each finger (and enlarge them slightly) and write the
    patched descriptor to a temp file. Returns that path, or the stock path on failure."""
    import tempfile

    try:
        import yaml

        with open(ROBOT_DESC, encoding="utf-8") as handle:
            desc = yaml.safe_load(handle)
        # collision_spheres is a LIST of single-key dicts: [{link: [{center,radius}, ...]}, ...].
        # The stock Franka descriptor has NO finger entries (links 0-7 + hand only), so the
        # fingertips are uncovered and a plan can clip a thin obstacle with the open fingers.
        # Add a column of spheres down each finger (frames exist in the kinematics).
        cspheres = [e for e in desc.get("collision_spheres", [])
                    if not any(k in ("panda_leftfinger", "panda_rightfinger") for k in e)]
        for link in ("panda_leftfinger", "panda_rightfinger"):
            cspheres.append({link: [{"center": [0.0, 0.0, z], "radius": 0.012}
                                    for z in (0.005, 0.018, 0.031, 0.045)]})
        desc["collision_spheres"] = cspheres
        tmp = tempfile.NamedTemporaryFile("w", suffix="_robot_descriptor.yaml", delete=False, encoding="utf-8")
        yaml.safe_dump(desc, tmp)
        tmp.close()
        n = sum(len(list(e.values())[0]) for e in cspheres)
        print(f"[rrtc-worker]: densified finger spheres -> {tmp.name} (total {n} robot spheres)", flush=True)
        return tmp.name
    except Exception as exc:  # noqa: BLE001
        print(f"[rrtc-worker]: could not densify finger spheres ({exc!r}); using stock descriptor", flush=True)
        return ROBOT_DESC


# ----------------------------------------------------------------------
# RRT-Connect over the 7-DoF c-space.
# ----------------------------------------------------------------------
def _nearest(tree, q):
    diffs = tree - q  # (N,7)
    return int(np.argmin(np.einsum("ij,ij->i", diffs, diffs)))


def _steer(q_from, q_to, step):
    delta = q_to - q_from
    dist = float(np.linalg.norm(delta))
    if dist <= step:
        return q_to.copy()
    return q_from + delta * (step / dist)


def _edge_clear(q_a, q_b, clear, root_a_dist=None, root_b_dist=None):
    """All interior points of edge a->b collision-free, sampled every RES. Points within
    ESCAPE of a tree root (root_*_dist gives the cspace distance from that root to q_a/q_b
    end) are skipped so the planner can leave a root the conservative spheres flag."""
    seg = q_b - q_a
    dist = float(np.linalg.norm(seg))
    n = max(1, int(np.ceil(dist / RES)))
    for k in range(1, n + 1):
        t = k / n
        q = q_a + seg * t
        if root_a_dist is not None and (root_a_dist + t * dist) < ESCAPE:
            continue
        if root_b_dist is not None and (root_b_dist + (1.0 - t) * dist) < ESCAPE:
            continue
        if not clear(q):
            return False
    return True


def rrt_connect(start, goal, clear, sample, step, max_iter, rng):
    """Bidirectional RRT-Connect. Trees are (configs array, parent list, root-dist list).
    Returns the list of configs start..goal, or None. Roots are exempt from collision
    (added unconditionally) and get an ESCAPE zone so an in-contact start/goal can back
    out."""
    a_q = [start.copy()]; a_p = [-1]; a_r = [0.0]   # start-rooted
    b_q = [goal.copy()];  b_p = [-1]; b_r = [0.0]   # goal-rooted
    a_is_start = True

    def reconstruct(q_list, p_list, idx):
        chain = []
        while idx != -1:
            chain.append(q_list[idx])
            idx = p_list[idx]
        return chain  # leaf..root

    for _ in range(max_iter):
        q_rand = goal if (a_is_start and rng.random() < GOAL_BIAS) else sample()
        tree_a = np.asarray(a_q)
        ia = _nearest(tree_a, q_rand)
        q_new = _steer(tree_a[ia], q_rand, step)
        if not _edge_clear(tree_a[ia], q_new, clear, root_a_dist=a_r[ia]) or not (
            a_r[ia] + float(np.linalg.norm(q_new - tree_a[ia])) < ESCAPE or clear(q_new)
        ):
            a_q, a_p, a_r, b_q, b_p, b_r = b_q, b_p, b_r, a_q, a_p, a_r
            a_is_start = not a_is_start
            continue
        a_q.append(q_new); a_p.append(ia); a_r.append(a_r[ia] + float(np.linalg.norm(q_new - tree_a[ia])))
        ia_new = len(a_q) - 1

        # CONNECT the other tree greedily toward q_new.
        tree_b = np.asarray(b_q)
        ib = _nearest(tree_b, q_new)
        cur = ib
        cur_q = b_q[ib].copy()
        cur_r = b_r[ib]
        connected = False
        while True:
            q_step = _steer(cur_q, q_new, step)
            seg_len = float(np.linalg.norm(q_step - cur_q))
            if not _edge_clear(cur_q, q_step, clear, root_a_dist=cur_r) or not (
                cur_r + seg_len < ESCAPE or clear(q_step)
            ):
                break
            b_q.append(q_step); b_p.append(cur); b_r.append(cur_r + seg_len)
            cur = len(b_q) - 1
            cur_q = q_step
            cur_r = cur_r + seg_len
            if float(np.linalg.norm(q_step - q_new)) < 1.0e-3:
                connected = True
                break

        if connected:
            branch_a = reconstruct(a_q, a_p, ia_new)   # q_new..rootA
            branch_b = reconstruct(b_q, b_p, cur)       # q_step(~q_new)..rootB
            if a_is_start:
                path = branch_a[::-1] + branch_b        # start..q_new , q_step..goal
            else:
                path = branch_b[::-1] + branch_a        # start..q_step , q_new..goal
            return [np.asarray(q) for q in path]

        a_q, a_p, a_r, b_q, b_p, b_r = b_q, b_p, b_r, a_q, a_p, a_r
        a_is_start = not a_is_start
    return None


def _shortcut(path, clear, rng, iters):
    """Greedy shortcut smoothing: repeatedly try to replace a sub-path between two random
    indices with a straight c-space segment when that segment is collision-free."""
    if len(path) < 3:
        return path
    pts = list(path)
    for _ in range(iters):
        if len(pts) < 3:
            break
        i = rng.integers(0, len(pts) - 2)
        j = rng.integers(i + 2, len(pts))
        if _edge_clear(pts[i], pts[j], clear):
            pts = pts[: i + 1] + pts[j:]
    return pts


def _interpolate(path, res):
    """Densely resample the polyline path at <= ``res`` rad spacing for smooth tracking."""
    if len(path) < 2:
        return path
    out = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        seg = b - a
        dist = float(np.linalg.norm(seg))
        n = max(1, int(np.ceil(dist / res)))
        for k in range(1, n + 1):
            out.append(a + seg * (k / n))
    return out


def main(request_path: str, response_path: str):
    import lula

    req = json.loads(open(request_path, encoding="utf-8").read())
    base_pos = req["base_pos"]
    base_quat = req["base_quat"]  # (w, x, y, z)
    joint_names = list(req["joint_names"])
    start_joints = [float(v) for v in req["start_joints"]]
    ee_frame = req.get("ee_frame", "panda_hand")
    target_pos_b = _to_base(base_pos, base_quat, req["target_pos_w"])
    target_quat_w = req.get("target_quat_w")
    spheres_w = req.get("obstacle_spheres_w", [])
    max_iter = int(req.get("max_iterations", 80000))
    seeds = req.get("seeds", [12345, 7, 999])

    robot = lula.load_robot(_densified_robot_descriptor(), URDF)
    kin = robot.kinematics()
    cspace = [kin.c_space_coord_name(i) for i in range(kin.num_c_space_coords())]
    lower = np.array([kin.c_space_coord_limits(i).lower for i in range(len(cspace))])
    upper = np.array([kin.c_space_coord_limits(i).upper for i in range(len(cspace))])

    name_to_val = dict(zip(joint_names, start_joints))
    start = np.array([name_to_val.get(n, 0.0) for n in cspace], dtype=np.float64)

    # World + robot-vs-world inspector (collision checker).
    world = lula.create_world()
    for s in spheres_w:
        ob = lula.create_obstacle(lula.Obstacle.Type.SPHERE)
        ob.set_attribute(lula.Obstacle.Attribute.RADIUS, float(s["radius"]))
        c_b = _to_base(base_pos, base_quat, s["center"])
        world.add_obstacle(ob, lula.Pose3(lula.Rotation3(1.0, 0.0, 0.0, 0.0), np.array(c_b, dtype=np.float64)))
    inspector = lula.create_robot_world_inspector(robot, world.add_world_view())

    def clear(q):
        return not inspector.in_collision_with_obstacle(np.asarray(q, dtype=np.float64))

    # Goal config(s) via IK on the EE pose; keep the first collision-free solution.
    target_pos = np.array(target_pos_b, dtype=np.float64)
    if target_quat_w is not None:
        q_b = _quat_to_base(base_quat, target_quat_w)
        target_pose = lula.Pose3(lula.Rotation3(*(float(v) for v in q_b)), target_pos)
    else:
        target_pose = lula.Pose3(lula.Rotation3(1.0, 0.0, 0.0, 0.0), target_pos)

    print(f"[rrtc-worker]: cspace={cspace} ee_frame={ee_frame} spheres={len(spheres_w)} "
          f"target_base={[round(v, 3) for v in target_pos_b]}", flush=True)

    goal = None
    rng = np.random.default_rng(int(seeds[0]) if seeds else 0)
    for attempt in range(40):
        ik_cfg = lula.CyclicCoordDescentIkConfig()
        # A translation-only goal frees the orientation (large tolerance) so IK solves
        # for position alone; otherwise the recorded EE orientation is matched.
        if target_quat_w is None:
            ik_cfg.orientation_tolerance = 1.0e3
        seed_q = start if attempt == 0 else lower + rng.random(len(cspace)) * (upper - lower)
        ik_cfg.cspace_seeds = [seed_q]
        res = lula.compute_ik_ccd(kin, target_pose, ee_frame, ik_cfg)
        if not res.success:
            continue
        q_goal = np.asarray(res.cspace_position, dtype=np.float64).ravel()
        if np.any(q_goal < lower - 1e-6) or np.any(q_goal > upper + 1e-6):
            continue
        if clear(q_goal):
            goal = q_goal
            print(f"[rrtc-worker]: IK goal found (attempt {attempt}) q={[round(v,3) for v in goal]}", flush=True)
            break
    if goal is None:
        with open(response_path, "w", encoding="utf-8") as handle:
            json.dump({"success": False, "positions": [], "joint_names": []}, handle)
        print("[rrtc-worker]: NO collision-free IK goal -> NO PLAN", flush=True)
        return

    def sample():
        return lower + rng.random(len(cspace)) * (upper - lower)

    path = None
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        path = rrt_connect(start, goal, clear, sample, STEP, max_iter, rng)
        if path is not None:
            print(f"[rrtc-worker]: RRT-Connect solved with seed={seed} ({len(path)} waypoints)", flush=True)
            break
        print(f"[rrtc-worker]: seed={seed} failed", flush=True)

    if path is None:
        with open(response_path, "w", encoding="utf-8") as handle:
            json.dump({"success": False, "positions": [], "joint_names": []}, handle)
        print("[rrtc-worker]: NO PLAN FOUND", flush=True)
        return

    path = _shortcut(path, clear, np.random.default_rng(12345), SHORTCUT_ITERS)
    dense = _interpolate(path, RES)
    positions = np.asarray(dense)
    with open(response_path, "w", encoding="utf-8") as handle:
        json.dump({"success": True, "positions": positions.tolist(), "joint_names": cspace}, handle)
    print(f"[rrtc-worker]: wrote path with {positions.shape[0]} points -> {response_path}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
