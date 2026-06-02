"""
arm_motion.py

Turn the scene-context waypoints into the ordered list of ``Waypoint`` goals the
controller follows.

The robot simply FOLLOWS the report's waypoints in order, at their exact world
positions (the controller adds the fixed gripper-tip->hand offset). The motion
config only says how many sim steps to spend reaching each named waypoint; the
gripper grasps automatically at the lowest waypoint and releases at the last, so
there are no hard-coded heights, hovers, presses, or lifts.

Imports ``Waypoint`` from ``robot_controller`` (which pulls in Isaac Lab), so it
must only be imported after ``AppLauncher`` has started the simulator.
"""

from __future__ import annotations

from robot_controller import EE_QUAT_DOWN, Waypoint

# Built-in fallbacks (sim steps). DEFAULT_WAYPOINT_STEPS is used for any waypoint
# the JSON does not list; the grasp/release dwells are how long the arm holds in
# place while the fingers close on / open off the object.
DEFAULT_WAYPOINT_STEPS = 220
GRASP_DWELL_STEPS = 140
RELEASE_DWELL_STEPS = 120
# A multi-point sweep covers a long path, so by default it gets this many steps
# PER via point. This keeps the commanded EE speed slow enough for the IK arm to
# actually track the path (a single flat default would race through it and skip).
STEPS_PER_SWEEP_SAMPLE = 30
# How many intermediate points to sample on each curved (Catmull-Rom) segment.
CURVE_SAMPLES_PER_SEGMENT = 14


def _waypoint_world_pos(waypoint: dict) -> tuple[float, float, float]:
    return tuple(float(v) for v in waypoint["world_location"]["position_xyz_m"])


def _catmull_rom_segment(p0, p1, p2, p3, samples: int, alpha: float = 0.5):
    """Centripetal Catmull-Rom spline points along the segment p1->p2 (excluding
    p1, including p2). Uses the neighbouring points p0/p3 as tangents so the whole
    waypoint chain becomes one smooth, curved path. ``alpha=0.5`` (centripetal)
    avoids the loops/overshoot that uniform Catmull-Rom can produce."""

    def _t(ti, a, b):
        d = sum((b[k] - a[k]) ** 2 for k in range(3)) ** 0.5
        return ti + max(d, 1.0e-6) ** alpha

    t0 = 0.0
    t1 = _t(t0, p0, p1)
    t2 = _t(t1, p1, p2)
    t3 = _t(t2, p2, p3)
    # Degenerate spacing -> fall back to a straight line p1->p2.
    if t2 - t1 < 1.0e-6:
        return [tuple(p2)]

    def _lerp(a, b, ta, tb, t):
        w = 0.0 if tb - ta < 1.0e-9 else (t - ta) / (tb - ta)
        return tuple(a[k] + (b[k] - a[k]) * w for k in range(3))

    points = []
    for s in range(1, samples + 1):
        t = t1 + (t2 - t1) * (s / samples)
        a1 = _lerp(p0, p1, t0, t1, t)
        a2 = _lerp(p1, p2, t1, t2, t)
        a3 = _lerp(p2, p3, t2, t3, t)
        b1 = _lerp(a1, a2, t0, t2, t)
        b2 = _lerp(a2, a3, t1, t3, t)
        points.append(_lerp(b1, b2, t1, t2, t))
    return points


def _waypoint_world_quat(waypoint: dict, force_down: bool) -> tuple[float, float, float, float]:
    """The waypoint's recorded world orientation as Isaac (w, x, y, z).

    The report stores quaternions in scalar-last ``xyzw`` (RLBench/CoppeliaSim);
    Isaac uses scalar-first ``wxyz``. AHA plans to this exact orientation, so we
    follow it by default. ``force_down`` overrides it with a top-down gripper for
    simple grasps (the old behaviour)."""
    if force_down:
        return EE_QUAT_DOWN
    q = waypoint.get("world_location", {}).get("quaternion_xyzw")
    if not q or len(q) != 4:
        return EE_QUAT_DOWN
    qx, qy, qz, qw = (float(v) for v in q)
    return (qw, qx, qy, qz)


def _waypoint_steps(motion_config: dict, name: str, default: int = DEFAULT_WAYPOINT_STEPS) -> int:
    return int(motion_config.get("waypoint_steps", {}).get(name, default))


def _smooth_polyline(
    control_points: list[tuple[float, float, float]], samples_per_segment: int = 10
) -> list[tuple[float, float, float]]:
    """A dense, smooth (centripetal Catmull-Rom) curve through all ``control_points``.

    Chains :func:`_catmull_rom_segment` over each consecutive pair (duplicating the
    end points as tangents), so the result is one continuous curve the controller can
    sweep as via points - the arm traces an arc, not straight segments. The first
    control point is the start (the controller prepends the live EE), so it is not
    repeated in the output."""
    pts = list(control_points)
    if len(pts) <= 2:
        return pts[1:] if len(pts) == 2 else pts
    curve: list[tuple[float, float, float]] = []
    for i in range(len(pts) - 1):
        p0 = pts[i - 1] if i - 1 >= 0 else pts[i]
        p1, p2 = pts[i], pts[i + 1]
        p3 = pts[i + 2] if i + 2 < len(pts) else pts[i + 1]
        curve.extend(_catmull_rom_segment(p0, p1, p2, p3, samples_per_segment))
    return curve


def _grasp_index(
    waypoints: list[dict], positions: list[tuple[float, float, float]], graspable_name: str | None
) -> int:
    """Index of the waypoint where the gripper should close on the object.

    The grasp waypoint is the one whose pose is expressed RELATIVE TO the graspable
    object with the smallest offset - i.e. the waypoint sitting ON the object itself,
    not a pre-grasp approach (larger offset to the same object) nor a later carry pose
    (expressed relative to a different object). For beat_the_buzz this is waypoint1
    (offset 0.016 m from `wand`), not the lowest-z waypoint0 (offset 0.088 m). Falls
    back to the lowest-z waypoint when the graspable object can't be identified."""
    if graspable_name:
        best = None
        for i, w in enumerate(waypoints):
            rel = (w.get("relative_to") or {}).get("reference_name")
            if rel != graspable_name:
                continue
            off = w.get("fixed_offset_xyz_m") or (0.0, 0.0, 0.0)
            mag = sum(float(v) ** 2 for v in off) ** 0.5
            if best is None or mag < best[1]:
                best = (i, mag)
        if best is not None:
            return best[0]
    return min(range(len(positions)), key=lambda i: positions[i][2])


def _build_lift_over_motion(
    positions: list[tuple[float, float, float]],
    quats: list[tuple[float, float, float, float]],
    names: list[str],
    motion_config: dict,
    carry_lift: float,
    grasp_index: int,
) -> list[Waypoint]:
    """A deterministic collision-free carry that lifts the grasped object OVER a tall
    obstacle instead of dragging it through.

    Beat-the-buzz is the motivating case: the recorded carry waypoints keep the hand
    at arch height the whole way, so the ring (rigidly mounted above the hand) sweeps
    straight through the wire. Here we instead: approach from straight above, grasp,
    lift vertically by ``carry_lift`` so the ring clears the arch top, traverse the
    remaining waypoints at that raised height (holding the grasp orientation so the
    object does not tilt back down into the obstacle), then release - still clear.

    No external planner: plain diff-IK tracks these straight segments reliably and the
    eased motion keeps the friction grasp.
    """
    gx, gy, gz = positions[grasp_index]
    grasp_quat = quats[grasp_index]
    hover = (gx, gy, gz + carry_lift)

    def _raise(p: tuple[float, float, float]) -> tuple[float, float, float]:
        return (p[0], p[1], p[2] + carry_lift)

    motion: list[Waypoint] = []
    # Any waypoints before the grasp are open-gripper approach moves; keep them.
    for i in range(grasp_index):
        motion.append(
            Waypoint(names[i], positions[i], quat_w=quats[i], gripper="open",
                     duration_steps=_waypoint_steps(motion_config, names[i]))
        )
    # Come down onto the grasp from straight above (no sideways swing into the arch).
    motion.append(
        Waypoint(f"Hover above {names[grasp_index]}", hover, quat_w=grasp_quat, gripper="open",
                 duration_steps=_waypoint_steps(motion_config, names[grasp_index]))
    )
    motion.append(
        Waypoint(f"Descend to {names[grasp_index]}", (gx, gy, gz), quat_w=grasp_quat, gripper="open",
                 duration_steps=GRASP_DWELL_STEPS)
    )
    motion.append(
        Waypoint(f"Grasp at {names[grasp_index]}", (gx, gy, gz), quat_w=grasp_quat, gripper="closed",
                 duration_steps=GRASP_DWELL_STEPS)
    )
    # Curved carry: ONE smooth Catmull-Rom arc that rises off the grasp, up OVER the
    # obstacle, and across to the (raised) remaining waypoints - swept as a single
    # continuous via-point motion so the arm traces a curve, not straight segments.
    # ``hover`` (straight above the grasp) is the first arc control point, so the arc
    # climbs vertically clear of the arch before any horizontal travel over it.
    control = [(gx, gy, gz), hover]
    control += [_raise(positions[i]) for i in range(grasp_index + 1, len(positions))]
    arc = _smooth_polyline(control, samples_per_segment=10) or [hover]
    motion.append(
        Waypoint("Carry over rod (curved)", arc[-1], quat_w=grasp_quat, gripper="closed",
                 duration_steps=max(len(arc) * STEPS_PER_SWEEP_SAMPLE, DEFAULT_WAYPOINT_STEPS),
                 via_points_w=arc)
    )
    # Release at the final position, still raised clear of the obstacle.
    motion.append(
        Waypoint("Release", _raise(positions[-1]), quat_w=grasp_quat, gripper="open",
                 duration_steps=RELEASE_DWELL_STEPS)
    )
    return motion


def build_arm_motion(
    waypoints: list[dict], motion_config: dict, force_down: bool = False, curvy: bool = True,
    carry_lift: float = 0.0, graspable_name: str | None = None,
) -> list[Waypoint]:
    if not waypoints:
        return []

    positions = [_waypoint_world_pos(w) for w in waypoints]
    quats = [_waypoint_world_quat(w, force_down) for w in waypoints]
    names = [w.get("name", f"waypoint{i}") for i, w in enumerate(waypoints)]

    # The gripper closes at the waypoint sitting ON the graspable object (smallest
    # offset relative to it), not just the lowest one - so it grasps the wand at
    # waypoint1 rather than closing on air at the pre-grasp waypoint0.
    grasp_index = _grasp_index(waypoints, positions, graspable_name)

    # Lift-over carry: a planner-free, collision-free path that raises the grasped
    # object clear of a tall obstacle (e.g. the buzz-wire arch) during transit.
    if carry_lift and carry_lift > 0.0:
        return _build_lift_over_motion(positions, quats, names, motion_config, carry_lift, grasp_index)

    def _curve_to(index: int) -> list[tuple[float, float, float]]:
        """Curved via points from the previous waypoint to ``index`` (the controller
        prepends the live EE pose, so we omit the start point)."""
        p1, p2 = positions[index - 1], positions[index]
        p0 = positions[index - 2] if index - 2 >= 0 else p1
        p3 = positions[index + 1] if index + 1 < len(positions) else p2
        return _catmull_rom_segment(p0, p1, p2, p3, CURVE_SAMPLES_PER_SEGMENT)

    motion: list[Waypoint] = []
    grip = "open"
    for index, waypoint in enumerate(waypoints):
        name = waypoint.get("name", f"waypoint{index}")
        pos = positions[index]
        quat = quats[index]

        path_samples = waypoint.get("cartesian_path_samples") or []
        if path_samples:
            # Sweep the whole predefined cartesian path as ONE continuous, eased
            # motion (no stop-and-go at each sample), to keep the stroke smooth.
            via_points = [tuple(float(c) for c in s["position_xyz_m"]) for s in path_samples]
            sweep_default = max(len(via_points) * STEPS_PER_SWEEP_SAMPLE, DEFAULT_WAYPOINT_STEPS)
            motion.append(
                Waypoint(
                    f"{name} sweep ({len(via_points)} pts)",
                    via_points[-1],
                    quat_w=quat,
                    gripper=grip,
                    duration_steps=_waypoint_steps(motion_config, name, sweep_default),
                    via_points_w=via_points,
                )
            )
        elif curvy and index > 0:
            # Curve smoothly from the previous waypoint through this one.
            via_points = _curve_to(index)
            motion.append(
                Waypoint(
                    name,
                    pos,
                    quat_w=quat,
                    gripper=grip,
                    duration_steps=_waypoint_steps(motion_config, name),
                    via_points_w=via_points,
                )
            )
        else:
            motion.append(
                Waypoint(name, pos, quat_w=quat, gripper=grip, duration_steps=_waypoint_steps(motion_config, name))
            )

        # Close the gripper in place once we have descended onto the grasp waypoint.
        if index == grasp_index:
            motion.append(
                Waypoint(f"Grasp at {name}", pos, quat_w=quat, gripper="closed", duration_steps=GRASP_DWELL_STEPS)
            )
            grip = "closed"

    # Release the object in place at the final waypoint (no lift, just follow).
    motion.append(
        Waypoint("Release", positions[-1], quat_w=quats[-1], gripper="open", duration_steps=RELEASE_DWELL_STEPS)
    )
    return motion
