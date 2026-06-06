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

import math

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


def _rpy_to_quat_wxyz(rpy) -> tuple[float, float, float, float]:
    """Convert a report ``orientation_rpy_rad`` [roll_x, pitch_y, yaw_z] to Isaac
    ``(w, x, y, z)``. Matches the report's own xyzw quaternion: q = qx(roll) * qy(pitch)
    * qz(yaw) (verified against waypoint world_location entries that carry both)."""

    def _qmul(a, b):  # a * b, both (x, y, z, w)
        ax, ay, az, aw = a
        bx, by, bz, bw = b
        return (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        )

    r, p, y = (float(v) for v in rpy)
    qx = (math.sin(r / 2.0), 0.0, 0.0, math.cos(r / 2.0))
    qy = (0.0, math.sin(p / 2.0), 0.0, math.cos(p / 2.0))
    qz = (0.0, 0.0, math.sin(y / 2.0), math.cos(y / 2.0))
    x, yy, z, w = _qmul(qx, _qmul(qy, qz))
    return (w, x, yy, z)


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
            # Offset to the graspable via the DIRECT relation (relative_to) AND/OR the
            # nearest-object relation. The waypoint sitting ON the object has the smallest
            # offset by either path. wipe_desk's grasp (waypoint1) is relative_to waypoint0,
            # not the sponge, so it's only found via its nearest-object relation - without
            # this we'd wrongly grasp at waypoint0 (42 mm ABOVE the sponge), missing it.
            mags = []
            if (w.get("relative_to") or {}).get("reference_name") == graspable_name:
                off = w.get("fixed_offset_xyz_m") or (0.0, 0.0, 0.0)
                mags.append(sum(float(v) ** 2 for v in off) ** 0.5)
            near = w.get("relative_to_nearest_object") or {}
            if near.get("reference_name") == graspable_name:
                pos = (near.get("location_in_reference_frame") or {}).get("position_xyz_m")
                if pos:
                    mags.append(sum(float(v) ** 2 for v in pos) ** 0.5)
            if not mags:
                continue
            mag = min(mags)
            if best is None or mag < best[1]:
                best = (i, mag)
        if best is not None:
            return best[0]
    return min(range(len(positions)), key=lambda i: positions[i][2])


def build_arm_motion(
    waypoints: list[dict], motion_config: dict, force_down: bool = False, curvy: bool = True,
    carry_lift: float = 0.0, graspable_name: str | None = None,
    slide_along_rod: float = 0.0, slide_axis_w: tuple[float, float, float] = (0.0, -1.0, 0.0),
) -> list[Waypoint]:
    if not waypoints:
        return []

    positions = [_waypoint_world_pos(w) for w in waypoints]
    # A predefined cartesian-path waypoint's recorded "position" is a nominal reference that does
    # NOT coincide with where its sweep actually ENDS, so use the last path sample as that
    # waypoint's effective position. The sweep itself already targets that sample; the fix is for
    # the NEXT segment, whose smooth curve is built from this waypoint's position -- using the stale
    # nominal point makes the arm detour out to it after the sweep (wipe_desk swung ~28 cm to
    # mid-table and back after the wipe) instead of continuing smoothly from where the sweep ended.
    for i, w in enumerate(waypoints):
        samples = w.get("cartesian_path_samples") or []
        if samples:
            positions[i] = tuple(float(c) for c in samples[-1]["position_xyz_m"])
    quats = [_waypoint_world_quat(w, force_down) for w in waypoints]
    names = [w.get("name", f"waypoint{i}") for i, w in enumerate(waypoints)]

    # The gripper closes at the waypoint sitting ON the graspable object (smallest
    # offset relative to it), not just the lowest one - so it grasps the wand at
    # waypoint1 rather than closing on air at the pre-grasp waypoint0.
    grasp_index = _grasp_index(waypoints, positions, graspable_name)

    def _curve_to(index: int) -> list[tuple[float, float, float]]:
        """Curved via points from the previous waypoint to ``index`` (the controller
        prepends the live EE pose, so we omit the start point)."""
        p1, p2 = positions[index - 1], positions[index]
        p0 = positions[index - 2] if index - 2 >= 0 else p1
        p3 = positions[index + 1] if index + 1 < len(positions) else p2
        return _catmull_rom_segment(p0, p1, p2, p3, CURVE_SAMPLES_PER_SEGMENT)

    motion: list[Waypoint] = []

    def _append_follow(index: int, grip: str) -> None:
        """Append one recorded waypoint at gripper ``grip``: its predefined cartesian
        sweep, a smooth curve from the previous waypoint, or a plain straight segment."""
        waypoint = waypoints[index]
        name = waypoint.get("name", f"waypoint{index}")
        pos = positions[index]
        quat = quats[index]
        path_samples = waypoint.get("cartesian_path_samples") or []
        if path_samples:
            # Sweep the whole predefined cartesian path as ONE continuous, eased motion.
            via_points = [tuple(float(c) for c in s["position_xyz_m"]) for s in path_samples]
            sweep_default = max(len(via_points) * STEPS_PER_SWEEP_SAMPLE, DEFAULT_WAYPOINT_STEPS)
            # When the task opts in (follow_path_orientation), reorient the gripper GRADUALLY
            # through the path's own per-sample orientations, so it tracks a path that turns the
            # wrist (e.g. close_box swinging the lid shut). Otherwise HOLD the entry orientation:
            # a surface drag (wipe_desk) must keep a fixed orientation, and slerping to the path
            # waypoint's recorded quaternion (often in a different frame) would flip the wrist
            # ~180 deg through a singularity and blow the IK arm up.
            via_quats = None
            if motion_config.get("follow_path_orientation") and not force_down and all(
                s.get("orientation_rpy_rad") for s in path_samples
            ):
                via_quats = [_rpy_to_quat_wxyz(s["orientation_rpy_rad"]) for s in path_samples]
                sweep_quat = via_quats[-1]  # final orientation, for the tracking-error report
            else:
                sweep_quat = quats[index - 1] if index > 0 else quat
            motion.append(
                Waypoint(f"{name} sweep ({len(via_points)} pts)", via_points[-1], quat_w=sweep_quat, gripper=grip,
                         duration_steps=_waypoint_steps(motion_config, name, sweep_default),
                         via_points_w=via_points, via_quats_w=via_quats)
            )
        elif curvy and index > 0:
            # Curve smoothly from the previous waypoint through this one.
            motion.append(
                Waypoint(name, pos, quat_w=quat, gripper=grip,
                         duration_steps=_waypoint_steps(motion_config, name), via_points_w=_curve_to(index))
            )
        else:
            motion.append(
                Waypoint(name, pos, quat_w=quat, gripper=grip, duration_steps=_waypoint_steps(motion_config, name))
            )

    # Push / press / close tasks grasp nothing: just follow every recorded waypoint in order
    # with a fixed gripper, so the arm traces the demonstrated motion exactly (e.g. close_box
    # swinging the lid shut) with no grasp dwell, carry lift, or release detour. Opt in with
    # "no_grasp": true; "gripper_state" (default "closed") sets the fixed finger pose.
    if motion_config.get("no_grasp"):
        grip = str(motion_config.get("gripper_state", "closed"))
        for index in range(len(waypoints)):
            _append_follow(index, grip)
        return motion

    # Approach + grasp: follow the recorded waypoints up to and including the grasp,
    # then close in place. This is the path proven to reach the object; a lift only
    # changes the CARRY (below), never the approach.
    for index in range(grasp_index + 1):
        _append_follow(index, "open")
    motion.append(
        Waypoint(f"Grasp at {names[grasp_index]}", positions[grasp_index], quat_w=quats[grasp_index],
                 gripper="closed", duration_steps=GRASP_DWELL_STEPS)
    )

    carry_positions = positions[grasp_index + 1:]
    release_pos = positions[-1]
    release_quat = quats[-1]
    released_during_carry = False
    if slide_along_rod and slide_along_rod > 0.0:
        # The ring is topologically CAPTIVE on the frame's rod, so the recorded sideways carry
        # (waypoint3) is impossible - it can only tunnel/eject the ring. Instead slide the grasp
        # pose along the rod axis by slide_along_rod (one slow straight diff-IK sweep, orientation
        # + grip fixed), so the ring stays threaded the whole time (the real beat-the-buzz motion).
        gx, gy, gz = positions[grasp_index]
        ax, ay, az = slide_axis_w
        end = (gx + ax * slide_along_rod, gy + ay * slide_along_rod, gz + az * slide_along_rod)
        n = 24
        via = [
            (gx + ax * slide_along_rod * t / n,
             gy + ay * slide_along_rod * t / n,
             gz + az * slide_along_rod * t / n)
            for t in range(1, n + 1)
        ]
        motion.append(
            Waypoint("Slide ring along rod", end, quat_w=quats[grasp_index], gripper="closed",
                     duration_steps=max(n * STEPS_PER_SWEEP_SAMPLE, DEFAULT_WAYPOINT_STEPS), via_points_w=via)
        )
        release_pos = end
        release_quat = quats[grasp_index]
    elif carry_lift and carry_lift > 0.0 and carry_positions:
        # Up-and-over carry: lift the grasped object straight UP clear of a tall
        # obstacle, traverse at that raised height over the remaining waypoints' XY,
        # then descend onto the final one - one smooth eased diff-IK sweep. The
        # recorded flat carry instead drags across at obstacle height (e.g. the
        # buzz-wire), clipping it; a pure vertical lift stays within the arm's reach.
        gx, gy, gz = positions[grasp_index]
        final = positions[-1]
        over_path = (
            [(gx, gy, gz + carry_lift)]
            + [(p[0], p[1], p[2] + carry_lift) for p in carry_positions]
            + [final]
        )
        motion.append(
            Waypoint("Lift over obstacle (curved)", over_path[-1], quat_w=quats[grasp_index], gripper="closed",
                     duration_steps=max(len(over_path) * STEPS_PER_SWEEP_SAMPLE * 4, DEFAULT_WAYPOINT_STEPS),
                     via_points_w=over_path)
        )
    else:
        # Flat carry: follow the remaining recorded waypoints in place. By default the gripper stays
        # CLOSED through all of them and opens at the separate Release step below (the last waypoint).
        # If the task sets 'release_at_waypoint', the gripper OPENS at THAT waypoint instead - placing
        # the object there - and any later waypoints are then traversed already open (e.g. open the
        # pepper onto the scale tray, then retreat up), so there is no trailing in-air release.
        release_name = motion_config.get("release_at_waypoint")
        release_index = names.index(release_name) if release_name in names else None
        if release_name and release_index is None:
            print(f"[WARN]: release_at_waypoint '{release_name}' is not a waypoint name; releasing at the end.")
        if release_index is not None and not (grasp_index < release_index <= len(waypoints) - 1):
            print(f"[WARN]: release_at_waypoint '{release_name}' must come after the grasp; releasing at the end.")
            release_index = None
        for index in range(grasp_index + 1, len(waypoints)):
            _append_follow(index, "open" if release_index is not None and index >= release_index else "closed")
        released_during_carry = release_index is not None
        if released_during_carry:
            print(f"[INFO]: weighing-style release: gripper opens at '{names[release_index]}', "
                  f"then retreats through the later waypoint(s) already open.")

    # Release the object at the final carry position - unless the flat carry already opened the
    # gripper at its configured release waypoint above.
    if not released_during_carry:
        motion.append(
            Waypoint("Release", release_pos, quat_w=release_quat, gripper="open", duration_steps=RELEASE_DWELL_STEPS)
        )
    return motion
