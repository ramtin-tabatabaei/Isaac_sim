"""
usd_uv.py

Shared helper for authoring a 'st' UV primvar on a mesh that lacks one, so an
image texture can map onto it. Used both offline by ``add_physics_to_usds.py``
(baking object UVs) and at runtime by ``scene_builder.py`` (e.g. the table).

Imports ``pxr`` at module load, so import it only after the simulator/USD
libraries are initialised.
"""

from __future__ import annotations

import math

from pxr import Gf, Sdf, UsdGeom, Vt


def detect_uv_shape(points, centroid, extents: list[float]) -> str:
    """Guess the best UV projection from the mesh geometry: 'planar' for a flat
    object, 'spherical' for a ball-like one, otherwise 'box'."""
    small, _, large = sorted(extents)
    if large < 1.0e-9:
        return "box"
    # Much thinner in one axis than the others -> a flat-ish object.
    if small < 0.18 * large:
        return "planar"
    # Ball test: roughly equal extents, points nearly equidistant from the center
    # (low radius variation), AND that radius matches the bounding sphere - the last
    # check rejects a cube, whose corners are equidistant but ~1.7x the half-extent.
    cx, cy, cz = centroid
    radii = [math.sqrt((p[0] - cx) ** 2 + (p[1] - cy) ** 2 + (p[2] - cz) ** 2) for p in points]
    mean = sum(radii) / len(radii)
    half_extent = large / 2.0
    if mean > 1.0e-9 and half_extent > 1.0e-9:
        cv = (sum((r - mean) ** 2 for r in radii) / len(radii)) ** 0.5 / mean
        radius_ratio = mean / half_extent
        if cv < 0.15 and 0.85 <= radius_ratio <= 1.15 and small > 0.6 * large:
            return "spherical"
    return "box"


def has_uvs(prim) -> bool:
    """True if the mesh prim already carries a 'st' (or 'uv') texture-coord primvar."""
    api = UsdGeom.PrimvarsAPI(prim)
    return api.HasPrimvar("st") or api.HasPrimvar("uv")


def generate_uvs(prim, mode: str) -> str | None:
    """Author a 'st' UV primvar on ``prim``. ``mode`` is 'auto' (detect from the
    shape), 'spherical', 'box'/'planar', or 'none'. Returns the projection used."""
    mode = (mode or "none").lower()
    if mode == "none":
        return None
    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get()
    if not points:
        return None

    n = len(points)
    centroid = (sum(p[0] for p in points) / n, sum(p[1] for p in points) / n, sum(p[2] for p in points) / n)
    mins = [min(p[k] for p in points) for k in range(3)]
    maxs = [max(p[k] for p in points) for k in range(3)]
    extents = [maxs[k] - mins[k] for k in range(3)]

    if mode == "auto":
        mode = detect_uv_shape(points, centroid, extents)

    def _frac(value, axis):
        span = maxs[axis] - mins[axis]
        return 0.5 if span < 1.0e-9 else (value - mins[axis]) / span

    st = []
    if mode == "spherical":
        cx, cy, cz = centroid
        for p in points:
            x, y, z = p[0] - cx, p[1] - cy, p[2] - cz
            r = math.sqrt(x * x + y * y + z * z) or 1.0
            u = 0.5 + math.atan2(y, x) / (2.0 * math.pi)
            v = 0.5 + math.asin(max(-1.0, min(1.0, z / r))) / math.pi
            st.append(Gf.Vec2f(float(u), float(v)))
    else:  # box / planar -> project onto the two largest-extent axes
        axis_a, axis_b = sorted(range(3), key=lambda k: extents[k], reverse=True)[:2]
        for p in points:
            st.append(Gf.Vec2f(float(_frac(p[axis_a], axis_a)), float(_frac(p[axis_b], axis_b))))

    primvar = UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    primvar.Set(Vt.Vec2fArray(st))
    return mode
