"""
scene_builder.py

Spawn the design scene into Isaac Sim: floor, dining table (visual + an invisible
top collider), the USD objects, the waypoint markers, and the Franka arm.

``SceneBuilder`` takes the parsed CLI ``args`` and a ``SceneContext`` and exposes
``design_scene()``, which spawns everything and returns the robot ``Articulation``
(or ``None`` when ``--no-robot``).

This imports Isaac Lab / USD at module load, so it must only be imported after
``AppLauncher`` has started the simulator.
"""

from __future__ import annotations

import math
import re
import statistics
from pathlib import Path

import isaaclab.sim as sim_utils
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

from robot_arm import spawn_franka
from scene_context import (
    SceneContext,
    _qapply,
    _qinv,
    _qmul,
    pose_from_location,
    pose_from_world_location,
    task_root_object,
)
from usd_uv import generate_uvs, has_uvs

# Local top of the bundled diningTable.usdc mesh (used to drop it onto our table top).
DINING_TABLE_LOCAL_TOP_Z = 0.750022
# Invisible flat collider placed at the table top so objects rest on a real
# surface without cooking the (expensive) dining-table mesh.
TABLE_COLLIDER_SIZE_XY = (2.0, 2.0)
TABLE_COLLIDER_THICKNESS = 0.04

WAYPOINT_COLORS = (
    (0.1, 0.35, 1.0),
    (0.0, 0.75, 0.35),
    (1.0, 0.72, 0.05),
    (1.0, 0.15, 0.1),
)

# In task-root mode the rigid transform places every object so its measured world
# centre lands within ~1.6 cm of its reported pose. An object left FURTHER than this
# from its report pose is one RLBench re-placed independently (a second jar, a
# distractor cup/block, the hockey ball) that is not tagged graspable; it is snapped
# onto its reported pose. The margin sits well below the smallest such displacement
# (>=10 cm in the data) so correctly-placed rigid objects are never disturbed.
SNAP_MOVER_THRESHOLD_M = 0.05

def _prim_name(name: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", name)
    return "".join(word[:1].upper() + word[1:] for word in words) or "Object"


# RLBench pairs a render mesh with a collision mesh. A render mesh's name contains
# "_visual"/"_vis" (possibly with more parts after it, e.g. wand_visual_sub or
# book0_visual_book0_side). Its physics body is the sibling sharing the base name
# (everything BEFORE "_vis"), after also stripping a BODY_SUFFIX. The body provides
# collision and is hidden; every render part is glued onto it so they move together.
BODY_SUFFIXES = ("_respondable", "_resp")


def _strip_suffix(name: str, suffixes: tuple[str, ...]) -> str:
    for suffix in suffixes:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _is_render(name: str) -> bool:
    return "_vis" in name.lower()


def _render_base(name: str) -> str:
    """Base (physics-body) name for a render mesh: everything before '_vis'."""
    lower = name.lower()
    index = lower.find("_vis")
    return name[:index] if index != -1 else name


def _quat_from_z_axis_to_vector(vector) -> tuple[float, float, float, float]:
    length = sum(component * component for component in vector) ** 0.5
    if length < 1.0e-9:
        return (1.0, 0.0, 0.0, 0.0)

    bx, by, bz = (component / length for component in vector)
    dot = bz
    if dot < -0.999999:
        return (0.0, 1.0, 0.0, 0.0)

    # Cross product from local +Z to target direction.
    q = (1.0 + dot, -by, bx, 0.0)
    norm = math.sqrt(sum(component * component for component in q))
    return tuple(component / norm for component in q)


def _path_sample_positions(waypoint: dict) -> list[tuple[float, float, float]]:
    samples = waypoint.get("cartesian_path_samples") or []
    positions = []
    for sample in samples:
        position = sample.get("position_xyz_m")
        if position is not None:
            positions.append(tuple(float(v) for v in position))
    return positions


class SceneBuilder:
    """Spawns the design scene described by a ``SceneContext``."""

    def __init__(self, args, ctx: SceneContext, appearance_config: dict | None = None):
        self.args = args
        self.ctx = ctx
        appearance_config = appearance_config or {}
        # Texture paths in the appearance config are relative to this package dir (where textures/
        # lives), independent of where the appearance config files themselves are.
        self._appearance_dir = Path(__file__).resolve().parent
        self._appearance_defaults = appearance_config.get("_defaults", {})
        self._appearance_scene = appearance_config.get("_scene", {})
        self._appearance_task = {
            k: v for k, v in appearance_config.get(ctx.task_name, {}).items() if not k.startswith("_")
        }
        # Populated during design_scene(); used to build planner obstacles.
        self.body_prim_paths: dict[str, str] = {}
        # Render-skin prim paths (e.g. the wand's ring, glued onto its body). Used to
        # add the grasped object's protruding parts as planner obstacles.
        self.skin_prim_paths: dict[str, str] = {}
        # Render-skin -> physics-body pairing, filled in _spawn_usd_objects().
        self._skin_to_body: dict[str, str] = {}
        self.table_prim_path: str | None = None

    def _scene_spec(self, key: str) -> dict | None:
        """Appearance for a non-object scene part ('table'/'floor'), or None if unset."""
        if key not in self._appearance_scene:
            return None
        spec = dict(self._appearance_defaults)
        spec.update(self._appearance_scene[key])
        return spec

    def _ensure_uvs(self, prim_path: str, mode: str = "auto"):
        """Generate UVs for any mesh under prim_path that lacks them (so a texture
        can map). Meshes that already have UVs are left untouched."""
        stage = sim_utils.get_current_stage()
        root = stage.GetPrimAtPath(prim_path)
        if not root.IsValid():
            return
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdGeom.Mesh) and not has_uvs(prim):
                generate_uvs(prim, mode)

    # ------------------------------------------------------------------
    # Appearance (visibility / texture / color), per object.
    # ------------------------------------------------------------------
    def _appearance_for(self, object_name: str) -> dict:
        """Merge the best-matching appearance entry over the defaults (longest key wins)."""
        name = object_name.lower()
        best_key, best_len = None, -1
        for key in self._appearance_task:
            token = key.lower()
            if (name == token or name.endswith(token)) and len(token) > best_len:
                best_key, best_len = key, len(token)
        spec = dict(self._appearance_defaults)
        if best_key is not None:
            spec.update(self._appearance_task[best_key])
        return spec

    def _resolve_texture(self, texture) -> str | None:
        if not texture:
            return None
        path = Path(texture)
        if not path.is_absolute():
            path = self._appearance_dir / texture
        if path.is_file():
            return str(path)
        print(f"[WARN]: Texture not found ({path}); using solid color instead.")
        return None

    def _author_appearance(self, prim_path: str, spec: dict):
        """Author a UsdPreviewSurface (texture or solid color) and bind it to the prim."""
        color = spec.get("color")
        texture = self._resolve_texture(spec.get("texture"))
        if not color and not texture:
            return
        stage = sim_utils.get_current_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return

        if texture:
            # A texture needs UVs; generate them where the mesh has none (baked
            # objects already have them, so this is a no-op for those).
            self._ensure_uvs(prim_path, "auto")

        material = UsdShade.Material.Define(stage, f"{prim_path}/AppearanceMaterial")
        shader = UsdShade.Shader.Define(stage, f"{prim_path}/AppearanceMaterial/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(spec.get("roughness", 0.8)))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(spec.get("metallic", 0.0)))
        diffuse = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
        rgb = Gf.Vec3f(*(float(c) for c in color)) if color else Gf.Vec3f(0.8, 0.8, 0.8)

        if texture:
            # st reader + UV texture; falls back to the solid color where the mesh
            # has no UVs (these exports usually don't), so it never looks broken.
            st_reader = UsdShade.Shader.Define(stage, f"{prim_path}/AppearanceMaterial/stReader")
            st_reader.CreateIdAttr("UsdPrimvarReader_float2")
            st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
            tex = UsdShade.Shader.Define(stage, f"{prim_path}/AppearanceMaterial/diffuseTexture")
            tex.CreateIdAttr("UsdUVTexture")
            tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture)
            tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_reader.ConnectableAPI(), "result")
            tex.CreateInput("fallback", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(rgb[0], rgb[1], rgb[2], 1.0))
            tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            diffuse.ConnectToSource(tex.ConnectableAPI(), "rgb")
        else:
            diffuse.Set(rgb)

        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI(prim).Bind(material)

    # ------------------------------------------------------------------
    # Low-level helpers.
    # ------------------------------------------------------------------
    def _ensure_xform(self, prim_path: str):
        stage = sim_utils.get_current_stage()
        if not stage.GetPrimAtPath(prim_path).IsValid():
            sim_utils.create_prim(prim_path, "Xform")

    def _usd_bbox_center(self, usd_path) -> tuple[float, float, float]:
        stage = Usd.Stage.Open(str(usd_path))
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], useExtentsHint=True
        )
        box = bbox_cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedBox()
        center = (box.GetMin() + box.GetMax()) * 0.5
        return tuple(float(v) for v in center)

    def _canonical_task_root_pos(self) -> tuple[float, float, float]:
        """Position of the task-root origin in the baked-USD frame. Every object is
        shifted by ``-canonical`` so the canonical task-root origin lands on the
        TaskRoot xform; the sampled task-root transform then re-places the whole scene.

        Derived as the consensus (component-wise median) of each object's *implied*
        task-root origin -- ``bbox_center(obj) - R_sampled^-1 . (report_pos(obj) -
        sampled_pos)`` -- rather than from the single task-root anchor's bbox centre.
        The lone anchor is unreliable: in some tasks its collision USD is baked in a
        different frame than the rest of the assembly (e.g. change_clock's ``clock``
        collider is baked lying flat and displaced), which uniformly shifts every other
        object by tens of centimetres. The median is robust to such a mis-baked anchor
        and to a minority of independently-placed objects. For a healthy task every
        object agrees, and the median provably equals the old single-anchor value
        (``impliedC(anchor) == bbox_center(anchor) - anchor_local_pos``), so this is a
        no-op there.
        """
        objects = self.ctx.objects
        sampled_pos = self.ctx.sampled_task_root_pos
        inv_quat = _qinv(self.ctx.sampled_task_root_quat)
        # Graspables (and what is mounted on them) sit at their own sampled pose, not
        # the rigid layout, so they must not vote on the canonical origin.
        movable = set(self.ctx.graspable_names) | {
            name for name, entry in objects.items() if entry.get("mounted_on_graspable")
        }
        implied: list[tuple[float, float, float]] = []
        for name, entry in objects.items():
            if name in movable or name not in self.ctx.usd_paths:
                continue
            report = (entry.get("world_location") or {}).get("position_xyz_m")
            if not report or len(report) != 3:
                continue
            center = self._usd_bbox_center(self.ctx.usd_paths[name])
            rel = _qapply(inv_quat, tuple(float(report[i]) - sampled_pos[i] for i in range(3)))
            implied.append(tuple(center[i] - rel[i] for i in range(3)))
        if not implied:
            # Fallback: original single-anchor estimate.
            root_entry = task_root_object(objects, self.ctx.task_name)
            root_center = self._usd_bbox_center(self.ctx.usd_paths[root_entry["name"]])
            root_local_pos, _ = pose_from_location(root_entry.get("task_root_local_location"))
            return tuple(root_center[i] - root_local_pos[i] for i in range(3))
        return tuple(statistics.median(c[i] for c in implied) for i in range(3))

    # ------------------------------------------------------------------
    # Scene pieces.
    # ------------------------------------------------------------------
    def _spawn_lights(self):
        dome_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9))
        dome_cfg.func("/World/Light", dome_cfg)

        distant_cfg = sim_utils.DistantLightCfg(intensity=1800.0, color=(0.95, 0.92, 0.86))
        distant_cfg.func("/World/KeyLight", distant_cfg, translation=(1.5, -1.2, 4.0))

    def _spawn_floor_and_table(self):
        floor_cfg = sim_utils.CuboidCfg(
            size=(20.0, 20.0, 0.02),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=False),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.08, 0.08, 0.08), roughness=0.9),
        )
        floor_cfg.func("/World/Floor", floor_cfg, translation=(0.0, 0.0, -0.01))
        floor_spec = self._scene_spec("floor")
        if floor_spec:
            self._author_appearance("/World/Floor", floor_spec)

        if self.args.no_table:
            return

        table_usd = self.ctx.table_usd
        if not table_usd.is_file():
            raise FileNotFoundError(f"Dining table USD file not found: {table_usd}")

        # Spawn the dining table as a *visual-only* prim. Cooking a triangle-mesh
        # collider for the full table mesh can hang PhysX, so instead we add a thin
        # invisible box collider at the table top (below) for objects to rest on.
        table_top_z = self.ctx.table_top_z
        table_cfg = sim_utils.UsdFileCfg(usd_path=str(table_usd))
        table_translation = (0.0, 0.0, table_top_z - DINING_TABLE_LOCAL_TOP_Z)
        table_path = "/World/DesignScene/DiningTable"
        table_cfg.func(table_path, table_cfg, translation=table_translation)
        self.table_prim_path = table_path
        table_spec = self._scene_spec("table")
        if table_spec:
            self._author_appearance(table_path, table_spec)

        top_cfg = sim_utils.CuboidCfg(
            size=(TABLE_COLLIDER_SIZE_XY[0], TABLE_COLLIDER_SIZE_XY[1], TABLE_COLLIDER_THICKNESS),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=False),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visible=False,
        )
        top_cfg.func(
            "/World/DesignScene/TableTopCollider",
            top_cfg,
            translation=(0.0, 0.0, table_top_z - TABLE_COLLIDER_THICKNESS / 2.0),
        )
        print(
            f"[INFO]: Placed dining table (visual) at translation="
            f"{tuple(round(v, 6) for v in table_translation)} with an invisible top collider at z={table_top_z:.4f}."
        )

    def _spawn_usd_objects(self):
        self._ensure_xform("/World/DesignScene")
        objects = self.ctx.objects
        usd_paths = self.ctx.usd_paths
        hidden_objects = set(self.args.hide_object)
        # Pair each render mesh (..._visual) with its physics body. We SHOW the render
        # mesh (it has the UVs/detail) and HIDE the body (collision only), gluing the
        # render onto it so it follows the body when grasped or simulated.
        body_base_to_obj: dict[str, str] = {}
        for name in objects:
            if not _is_render(name):
                body_base_to_obj.setdefault(_strip_suffix(name, BODY_SUFFIXES), name)
        # Pair each render skin to its physics body. The report's parent relationship is
        # authoritative (e.g. pepper_visual0's parent is pepper0). The name-based "render
        # base" pairing is only a fallback: it fails when an index trails the _visual token
        # (pepper_visual0 -> base "pepper", but the body is "pepper0"), which left the skin
        # unparented so it could not follow its body when the body moved or was grasped.
        def _nearest_body_ancestor(start: str | None) -> str | None:
            """Walk up the report parent chain from ``start`` to the nearest ancestor that is a
            real (non-render) body. This lets a render mesh nested UNDER another render mesh glue
            onto the moving body instead of being orphaned. close_grill's handle_visual has parent
            lid_visual (itself a _visual skin), so the immediate-parent check below misses it and
            it would spawn standalone - the visible handle then floats in place while the lid body
            swings, which reads as a disconnected handle / fake motion. Walking lid_visual -> lid
            glues the handle onto the lid so it rides the hinge with it."""
            cur = start
            seen: set[str] = set()
            while cur and cur in objects and cur not in seen:
                if not _is_render(cur):
                    return cur
                seen.add(cur)
                cur = objects[cur].get("parent")
            return None

        skin_to_body: dict[str, str] = {}
        for name, entry in objects.items():
            if not _is_render(name):
                continue
            parent = entry.get("parent")
            if parent in objects and not _is_render(parent):
                skin_to_body[name] = parent
            else:
                fallback = body_base_to_obj.get(_render_base(name))
                if fallback:
                    skin_to_body[name] = fallback
                else:
                    # Last resort: a render mesh nested under another render mesh (so the two checks
                    # above both miss) rides the nearest non-render ancestor body. Strictly additive -
                    # it only rescues skins that would otherwise be orphaned (spawn standalone and not
                    # follow their body), e.g. close_grill's handle_visual -> lid_visual -> lid.
                    ancestor = _nearest_body_ancestor(parent)
                    if ancestor is not None:
                        skin_to_body[name] = ancestor
        body_to_skin: dict[str, str] = {}
        for skin, body in skin_to_body.items():
            body_to_skin.setdefault(body, skin)
        self._skin_to_body = skin_to_body
        # Child shapes that should RIDE their parent body rather than spawn as a standalone
        # physics body (data-driven: physics-shapes "mount_on_parent": true). Used for the
        # change_channel button caps (wrapN): in CoppeliaSim they are rigidly mounted on the
        # pressable topPlateN, so gluing them under the plate makes them follow BOTH the press
        # (the plate's prismatic joint) and the remote when it is carried - instead of a
        # collider-less mesh floating where it spawned. Same gluing path as a render skin.
        phys_shapes = (self.ctx.report.get("physics") or {}).get("shapes", {}) or {}
        mounted_to_body: dict[str, str] = {}
        for name, entry in objects.items():
            if not (phys_shapes.get(name, {}) or {}).get("mount_on_parent"):
                continue
            mount_parent = entry.get("parent")
            if mount_parent in objects and not _is_render(mount_parent):
                mounted_to_body[name] = mount_parent
            else:
                print(f"[WARN]: '{name}' mount_on_parent set but parent '{mount_parent}' is not a "
                      "spawned body; it will spawn standalone.")
        self._mounted_to_body = mounted_to_body
        parent_path = "/World/DesignScene"
        task_root_child_translation = (0.0, 0.0, 0.0)
        task_root_child_orientation = (1.0, 0.0, 0.0, 0.0)

        if self.args.object_pose_mode == "task-root":
            parent_path = "/World/DesignScene/TaskRoot"
            canonical_root_pos = self._canonical_task_root_pos()
            sim_utils.create_prim(
                parent_path,
                "Xform",
                translation=self.ctx.sampled_task_root_pos,
                orientation=self.ctx.sampled_task_root_quat,
            )
            task_root_child_translation = tuple(-v for v in canonical_root_pos)
            print(
                "[INFO]: Task-root placement "
                f"sampled_pos={tuple(round(v, 6) for v in self.ctx.sampled_task_root_pos)} "
                f"sampled_quat_wxyz={tuple(round(v, 6) for v in self.ctx.sampled_task_root_quat)} "
                f"canonical_pos={tuple(round(v, 6) for v in canonical_root_pos)}"
            )

        # In task-root mode every object shares the same prim transform, so a
        # "<name>" collision body and its "<name>_visual" skin can be glued
        # together: keep the box body as the grasped physics (its render hidden)
        # and attach the detailed visual as a child so it rides along when picked.
        glue = self.args.object_pose_mode == "task-root"

        def _spawn_pose():
            if self.args.object_pose_mode == "task-root":
                return task_root_child_translation, task_root_child_orientation
            if self.args.object_pose_mode == "baked":
                # USDs already bake the world transform into the mesh hierarchy.
                return (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)
            return None  # scene-context: use each object's own world pose

        body_prim_paths: dict[str, str] = {}
        spawn_pose = _spawn_pose()

        # Pass 1: spawn the physics bodies (everything that is not a render skin).
        for object_name, entry in objects.items():
            if object_name not in usd_paths:
                print(f"[WARN]: No USD mapping for object '{object_name}', skipping.")
                continue
            if glue and (_is_render(object_name) or object_name in mounted_to_body):
                continue  # render skin or mounted child; spawned in pass 2 (glued onto its body)

            scene_pos, scene_quat = pose_from_world_location(entry)
            pos, quat = spawn_pose if spawn_pose is not None else (scene_pos, scene_quat)

            # Optional manual nudge of the wand body (its glued ring skin rides along)
            # so the rod can be centered in the ring for collider testing (--wand-offset).
            off = getattr(self.args, "wand_offset", None)
            if off and any(off) and "wand" in object_name.lower() and not _is_render(object_name):
                pos = (pos[0] + off[0], pos[1] + off[1], pos[2] + off[2])
                print(f"[INFO]: Applied --wand-offset {tuple(off)} to '{object_name}'.")

            hidden_by_flags = object_name in hidden_objects or (
                self.args.hide_root and (object_name.endswith("_root") or "boundary_root" in object_name)
            )
            skin_name = body_to_skin.get(object_name) if glue else None
            is_body = skin_name is not None  # this body has a render skin glued on top
            # A body with a skin is hidden by default (the skin covers it) UNLESS the
            # skin itself is hidden, in which case the body becomes the visible mesh.
            if is_body:
                skin_visible = self._appearance_for(skin_name).get("visible", True)
                default_visible = not skin_visible
            else:
                default_visible = True
            appearance = self._appearance_for(object_name)
            visible = bool(appearance.get("visible", default_visible)) and not hidden_by_flags

            cfg = sim_utils.UsdFileCfg(usd_path=str(usd_paths[object_name]), visible=visible)
            prim_path = f"{parent_path}/{_prim_name(object_name)}"
            cfg.func(prim_path, cfg, translation=pos, orientation=quat)
            body_prim_paths[object_name] = prim_path
            if visible:
                self._author_appearance(prim_path, appearance)
            print(f"[INFO]: Placed {object_name} (visible={visible}, body_with_skin={is_body}).")
        self.body_prim_paths = dict(body_prim_paths)

        # Pass 2: spawn each render skin, glued onto its body so it rides along when
        # grasped (or standalone if it has no matching body). Hidden skins are skipped.
        if glue:
            for object_name in objects:
                if not _is_render(object_name) or object_name not in usd_paths:
                    continue
                appearance = self._appearance_for(object_name)
                if not appearance.get("visible", True):
                    print(f"[INFO]: Hiding render mesh '{object_name}' (per appearance config).")
                    continue

                body_name = skin_to_body.get(object_name)
                if body_name and body_name in body_prim_paths:
                    skin_path = f"{body_prim_paths[body_name]}/{_prim_name(object_name)}"
                    translation, orientation = (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)
                    note = f"glued onto body '{body_name}'"
                else:
                    # No paired body: place it standalone at the body spawn pose.
                    skin_path = f"{parent_path}/{_prim_name(object_name)}"
                    scene_pos, scene_quat = pose_from_world_location(objects[object_name])
                    translation, orientation = spawn_pose if spawn_pose is not None else (scene_pos, scene_quat)
                    note = "standalone (no paired body)"

                cfg = sim_utils.UsdFileCfg(usd_path=str(usd_paths[object_name]), visible=True)
                cfg.func(skin_path, cfg, translation=translation, orientation=orientation)
                self._author_appearance(skin_path, appearance)
                self.skin_prim_paths[object_name] = skin_path
                print(f"[INFO]: Placed render mesh '{object_name}' ({note}).")

            # Pass 2b: mounted child shapes (mount_on_parent) glued under their parent body so
            # they ride it through the press and when the body is carried. Same gluing as a skin.
            for object_name, body_name in mounted_to_body.items():
                if object_name not in usd_paths or body_name not in body_prim_paths:
                    continue
                appearance = self._appearance_for(object_name)
                if not appearance.get("visible", True):
                    print(f"[INFO]: Hiding mounted child '{object_name}' (per appearance config).")
                    continue
                skin_path = f"{body_prim_paths[body_name]}/{_prim_name(object_name)}"
                cfg = sim_utils.UsdFileCfg(usd_path=str(usd_paths[object_name]), visible=True)
                cfg.func(skin_path, cfg, translation=(0.0, 0.0, 0.0), orientation=(1.0, 0.0, 0.0, 0.0))
                self._author_appearance(skin_path, appearance)
                self.skin_prim_paths[object_name] = skin_path
                print(f"[INFO]: Placed mounted child '{object_name}' glued onto body '{body_name}'.")

        if glue and getattr(self.args, "match_graspable_to_report", True):
            self._snap_graspable_to_report()

    def _snap_graspable_to_report(self):
        """Move graspable objects (e.g. the weights) -- and anything rigidly mounted on
        them -- onto the sampled world pose from the scene-context report, overriding the
        pose baked into their USD.

        In task-root mode every prim shares one TaskRoot child transform, so an object's
        position lives in its baked mesh, not its prim transform. We measure the object's
        spawned world-space geometry centroid and add a corrective translate (rotated into
        the TaskRoot frame) so the centroid lands on the reported position.

        Objects mounted on a graspable (e.g. change_channel's +/- buttons on the
        ``tv_remote``) are tagged ``mounted_on_graspable`` in the report. They spawn at the
        *canonical* graspable layout, so without this they would be left behind when the
        graspable snaps to its (translated and yaw-rotated) sampled pose. Snapping each to
        its own reported world pose keeps the whole assembly together and aligned with the
        button-press waypoints, which are defined against the same sampled configuration.

        The translate is applied to the physics body; its glued visual skin is a child and
        rides along. Because the body's collision mesh is usually hidden (the skin covers
        it), we measure the visible skin instead - a hidden prim has an empty world bound.
        Render skins are not in ``body_prim_paths`` and are skipped here (they ride bodies).

        Beyond the tagged graspables, this also snaps any *other* body the rigid
        task-root transform leaves further than ``SNAP_MOVER_THRESHOLD_M`` from its
        reported pose: an object RLBench re-places independently each reset (a second
        jar, a distractor cup/block, the hockey ball) that is not tagged graspable and
        would otherwise float at the canonical layout pose. Correctly-placed rigid
        objects measure within ~1.6 cm of their report pose, well under the threshold,
        so they are left exactly as the transform placed them.
        """
        graspable = set(self.ctx.graspable_names)
        mounted = {
            name
            for name, entry in self.ctx.objects.items()
            if entry.get("mounted_on_graspable")
        }
        snap_names = graspable | mounted
        # Every spawned body is a snap candidate; non-graspable ones are only moved if
        # measured far from their report pose (the displacement gate in the loop).
        candidates = sorted(set(self.body_prim_paths) | snap_names)
        if not candidates:
            return

        from pxr import Gf, Usd, UsdGeom

        stage = sim_utils.get_current_stage()
        bbox = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
        )
        inv_root_quat = _qinv(self.ctx.sampled_task_root_quat)
        body_to_skin = {body: skin for skin, body in self._skin_to_body.items()}

        # Only graspable *assemblies* (a graspable that has mounted children, plus those
        # children) need their sampled spawn ORIENTATION applied. Lone graspables (weights,
        # a ball) keep the proven translate-only behaviour, so this can't regress them.
        mounted_graspables = {
            entry.get("mounted_on_graspable")
            for entry in self.ctx.objects.values()
            if entry.get("mounted_on_graspable")
        }
        orient_names = {
            name
            for name in snap_names
            if self.ctx.objects.get(name, {}).get("mounted_on_graspable") or name in mounted_graspables
        }
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

        for name in candidates:
            body_path = self.body_prim_paths.get(name)
            target = (self.ctx.objects.get(name, {}).get("world_location") or {}).get("position_xyz_m")
            if not body_path or not target or len(target) != 3:
                continue
            # Measure the visible geometry (the glued skin if present, else the body), but
            # move the body so the glued skin child rides along.
            skin_name = body_to_skin.get(name)
            measure_path = self.skin_prim_paths.get(skin_name, body_path) if skin_name else body_path
            measure_prim = stage.GetPrimAtPath(measure_path)
            body_prim = stage.GetPrimAtPath(body_path)
            if not measure_prim or not measure_prim.IsValid() or not body_prim or not body_prim.IsValid():
                continue

            # Apply the sampled spawn ORIENTATION first (the +/-pi yaw the task-root canonical
            # placement omits). Without it the remote is drawn at its canonical yaw, so the
            # buttons -- even at the right positions -- don't sit on its face and the press
            # waypoints don't match. Rotating about the prim origin shifts the geometry, so we
            # do it BEFORE measuring the centroid for the position correction below.
            report_quat_xyzw = (self.ctx.objects.get(name, {}).get("world_location") or {}).get("quaternion_xyzw")
            if name in orient_names and report_quat_xyzw and len(report_quat_xyzw) == 4:
                rx, ry, rz, rw = (float(v) for v in report_quat_xyzw)
                report_quat = (rw, rx, ry, rz)
                cur_gf = xform_cache.GetLocalToWorldTransform(body_prim).ExtractRotationQuat()
                cur_im = cur_gf.GetImaginary()
                cur_quat = (cur_gf.GetReal(), cur_im[0], cur_im[1], cur_im[2])
                # World rotation that fixes this body, expressed in the (rotated) TaskRoot frame:
                #   R_local = inv(root) . report . inv(current) . root
                r_local = _qmul(
                    inv_root_quat,
                    _qmul(report_quat, _qmul(_qinv(cur_quat), self.ctx.sampled_task_root_quat)),
                )
                xf_orient = UsdGeom.Xformable(body_prim)
                orient_op = next(
                    (op for op in xf_orient.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeOrient),
                    None,
                )
                if orient_op is None:
                    orient_op = xf_orient.AddOrientOp()
                if orient_op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
                    orient_op.Set(Gf.Quatf(r_local[0], Gf.Vec3f(r_local[1], r_local[2], r_local[3])))
                else:
                    orient_op.Set(Gf.Quatd(r_local[0], Gf.Vec3d(r_local[1], r_local[2], r_local[3])))
                bbox.Clear()  # the body just rotated; recompute its bounds for the position fix

            world_range = bbox.ComputeWorldBound(measure_prim).ComputeAlignedRange()
            if world_range.IsEmpty():
                continue
            midpoint = world_range.GetMidpoint()
            center = (float(midpoint[0]), float(midpoint[1]), float(midpoint[2]))
            target = tuple(float(v) for v in target)
            # Tagged graspables/assemblies always snap (and may carry an orientation fix). A plain
            # body only snaps if the rigid task-root transform left it far from its report pose -
            # i.e. it is an independently re-placed instance. Compare LIKE WITH LIKE: when the
            # measured visible geometry is a glued skin, gate against that SKIN's OWN reported pose,
            # not the body origin's. A skin can legitimately sit far from the body origin (a basket's
            # backboard skin reaches well beyond the ring's collision body); comparing the skin's
            # centre to the body-origin report reads that offset as a ~0.2 m misplacement and yanks
            # the whole assembly. Conversely it must NOT use the body's collision geometry, which on
            # some tasks is baked displaced from the visible mesh (e.g. change_clock's clock collider
            # lies flat ~0.26 m off) - that would drag the correctly-placed visible mesh with it.
            if name not in snap_names:
                if skin_name and measure_path != body_path:
                    skin_report = (self.ctx.objects.get(skin_name, {}).get("world_location") or {}).get("position_xyz_m")
                    if skin_report and len(skin_report) == 3:
                        target = tuple(float(v) for v in skin_report)
                if max(abs(target[i] - center[i]) for i in range(3)) < SNAP_MOVER_THRESHOLD_M:
                    continue
            delta_world = tuple(target[i] - center[i] for i in range(3))
            displacement = max(abs(d) for d in delta_world)
            if displacement < 1e-5:
                continue
            # The body's translate op is expressed in the (rotated) TaskRoot frame.
            delta_local = _qapply(inv_root_quat, delta_world)
            xf = UsdGeom.Xformable(body_prim)
            translate_op = next(
                (
                    op
                    for op in xf.GetOrderedXformOps()
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
                ),
                None,
            )
            if translate_op is None:
                translate_op = xf.AddTranslateOp()
            current = translate_op.Get()
            current = (0.0, 0.0, 0.0) if current is None else (current[0], current[1], current[2])
            translate_op.Set(Gf.Vec3d(*(current[i] + delta_local[i] for i in range(3))))
            oriented = name in orient_names and report_quat_xyzw and len(report_quat_xyzw) == 4
            kind = "assembly" if oriented else ("graspable" if name in snap_names else "mover")
            print(
                f"[INFO]: Matched {kind} '{name}' to report pose "
                f"{tuple(round(v, 4) for v in target)} "
                f"(moved {tuple(round(v, 4) for v in delta_world)} m"
                f"{', + sampled orientation' if oriented else ''} from its baked USD pose)."
            )

    def _spawn_path_segment(self, parent_path: str, name: str, start, end, color):
        mid = tuple((start[i] + end[i]) * 0.5 for i in range(3))
        delta = tuple(end[i] - start[i] for i in range(3))
        length = sum(component * component for component in delta) ** 0.5
        if length < 1.0e-6:
            return

        cfg = sim_utils.CapsuleCfg(
            radius=0.004,
            height=length,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.45),
        )
        cfg.func(f"{parent_path}/{name}", cfg, translation=mid, orientation=_quat_from_z_axis_to_vector(delta))

    def _spawn_waypoint_marker(self, parent_path: str, name: str, pos, quat, color, radius: float = 0.012):
        marker_cfg = sim_utils.SphereCfg(
            radius=radius,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.6),
        )
        marker_cfg.func(f"{parent_path}/{name}", marker_cfg, translation=pos, orientation=quat)

    def _spawn_waypoints(self):
        if self.args.no_waypoints:
            return

        self._ensure_xform("/World/Waypoints")
        for index, waypoint in enumerate(self.ctx.waypoints):
            pos, quat = pose_from_world_location(waypoint)
            color = WAYPOINT_COLORS[index % len(WAYPOINT_COLORS)]
            waypoint_path = f"/World/Waypoints/{waypoint['name']}"
            self._ensure_xform(waypoint_path)

            self._spawn_waypoint_marker(waypoint_path, "Pose", pos, quat, color)
            path_positions = _path_sample_positions(waypoint)
            if path_positions:
                for sample_index, sample_pos in enumerate(path_positions):
                    self._spawn_waypoint_marker(
                        waypoint_path,
                        f"PathSample{sample_index:02d}",
                        sample_pos,
                        (1.0, 0.0, 0.0, 0.0),
                        color,
                        radius=0.009,
                    )
                for segment_index, (start, end) in enumerate(zip(path_positions, path_positions[1:])):
                    self._spawn_path_segment(waypoint_path, f"PathSegment{segment_index:02d}", start, end, color)
                print(
                    f"[INFO]: Marked {waypoint['name']} path with {len(path_positions)} samples "
                    f"at start={tuple(round(v, 6) for v in path_positions[0])} "
                    f"end={tuple(round(v, 6) for v in path_positions[-1])}"
                )
            else:
                print(f"[INFO]: Marked {waypoint['name']} at pos={tuple(round(v, 6) for v in pos)}")

    def _spawn_robot(self):
        if self.args.no_robot:
            return None
        robot = spawn_franka(
            prim_path="/World/DesignScene/Robot",
            base_pos=self.ctx.robot_base_pos,
            base_quat_wxyz=self.ctx.robot_base_quat,
        )
        print(
            f"[INFO]: Spawned Franka at base pos={tuple(round(v, 6) for v in self.ctx.robot_base_pos)} "
            f"(x,y from report robot_base, z on table top)."
        )
        return robot

    # ------------------------------------------------------------------
    # Public API.
    # ------------------------------------------------------------------
    def _apply_collision_filters(self):
        """Disable collision between named object pairs (``--filter-collision A B``).

        Used when two exported objects' colliders overlap at spawn (e.g. the wand's
        handle pokes into the base box), which would otherwise eject one of them.
        """
        pairs = getattr(self.args, "filter_collision", None) or []
        if not pairs:
            return
        from pxr import UsdPhysics

        stage = sim_utils.get_current_stage()
        for a, b in pairs:
            pa, pb = self.body_prim_paths.get(a), self.body_prim_paths.get(b)
            if not pa or not pb:
                print(f"[WARN]: --filter-collision: object not found ({a} or {b}); have {list(self.body_prim_paths)}.")
                continue
            api = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(pa))
            api.CreateFilteredPairsRel().AddTarget(pb)
            print(f"[INFO]: Collision filtered: '{a}' <-> '{b}'.")

    def design_scene(self):
        self._ensure_xform("/World/DesignScene")
        self._spawn_lights()
        self._spawn_floor_and_table()
        self._spawn_usd_objects()
        self._apply_collision_filters()
        self._spawn_waypoints()
        return self._spawn_robot()
