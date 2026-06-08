"""
add_physics_to_usds.py

Batch-bake physics onto exported AHA/RLBench object USDs so a robot can actually
grasp/collide with them, then write self-contained copies to a new folder (the
originals are never modified).

Per object it applies one of:
  * ``rigid``      - movable rigid body the gripper can pick up (convex collider,
                     mass/density, friction material),
  * ``kinematic``  - static collider that stays put (exact triangle mesh),
  * ``visual``     - copied through with no physics (pure decoration),
  * ``deformable`` - FEM soft body (EXPERIMENTAL / untested in this sandbox).

Any object can also set ``"collision": false`` to keep its body type but author NO
collider, so the robot (and other bodies) pass straight through it. Use this to
clear a base/holder out of the arm's path while leaving the graspable object
collidable.

What each object becomes is driven by the per-task object physics config under
``task_data/object_physics/<task>.json`` (density, type, friction, ...), selected
with ``--task``:

    cd ~/IsaacLab
    ./isaaclab.sh -p scripts/aha_in_isaac/add_physics_to_usds.py \
        --input-dir task_usds/basketball_in_hoop \
        --output-dir task_usds/basketball_in_hoop_physics \
        --task basketball_in_hoop

Then point the scene runner at the new folder, e.g.
    --usd-dir task_usds/basketball_in_hoop_physics

To bake EVERY task in one Isaac session (skips already-baked folders and any with
no config block, reports failures at the end):
    ./isaaclab.sh -p scripts/aha_in_isaac/add_physics_to_usds.py \
        --batch-root task_usds

Without ``--task`` the legacy ``--dynamic/--kinematic/--visual`` name-suffix flags
(plus ``--density`` etc.) are used instead, classifying any unmatched file as
``--default``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ISAACLAB_ROOT = Path(__file__).resolve().parents[2]
for _package_dir in ("isaaclab", "isaaclab_assets", "isaaclab_tasks"):
    _source_path = ISAACLAB_ROOT / "source" / _package_dir
    if _source_path.is_dir() and str(_source_path) not in sys.path:
        sys.path.insert(0, str(_source_path))

from isaaclab.app import AppLauncher

USD_EXTENSIONS = (".usd", ".usdc", ".usda")
# Per-task object physics config: one file per task under task_data/object_physics/<task>.json,
# with the shared defaults/docs in task_data/object_physics/_defaults.json.
DEFAULT_CONFIG = Path(__file__).resolve().parent / "task_data" / "object_physics"

parser = argparse.ArgumentParser(description="Bake physics onto a folder of object USDs.")
parser.add_argument("--input-dir", type=Path, default=None, help="Folder of source USD files (single-task mode).")
parser.add_argument("--output-dir", type=Path, default=None, help="Folder to write physics-enabled copies.")
parser.add_argument(
    "--batch-root",
    type=Path,
    default=None,
    help="Bake EVERY task subfolder under this root (e.g. task_usds) to <task><output-suffix>, in one "
    "Isaac session. Uses each folder's name as --task. Skips folders already baked and those with no config block.",
)
parser.add_argument("--output-suffix", default="_physics", help="Suffix for batch output folders (default '_physics').")
parser.add_argument(
    "--config", type=Path, default=DEFAULT_CONFIG,
    help="Per-task object physics config: a directory of <task>.json files (+ _defaults.json), "
    "or a single legacy combined JSON file."
)
parser.add_argument(
    "--task",
    default=None,
    help="Task block in --config to drive baking. If omitted, the legacy --dynamic/--kinematic/--visual flags are used.",
)
# --- Legacy / fallback flags (used only when --task is not given) ---
parser.add_argument("--mass", type=float, default=0.05, help="Explicit mass (kg). Ignored if --density > 0.")
parser.add_argument("--density", type=float, default=0.0, help="Material density (kg/m^3); weight = density * volume.")
parser.add_argument("--compliant-stiffness", type=float, default=0.0, help="If > 0, soft/springy contact stiffness.")
parser.add_argument("--compliant-damping", type=float, default=0.0, help="Damping for compliant contact.")
parser.add_argument(
    "--approximation",
    default="convexHull",
    choices=("convexHull", "convexDecomposition", "boundingCube", "boundingSphere"),
    help="Collider approximation for DYNAMIC objects (static objects always use the exact mesh).",
)
parser.add_argument("--static-friction", type=float, default=1.2)
parser.add_argument("--dynamic-friction", type=float, default=1.0)
parser.add_argument("--restitution", type=float, default=0.0)
parser.add_argument("--dynamic", action="append", default=[], help="Name suffix -> rigid body. Repeatable.")
parser.add_argument("--kinematic", action="append", default=[], help="Name suffix -> static collider. Repeatable.")
parser.add_argument("--visual", action="append", default=[], help="Name suffix -> no physics. Repeatable.")
parser.add_argument(
    "--default",
    default="rigid",
    choices=("rigid", "kinematic", "visual", "deformable"),
    help="Type for files that match no filter / no config entry.",
)
parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files in --output-dir.")
parser.add_argument(
    "--no-import-physics", action="store_true",
    help="Ignore task_data/physics (CoppeliaSim mass/friction); use only the hand-authored config.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# This tool only edits USD on disk; no GUI needed.
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# pxr is only importable once the Isaac/USD libraries are initialised above.
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: E402

from usd_uv import generate_uvs, has_uvs  # noqa: E402

try:
    from pxr import PhysxSchema  # noqa: E402

    _HAS_PHYSX = True
except ImportError:
    _HAS_PHYSX = False

APPROXIMATION_TOKENS = {
    "convexHull": UsdPhysics.Tokens.convexHull,
    "convexDecomposition": UsdPhysics.Tokens.convexDecomposition,
    "boundingCube": UsdPhysics.Tokens.boundingCube,
    "boundingSphere": UsdPhysics.Tokens.boundingSphere,
    "none": UsdPhysics.Tokens.none,
}
# Material prims are created *inside* each object's root prim so the binding
# survives when the USD is later referenced into another stage.
PHYSICS_MATERIAL_NAME = "PhysicsMaterial"
DEFORMABLE_MATERIAL_NAME = "DeformableMaterial"

# Normalise the various spellings into the four canonical types.
TYPE_ALIASES = {
    "rigid": "rigid",
    "dynamic": "rigid",
    "kinematic": "kinematic",
    "static": "kinematic",
    "visual": "visual",
    "none": "visual",
    "deformable": "deformable",
    "soft": "deformable",
}


# ----------------------------------------------------------------------
# Spec resolution: figure out the physics spec (dict) for each USD file.
# ----------------------------------------------------------------------
def _legacy_type(stem: str) -> str:
    name = stem.lower()

    def matches(tokens):
        return any(name == t.lower() or name.endswith(t.lower()) for t in tokens)

    if matches(args_cli.visual):
        return "visual"
    if matches(args_cli.kinematic):
        return "kinematic"
    if matches(args_cli.dynamic):
        return "rigid"
    return TYPE_ALIASES.get(args_cli.default, "rigid")


def _legacy_spec(stem: str) -> dict:
    return {
        "type": _legacy_type(stem),
        "density": args_cli.density,
        "mass": None if args_cli.density > 0.0 else args_cli.mass,
        "static_friction": args_cli.static_friction,
        "dynamic_friction": args_cli.dynamic_friction,
        "restitution": args_cli.restitution,
        "collider": args_cli.approximation,
        "compliant_stiffness": args_cli.compliant_stiffness,
        "compliant_damping": args_cli.compliant_damping,
        "youngs_modulus": 50000.0,
        "poissons_ratio": 0.4,
        "uv": "none",
    }


def _load_task_config(config_path: Path, task: str) -> tuple[dict, dict]:
    """Resolve (defaults, task_block) for ``task``. ``config_path`` is normally a directory of
    per-task ``<task>.json`` files plus a shared ``_defaults.json`` (``{"_defaults": {...}}``);
    a single combined JSON file is still accepted for backwards compatibility."""
    if config_path.is_dir():
        defaults_path = config_path / "_defaults.json"
        defaults = {}
        if defaults_path.is_file():
            defaults = (json.loads(defaults_path.read_text(encoding="utf-8")) or {}).get("_defaults", {})
        task_path = config_path / f"{task}.json"
        if not task_path.is_file():
            avail = ", ".join(sorted(p.stem for p in config_path.glob("*.json") if not p.stem.startswith("_"))) or "(none)"
            raise KeyError(f"Task '{task}' config not found at {task_path}. Available: {avail}")
        data = json.loads(task_path.read_text(encoding="utf-8")) or {}
        task_block = {k: v for k, v in data.items() if not k.startswith("_")}
        return defaults, task_block
    if not config_path.is_file():
        raise FileNotFoundError(f"Object physics config not found: {config_path}")
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if task not in data:
        tasks = ", ".join(k for k in data if not k.startswith("_")) or "(none)"
        raise KeyError(f"Task '{task}' not in {config_path}. Available: {tasks}")
    defaults = data.get("_defaults", {})
    task_block = {k: v for k, v in data[task].items() if not k.startswith("_")}
    return defaults, task_block


TASK_PHYSICS_DIR = Path(__file__).resolve().parent / "task_data" / "physics"


def _load_task_physics(task: str) -> dict:
    """Real CoppeliaSim per-shape physics (mass/friction), keyed by object name.
    Empty if not extracted yet (run ``build_task_physics.py``)."""
    path = TASK_PHYSICS_DIR / f"{task}.json"
    if not path.is_file():
        return {}
    return (json.loads(path.read_text(encoding="utf-8")) or {}).get("shapes", {}) or {}


def _physics_for_stem(stem: str, shapes: dict) -> dict:
    """The physics entry whose object name best (longest-suffix) matches a USD stem."""
    name = stem.lower()
    best, best_len = {}, -1
    for obj, vals in shapes.items():
        token = obj.lower()
        if (name == token or name.endswith(token)) and len(token) > best_len:
            best, best_len = vals or {}, len(token)
    return best


def _config_spec(stem: str, defaults: dict, task_block: dict, physics: dict | None = None) -> tuple[dict, bool]:
    """Merge the best-matching object entry over the defaults (longest key wins).

    Real CoppeliaSim physics (``physics``: mass/friction) sits between the global
    ``defaults`` and the per-task config, so it improves the defaults while explicit
    per-task values still win."""
    name = stem.lower()
    best_key, best_len = None, -1
    for key in task_block:
        token = key.lower()
        if (name == token or name.endswith(token)) and len(token) > best_len:
            best_key, best_len = key, len(token)
    spec = dict(defaults)
    if physics:
        if physics.get("dynamic") is not None:
            spec["type"] = "rigid" if bool(physics["dynamic"]) else "kinematic"
        if physics.get("collidable") is not None:
            spec["collision"] = bool(physics["collidable"])
        friction = physics.get("friction")
        if friction is not None:
            spec["static_friction"] = float(friction)
            spec["dynamic_friction"] = float(friction)
        # A hand-authored physics JSON may give a material density (kg/m^3); otherwise use the
        # CoppeliaSim-measured mass. Either one clears the other so the baker uses the one given.
        if physics.get("density") is not None:
            spec["density"] = float(physics["density"])
            spec["mass"] = None
        elif physics.get("mass") is not None:
            spec["mass"] = float(physics["mass"])
            spec["density"] = None
        # Collider approximation + contact tuning may also be authored in the physics JSON, so a
        # hollow box / thin lid can get a CONCAVE-correct collider ('convexDecomposition' = the
        # walls/cavity as separate convex pieces) instead of the convex-hull default that fills the
        # cavity into a solid block. Per-task config still overrides these.
        for _k in ("collider", "contact_offset", "rest_offset",
                   "max_convex_hulls", "hull_vertex_limit", "voxel_resolution", "sdf_resolution"):
            if physics.get(_k) is not None:
                spec[_k] = physics[_k]
    if best_key is not None:
        spec.update(task_block[best_key])
    spec["type"] = TYPE_ALIASES.get(str(spec.get("type", "rigid")).lower(), "rigid")
    return spec, best_key is not None


# ----------------------------------------------------------------------
# USD authoring.
# ----------------------------------------------------------------------
def _root_prim(stage: Usd.Stage) -> Usd.Prim:
    root = stage.GetDefaultPrim()
    if root and root.IsValid():
        return root
    for child in stage.GetPseudoRoot().GetChildren():
        return child
    raise RuntimeError("Stage has no usable root prim.")


def _make_physics_material(stage: Usd.Stage, root: Usd.Prim, spec: dict) -> UsdShade.Material:
    material_path = root.GetPath().AppendChild(PHYSICS_MATERIAL_NAME)
    prim = stage.GetPrimAtPath(material_path)
    if prim and prim.IsValid():
        return UsdShade.Material(prim)
    material = UsdShade.Material.Define(stage, material_path)
    physics_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_api.CreateStaticFrictionAttr(float(spec["static_friction"]))
    physics_api.CreateDynamicFrictionAttr(float(spec["dynamic_friction"]))
    physics_api.CreateRestitutionAttr(float(spec["restitution"]))
    if spec.get("density") and float(spec["density"]) > 0.0:
        physics_api.CreateDensityAttr(float(spec["density"]))
    # The friction COMBINE mode decides how this material's friction mixes with the
    # contacting body's. PhysX defaults to 'average', so a high object friction is
    # dragged down by a low-friction gripper finger (and the Franka finger colliders are
    # INSTANCED, so the runtime high-friction pad material silently fails to bind). Setting
    # 'max' here makes the grippable object's own friction win the contact regardless of the
    # finger material, so a flat/light object (e.g. a TV remote) isn't squeezed out of the grip.
    if _HAS_PHYSX and (spec.get("friction_combine_mode") or float(spec.get("compliant_stiffness") or 0.0) > 0.0):
        physx_material = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
        if spec.get("friction_combine_mode"):
            physx_material.CreateFrictionCombineModeAttr(str(spec["friction_combine_mode"]))
        if float(spec.get("compliant_stiffness") or 0.0) > 0.0:
            physx_material.CreateCompliantContactStiffnessAttr(float(spec["compliant_stiffness"]))
            physx_material.CreateCompliantContactDampingAttr(float(spec.get("compliant_damping") or 0.0))
    return material


def _bake_rigid(stage: Usd.Stage, spec: dict, kinematic: bool) -> int:
    root = _root_prim(stage)

    # ``collision`` (default True) decides whether the object gets a physics
    # collider at all. With it False the robot/other bodies pass straight through
    # the object (it is render-only for physics). A *static* (kinematic) prop with
    # no collider is just decoration, so we skip the rigid body too; a *dynamic*
    # body with no collider is still authored on request (it falls under gravity
    # but never makes contact).
    want_collision = bool(spec["collision"])
    author_body = want_collision or not kinematic
    if not want_collision and not kinematic:
        print("         collision=false on a rigid body: it will fall freely and never collide.")

    if author_body:
        rigid_api = UsdPhysics.RigidBodyAPI.Apply(root)
        rigid_api.CreateRigidBodyEnabledAttr(True)
        rigid_api.CreateKinematicEnabledAttr(kinematic)
        if _HAS_PHYSX:
            physx_rigid = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
            # Higher solver iteration counts give a more accurate, stable grasp.
            physx_rigid.CreateSolverPositionIterationCountAttr(16)
            physx_rigid.CreateSolverVelocityIterationCountAttr(4)
            # Pre-positioned objects (no support under them) can be pinned in place by
            # disabling gravity, so they don't fall/eject before being grasped.
            if spec.get("disable_gravity") and not kinematic:
                physx_rigid.CreateDisableGravityAttr(True)
            # Continuous collision detection stops a fast-falling thin body from
            # tunnelling through a thin collider (e.g. the table top) and then
            # exploding out of a deep penetration. Needs scene CCD on too.
            if spec.get("ccd") and not kinematic:
                physx_rigid.CreateEnableCCDAttr(True)
            # Cap the depenetration velocity so an SDF-vs-thin-rod contact resolves a deep
            # overlap gently instead of explosively ejecting the body (the ring-on-rod case).
            if spec.get("max_depenetration_velocity") is not None:
                physx_rigid.CreateMaxDepenetrationVelocityAttr(float(spec["max_depenetration_velocity"]))
        mass_api = UsdPhysics.MassAPI.Apply(root)
        if spec.get("density") and float(spec["density"]) > 0.0:
            mass_api.CreateDensityAttr(float(spec["density"]))
        else:
            mass_api.CreateMassAttr(float(spec.get("mass") or 0.05))

    material = _make_physics_material(stage, root, spec) if want_collision else None
    # Static (kinematic) bodies default to the EXACT mesh, but honour an explicit collider so a
    # hollow base box can be made SOLID: an exact triangle mesh is a one-sided shell, and a thin
    # dynamic body (e.g. the wand handle) can tunnel through a face and then sit INSIDE it with
    # no contact to push it out. 'convexDecomposition' makes the base a solid hull while keeping
    # the thin bent wire as separate thin hulls (a plain convexHull would fill the wire's arch).
    # Dynamic bodies need a convex/SDF collider.
    if kinematic:
        collider = str(spec.get("collider", "none"))
    else:
        collider = str(spec.get("collider", "convexHull"))
    # 'sdf' gives a DYNAMIC body an accurate *concave* collider via a signed-distance
    # field: it keeps holes/cavities (e.g. a ring that must thread onto a rod), where
    # a convex hull would fill the hole and eject the body. The mesh collision
    # approximation attribute MUST be the 'sdf' token (PhysxSchema.Tokens.sdf); merely
    # applying PhysxSDFMeshCollisionAPI while leaving approximation 'none' makes PhysX
    # treat it as a raw triangle mesh, which is INVALID for a dynamic body and silently
    # falls back to convexHull (which fills the hole and ejects the grasped body).
    use_sdf = collider == "sdf" and not kinematic and _HAS_PHYSX
    if use_sdf:
        approximation = PhysxSchema.Tokens.sdf
    else:
        approximation = APPROXIMATION_TOKENS.get(collider, APPROXIMATION_TOKENS["convexHull"])

    uv_mode = str(spec.get("uv") or "none")
    uv_used = set()
    mesh_count = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if want_collision:
            UsdPhysics.CollisionAPI.Apply(prim)
            mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_collision.CreateApproximationAttr(approximation)
            if _HAS_PHYSX:
                if use_sdf:
                    sdf_api = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
                    sdf_api.CreateSdfResolutionAttr(int(spec.get("sdf_resolution", 256)))
                elif collider == "convexHull" and not kinematic:
                    PhysxSchema.PhysxConvexHullCollisionAPI.Apply(prim)
                elif collider == "convexDecomposition":
                    # Many thin hulls so a bent wire is decomposed piece-by-piece (kept thin)
                    # rather than bridged into a solid blob that would fill the ring's path.
                    cd = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
                    cd.CreateMaxConvexHullsAttr(int(spec.get("max_convex_hulls", 64)))
                    cd.CreateHullVertexLimitAttr(int(spec.get("hull_vertex_limit", 64)))
                    cd.CreateVoxelResolutionAttr(int(spec.get("voxel_resolution", 500000)))
                    cd.CreateShrinkWrapAttr(True)
                # Widen the contact offset so a thin collider (the wand) registers contact
                # EARLY - before a fast/stiff gripper or a drop can penetrate it. rest_offset
                # keeps the resting separation at ~0 so it still sits flush on the rod.
                co = spec.get("contact_offset")
                ro = spec.get("rest_offset")
                if co is not None or ro is not None:
                    pc = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                    if co is not None:
                        pc.CreateContactOffsetAttr(float(co))
                    if ro is not None:
                        pc.CreateRestOffsetAttr(float(ro))
            binding = UsdShade.MaterialBindingAPI.Apply(prim)
            binding.Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")
        # Only synthesise UVs when the mesh has none, so we never clobber good ones.
        if uv_mode != "none" and not has_uvs(prim):
            resolved = generate_uvs(prim, uv_mode)
            if resolved:
                uv_used.add(resolved)
        mesh_count += 1
    if uv_used:
        detail = f"{uv_mode}->{'/'.join(sorted(uv_used))}" if uv_mode == "auto" else uv_mode
        print(f"         generated UVs ({detail}).")
    return mesh_count


def _bake_deformable(stage: Usd.Stage, spec: dict) -> int:
    """EXPERIMENTAL: apply a PhysX FEM soft body. Untested in this sandbox."""
    if not _HAS_PHYSX:
        print("[WARN]: deformable requested but PhysxSchema unavailable; baking as rigid instead.")
        return _bake_rigid(stage, spec, kinematic=False)

    root = _root_prim(stage)
    material = UsdShade.Material.Define(stage, root.GetPath().AppendChild(DEFORMABLE_MATERIAL_NAME))
    deformable_material = PhysxSchema.PhysxDeformableBodyMaterialAPI.Apply(material.GetPrim())
    deformable_material.CreateDensityAttr(float(spec.get("density") or 50.0))
    deformable_material.CreateYoungsModulusAttr(float(spec.get("youngs_modulus") or 50000.0))
    deformable_material.CreatePoissonsRatioAttr(float(spec.get("poissons_ratio") or 0.4))
    deformable_material.CreateDynamicFrictionAttr(float(spec.get("dynamic_friction") or 1.0))

    mesh_count = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        deformable = PhysxSchema.PhysxDeformableBodyAPI.Apply(prim)
        deformable.CreateSolverPositionIterationCountAttr(20)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")
        mesh_count += 1
    return mesh_count


def _bake(src: Path, dst: Path, spec: dict) -> tuple[str, int]:
    stage = Usd.Stage.Open(str(src))
    mode = spec["type"]

    if mode == "visual":
        count = 0
    elif mode == "deformable":
        count = _bake_deformable(stage, spec)
    else:  # rigid or kinematic
        count = _bake_rigid(stage, spec, kinematic=(mode == "kinematic"))

    dst.parent.mkdir(parents=True, exist_ok=True)
    stage.Export(str(dst))
    return mode, count


def bake_folder(input_dir: Path, output_dir: Path, task: str | None) -> int:
    """Bake every USD in ``input_dir`` to ``output_dir``. ``task`` selects the
    config block; if None, the legacy --dynamic/--kinematic/--visual flags apply.
    Returns the number of files baked."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    defaults, task_block = ({}, {})
    task_physics = {}
    if task:
        defaults, task_block = _load_task_config(args_cli.config, task)
        task_physics = {} if args_cli.no_import_physics else _load_task_physics(task)

    usd_files = sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in USD_EXTENSIONS
    )
    if not usd_files:
        print(f"[WARN]: No USD files found in {input_dir}")
        return 0

    print(f"[INFO]: Baking {len(usd_files)} file(s) {input_dir} -> {output_dir}")
    for src in usd_files:
        dst = output_dir / src.name
        if dst.exists() and not args_cli.overwrite:
            print(f"[SKIP]: {dst.name} already exists (use --overwrite).")
            continue
        if task:
            spec, matched = _config_spec(
                src.stem, defaults, task_block, _physics_for_stem(src.stem, task_physics)
            )
            if not matched:
                print(f"[WARN]: '{src.name}' matched no object rule; using default type '{spec['type']}'.")
        else:
            spec = _legacy_spec(src.stem)
        result_mode, mesh_count = _bake(src, dst, spec)
        density = spec.get("density")
        bits = []
        if result_mode != "visual" and density:
            bits.append(f"density={density}")
        if result_mode != "visual" and not spec["collision"]:
            bits.append("no-collision")
        detail = " ".join(bits)
        print(f"[OK]:   {src.name:<48} -> {result_mode:<10} ({mesh_count} mesh(es)) {detail}")
    return len(usd_files)


def _run_batch():
    root = args_cli.batch_root
    suffix = args_cli.output_suffix
    if not root.is_dir():
        raise FileNotFoundError(f"--batch-root not found: {root}")

    task_dirs = [d for d in sorted(root.iterdir()) if d.is_dir() and not d.name.endswith(suffix)]
    print(f"[INFO]: Batch baking under {root} ({len(task_dirs)} candidate task folder(s)).")

    baked, skipped, failed = [], [], []
    for task_dir in task_dirs:
        task = task_dir.name
        output_dir = root / f"{task}{suffix}"
        if output_dir.is_dir() and any(output_dir.iterdir()) and not args_cli.overwrite:
            skipped.append(f"{task} (already baked)")
            continue
        try:
            _load_task_config(args_cli.config, task)  # ensure a config block exists
        except KeyError:
            skipped.append(f"{task} (no config block)")
            continue
        print(f"\n=== {task} ===")
        try:
            bake_folder(task_dir, output_dir, task)
            baked.append(task)
        except Exception as exc:  # keep going; one bad task should not abort the batch
            print(f"[FAIL]: {task}: {exc}")
            failed.append(f"{task}: {exc}")

    print("\n========== BATCH SUMMARY ==========")
    print(f"  baked:   {len(baked)}")
    print(f"  skipped: {len(skipped)}")
    print(f"  failed:  {len(failed)}")
    for item in failed:
        print(f"    [FAIL] {item}")
    if skipped:
        print("  (skipped: " + ", ".join(skipped) + ")")


def main():
    if args_cli.batch_root:
        _run_batch()
    elif args_cli.input_dir and args_cli.output_dir:
        if args_cli.task:
            print(f"[INFO]: Using config '{args_cli.config.name}' task '{args_cli.task}'.")
        else:
            print("[INFO]: No --task given; using legacy --dynamic/--kinematic/--visual flags.")
        bake_folder(args_cli.input_dir, args_cli.output_dir, args_cli.task)
    else:
        raise SystemExit("Provide --input-dir and --output-dir (single task), or --batch-root (all tasks).")
    print("[INFO]: Done.")


if __name__ == "__main__":
    main()
    simulation_app.close()
