"""
generate_task_configs.py

One-shot generator: for every task that has a scene-context report, add a block
to object_physics_config.json and object_appearance_config.json, and synthesise a
texture image per object under textures/<task>/.

Everything is inferred from the object NAME (no Isaac/USD needed):
  * type      : '*_visual'/'*_vis' -> visual; structural/boundary/collision names
                -> kinematic; otherwise rigid (graspable).
  * texture   : a small procedural image whose style+color come from keywords in
                the name (wood / metal / food / fabric / plastic), with a stable
                hashed color as fallback so every object is distinct.
  * uv        : 'auto' for baked (rigid/kinematic) objects so textures can map.

Tasks already present in a config are left untouched (so hand-curated blocks like
basketball_in_hoop / wipe_desk are preserved). Pure Python; run with plain python3:

    python3 scripts/aha_in_isaac/generate_task_configs.py
"""

from __future__ import annotations

import colorsys
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = Path(__file__).resolve().parent
REPORTS_DIR = Path("/home/ramtin/AHA/portable_scene_reports")
TEXTURES_DIR = HERE / "textures"
PHYSICS_CONFIG = HERE / "object_physics_config.json"
APPEARANCE_CONFIG = HERE / "object_appearance_config.json"
TEX_SIZE = 128
# Hand-curated task blocks that the generator must never overwrite.
CURATED = {"basketball_in_hoop", "wipe_desk", "beat_the_buzz"}

# Keyword -> (base_color RGB 0..255, style, density kg/m^3). First match wins.
PALETTE = [
    (("table", "desk", "shelf", "book", "door", "drawer", "cupboard", "cabinet", "plank",
      "board", "frame", "stand", "chest", "crate", "block", "peg", "stick", "wand", "broom",
      "handle", "bat", "box"), (140, 94, 52), "wood", 400.0),
    (("knife", "blade", "fork", "spoon", "hook", "nail", "screw", "rim", "hoop", "pan", "pot",
      "grill", "key", "opener", "scissor", "wrench", "hinge", "needle", "crank", "bolt", "coin",
      "weight", "dumbbell", "barbell", "metal", "saucepan", "kettle"), (150, 152, 158), "metal", 1200.0),
    (("banana", "mustard", "lemon", "corn", "cheese"), (224, 196, 48), "smooth", 150.0),
    (("carrot", "orange", "basketball"), (224, 120, 36), "smooth", 150.0),
    (("tomato", "apple", "strawberry", "pepper", "button_plus", "meat", "steak"), (196, 48, 40), "smooth", 150.0),
    (("coffee", "chocolate", "spam", "tuna", "soup", "can", "cracker", "sugar", "jello", "bottle",
      "cup", "mug", "bowl", "plate", "grocery", "food"), (170, 120, 70), "speckle", 200.0),
    (("sponge", "cloth", "towel", "rag", "cushion", "pillow", "rug", "mat", "shoe", "sock",
      "hat", "glove", "umbrella"), (228, 200, 60), "fabric", 60.0),
    (("button_minus",), (40, 150, 70), "smooth", 200.0),
    (("button_power", "button"), (40, 90, 200), "smooth", 200.0),
]


def _is_visual(name: str) -> bool:
    n = name.lower()
    return n.endswith("_visual") or n.endswith("_vis") or "_visual_" in n or "_vis_" in n


def classify(name: str, structural: set[str]) -> tuple[str, str]:
    """Return (type, uv) for an object name. ``structural`` is the set of objects
    that are the PARENT of some non-visual object (i.e. bases/stands that hold
    other things) - those are static, so they bake as kinematic."""
    n = name.lower()
    if _is_visual(name):
        return "visual", "none"
    kinematic_markers = ("_respondable", "_resp", "_root", "_frame", "_base", "_stand", "_holder",
                         "_plane", "_stop", "_wrap", "topplate")
    if (name in structural
            or any(n.endswith(m) or m in n for m in kinematic_markers) or "boundary" in n
            or "cupboard" in n or "fridge" in n or "microwave" in n or n in ("table", "floor")):
        return "kinematic", "auto"
    return "rigid", "auto"


def appearance_for_name(name: str) -> tuple[tuple[int, int, int], str, float]:
    n = name.lower()
    for keywords, color, style, density in PALETTE:
        if any(k in n for k in keywords):
            return color, style, density
    # Fallback: stable hashed hue so each object gets a distinct, repeatable color.
    h = int(hashlib.md5(name.encode()).hexdigest(), 16)
    r, g, b = colorsys.hsv_to_rgb((h % 360) / 360.0, 0.45, 0.75)
    return (int(r * 255), int(g * 255), int(b * 255)), "speckle", 80.0


def make_texture(color: tuple[int, int, int], style: str, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    base = np.array(color, float)
    img = np.ones((TEX_SIZE, TEX_SIZE, 3)) * base

    if style == "wood":
        for _ in range(40):
            x = rng.integers(0, TEX_SIZE)
            img[:, max(0, x - 1):x + 2, :] += rng.normal(0, 14)
        img += rng.normal(0, 6, img.shape)
    elif style == "metal":
        img += np.sin(np.linspace(0, 30 * np.pi, TEX_SIZE))[None, :, None] * 8
        img += rng.normal(0, 5, img.shape)
    elif style == "fabric":
        out = Image.fromarray(np.clip(img + rng.normal(0, 9, img.shape), 0, 255).astype("uint8"))
        d = ImageDraw.Draw(out)
        for _ in range(120):
            x, y = rng.integers(0, TEX_SIZE, 2)
            d.point((int(x), int(y)), fill=tuple(int(c * 0.8) for c in color))
        return out.filter(ImageFilter.GaussianBlur(0.5))
    elif style == "speckle":
        img += rng.normal(0, 12, img.shape)
    else:  # smooth
        img += rng.normal(0, 4, img.shape)

    return Image.fromarray(np.clip(img, 0, 255).astype("uint8"))


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def main():
    physics = _load(PHYSICS_CONFIG)
    appearance = _load(APPEARANCE_CONFIG)

    reports = sorted(REPORTS_DIR.glob("*.scene_context.json"))
    print(f"[INFO]: {len(reports)} scene reports found.")

    added_tasks = 0
    tex_count = 0
    for report_path in reports:
        task = report_path.name.removesuffix(".scene_context.json")
        # Never touch hand-curated blocks.
        if task in CURATED:
            continue
        data = json.loads(report_path.read_text(encoding="utf-8"))
        entries = [o for o in data.get("objects", []) if o.get("name")]
        objects = [o["name"] for o in entries]
        if not objects:
            continue

        # Structural objects = parents of any NON-visual object (bases/stands that
        # hold things). These are static -> kinematic, so they don't fall/eject.
        object_names = set(objects)
        structural = {
            o["parent"]
            for o in entries
            if not _is_visual(o["name"]) and o.get("parent") in object_names
        }

        task_tex_dir = TEXTURES_DIR / task
        task_tex_dir.mkdir(parents=True, exist_ok=True)

        physics_block: dict = {}
        appearance_block: dict = {}
        for index, name in enumerate(objects):
            obj_type, uv = classify(name, structural)
            color, style, density = appearance_for_name(name)

            tex_rel = f"textures/{task}/{name}.png"
            tex_path = HERE / tex_rel
            if not tex_path.is_file():  # reuse existing textures on re-run
                make_texture(color, style, seed=hash((task, name)) & 0xFFFF).save(tex_path)
                tex_count += 1

            if obj_type == "visual":
                physics_block[name] = {"type": "visual"}
            elif obj_type == "kinematic":
                physics_block[name] = {"type": "kinematic", "uv": uv}
            else:
                physics_block[name] = {"type": "rigid", "density": density, "uv": uv}

            appearance_block[name] = {
                "texture": tex_rel,
                "color": [round(c / 255.0, 3) for c in color],
                "roughness": 0.85 if style in ("fabric", "speckle") else 0.5,
                "metallic": 0.6 if style == "metal" else 0.0,
            }

        physics[task] = physics_block  # overwrite auto blocks so re-runs pick up rule changes
        appearance[task] = appearance_block
        added_tasks += 1

    PHYSICS_CONFIG.write_text(json.dumps(physics, indent=2) + "\n", encoding="utf-8")
    APPEARANCE_CONFIG.write_text(json.dumps(appearance, indent=2) + "\n", encoding="utf-8")
    print(f"[INFO]: Added {added_tasks} task block(s); wrote {tex_count} textures under {TEXTURES_DIR}.")
    print(f"[INFO]: Updated {PHYSICS_CONFIG.name} and {APPEARANCE_CONFIG.name}.")


if __name__ == "__main__":
    main()
