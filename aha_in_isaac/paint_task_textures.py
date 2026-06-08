"""
paint_task_textures.py

Category-aware diffuse textures for EVERY task, replacing the flat 128px
procedural noise from ``generate_task_configs.py`` with realistic materials
rendered at each object's true face aspect ratio.

For each (task, object) in ``object_appearance_config.json`` it:
  * reads the object's base colour straight from that config (so colours stay
    consistent with what the scene already uses),
  * picks a material category from the object name (wood / brushed metal /
    ceramic glaze / plastic / woven fabric / glass / screen / organic / paper /
    stone / button / boundary),
  * sizes the image to the object's real aspect ratio (from
    ``textures/_usd_extents.json``; square if unknown), and
  * paints it supersampled (crisp) and overwrites ``textures/<task>/<obj>.png``.

It does NOT touch the appearance config, the shared ``textures/wood.png`` etc.,
or ``change_channel`` (already hand-painted by paint_change_channel_textures.py).
So a later ``generate_task_configs.py`` run keeps these PNGs (it only writes a
texture when the file is missing).

Run with the Pillow env (env_isaacsim51):
    python scripts/aha_in_isaac/paint_task_textures.py [task ...]
With no args it paints all tasks; pass task names to repaint just those.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from generate_task_configs import _is_visual, appearance_for_name

HERE = Path(__file__).resolve().parent
TEX = HERE / "textures"
CONFIG = HERE / "object_appearance_config.json"
EXTENTS = TEX / "_usd_extents.json"
SKIP_TASKS = {"change_channel"}     # already bespoke
SS = 2                              # supersample factor


# ---------------------------------------------------------------------------
# helpers (float HxWx3 in 0..255)
# ---------------------------------------------------------------------------
def _seed(task, obj):
    return abs(hash((task, obj))) & 0x7FFFFFFF


def vgrad(w, h, top, bot):
    t = np.linspace(0.0, 1.0, h)[:, None, None]
    return np.array(top, float) * (1 - t) + np.array(bot, float) * t + np.zeros((h, w, 1))


def radial(w, h, inner, outer, power=1.0, cx=0.5, cy=0.5):
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    nx = (xx - cx * (w - 1)) / max(1.0, (w - 1) * max(cx, 1 - cx))
    ny = (yy - cy * (h - 1)) / max(1.0, (h - 1) * max(cy, 1 - cy))
    r = np.clip(np.sqrt(nx * nx + ny * ny) / np.sqrt(2.0), 0, 1)[..., None] ** power
    return np.array(inner, float) * (1 - r) + np.array(outer, float) * r


def noise(img, rng, sigma):
    return np.clip(img + rng.normal(0.0, sigma, img.shape), 0, 255)


def shade(color, f):
    return tuple(int(np.clip(c * f, 0, 255)) for c in color)


def to_big(arr, w, h):
    return Image.fromarray(np.clip(arr, 0, 255).astype("uint8")).convert("RGBA").resize(
        (w * SS, h * SS), Image.LANCZOS)


# ---------------------------------------------------------------------------
# material painters: (w, h, color, rng) -> RGBA PIL image at (w*SS, h*SS)
# ---------------------------------------------------------------------------
def p_wood(w, h, color, rng):
    # Grain runs along the long (u/width) axis -> horizontal streaks stacked in y.
    base = vgrad(w, h, shade(color, 1.08), shade(color, 0.86))
    for _ in range(max(18, h // 3)):
        y = rng.integers(0, h)
        base[max(0, y - 1):y + 1, :, :] += rng.normal(0, 16)
    base = noise(base, rng, 5)
    big = to_big(base, w, h)
    d = ImageDraw.Draw(big, "RGBA")
    W, H = w * SS, h * SS
    for _ in range(max(1, H // (W // 2 + 1))):           # a couple of plank seams
        y = int(rng.integers(0, H))
        d.line([(0, y), (W, y)], fill=shade(color, 0.55) + (130,), width=max(1, H // 120))
    return big


def p_metal(w, h, color, rng):
    # Brushed streaks along length + a soft vertical specular band.
    base = np.ones((h, w, 3)) * np.array(color, float)
    base += np.sin(np.linspace(0, 24 * np.pi, w))[None, :, None] * 4
    for _ in range(h):                                   # fine horizontal brushing
        base[rng.integers(0, h), :, :] += rng.normal(0, 6)
    big = to_big(noise(base, rng, 3), w, h)
    W, H = w * SS, h * SS
    spec = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(spec).rectangle([int(0.30 * W), 0, int(0.46 * W), H], fill=(255, 255, 255, 60))
    big.alpha_composite(spec.filter(ImageFilter.GaussianBlur(W * 0.06)))
    return big


def p_plastic(w, h, color, rng):
    base = radial(w, h, shade(color, 1.12), shade(color, 0.82), power=1.3, cy=0.38)
    big = to_big(noise(base, rng, 2.5), w, h)
    W, H = w * SS, h * SS
    spec = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(spec).ellipse([int(0.22 * W), int(0.10 * H), int(0.60 * W), int(0.42 * H)],
                                 fill=(255, 255, 255, 70))
    big.alpha_composite(spec.filter(ImageFilter.GaussianBlur(min(W, H) * 0.05)))
    return big


def p_ceramic(w, h, color, rng):
    glaze = shade(color, 1.0) if sum(color) < 690 else (236, 236, 240)
    base = radial(w, h, shade(glaze, 1.10), shade(glaze, 0.86), power=1.5, cy=0.40)
    big = to_big(noise(base, rng, 1.6), w, h)
    W, H = w * SS, h * SS
    spec = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(spec).ellipse([int(0.26 * W), int(0.12 * H), int(0.56 * W), int(0.40 * H)],
                                 fill=(255, 255, 255, 110))
    big.alpha_composite(spec.filter(ImageFilter.GaussianBlur(min(W, H) * 0.045)))
    return big


def p_fabric(w, h, color, rng):
    base = noise(np.ones((h, w, 3)) * np.array(color, float), rng, 8)
    big = to_big(base, w, h)
    W, H = w * SS, h * SS
    d = ImageDraw.Draw(big, "RGBA")
    step = max(3, min(W, H) // 26)
    for x in range(0, W, step):
        d.line([(x, 0), (x, H)], fill=shade(color, 0.86) + (70,), width=1)
    for y in range(0, H, step):
        d.line([(0, y), (W, y)], fill=shade(color, 1.12) + (60,), width=1)
    return big.filter(ImageFilter.GaussianBlur(SS * 0.4))


def p_organic(w, h, color, rng):
    base = np.ones((h, w, 3)) * np.array(color, float)
    big = to_big(noise(base, rng, 6), w, h)
    W, H = w * SS, h * SS
    blob = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    bd = ImageDraw.Draw(blob)
    for _ in range(8):                                   # soft ripening blotches
        cx, cy = int(rng.integers(0, W)), int(rng.integers(0, H))
        r = int(rng.integers(W // 10, W // 4))
        f = 0.80 if rng.random() < 0.5 else 1.18
        bd.ellipse([cx - r, cy - r, cx + r, cy + r], fill=shade(color, f) + (60,))
    big.alpha_composite(blob.filter(ImageFilter.GaussianBlur(W * 0.04)))
    return big


def p_glass(w, h, color, rng):
    tint = (206, 220, 232) if sum(color) > 600 else shade(color, 1.2)
    base = radial(w, h, shade(tint, 1.06), shade(tint, 0.7), power=1.6)
    big = to_big(noise(base, rng, 1.2), w, h)
    W, H = w * SS, h * SS
    refl = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    rd = ImageDraw.Draw(refl)
    rd.polygon([(0, int(0.5 * H)), (int(0.45 * W), 0), (int(0.62 * W), 0),
                (int(0.12 * W), H), (0, H)], fill=(255, 255, 255, 50))
    big.alpha_composite(refl.filter(ImageFilter.GaussianBlur(W * 0.02)))
    return big


def p_screen(w, h, color, rng):
    base = radial(w, h, (16, 26, 52), (4, 6, 16), power=1.3)
    big = to_big(noise(base, rng, 2.0), w, h)
    W, H = w * SS, h * SS
    refl = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    rd = ImageDraw.Draw(refl)
    rd.polygon([(0, int(0.55 * H)), (int(0.42 * W), 0), (int(0.70 * W), 0),
                (int(0.18 * W), H), (0, H)], fill=(255, 255, 255, 26))
    big.alpha_composite(refl.filter(ImageFilter.GaussianBlur(H * 0.02)))
    ImageDraw.Draw(big, "RGBA").rectangle([1, 1, W - 2, H - 2], outline=(8, 8, 10, 255),
                                          width=max(1, int(0.02 * min(W, H))))
    return big


def p_paper(w, h, color, rng):
    paper = (228, 220, 200) if sum(color) > 540 else shade(color, 1.1)
    big = to_big(noise(np.ones((h, w, 3)) * np.array(paper, float), rng, 4), w, h)
    W, H = w * SS, h * SS
    d = ImageDraw.Draw(big, "RGBA")
    for i in range(6, H, max(8, H // 12)):               # faint ruled lines
        d.line([(int(0.08 * W), i), (int(0.92 * W), i)], fill=shade(paper, 0.8) + (60,), width=1)
    return big


def p_stone(w, h, color, rng):
    grey = (150, 148, 145) if abs(max(color) - min(color)) < 30 and sum(color) > 480 else color
    base = noise(np.ones((h, w, 3)) * np.array(grey, float), rng, 10)
    big = to_big(base, w, h)
    W, H = w * SS, h * SS
    sp = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    spd = ImageDraw.Draw(sp)
    for _ in range(W // 6):
        x, y = int(rng.integers(0, W)), int(rng.integers(0, H))
        spd.point((x, y), fill=shade(grey, 0.7) + (120,))
    big.alpha_composite(sp)
    return big


def p_button(w, h, color, rng):
    base = radial(w, h, shade(color, 1.2), shade(color, 0.7), power=1.4)
    big = to_big(noise(base, rng, 2), w, h)
    W, H = w * SS, h * SS
    d = ImageDraw.Draw(big, "RGBA")
    d.ellipse([int(0.06 * W), int(0.06 * H), int(0.94 * W), int(0.94 * H)],
              outline=shade(color, 0.5) + (150,), width=max(1, int(0.04 * min(W, H))))
    spec = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(spec).ellipse([int(0.26 * W), int(0.16 * H), int(0.58 * W), int(0.42 * H)],
                                 fill=(255, 255, 255, 90))
    big.alpha_composite(spec.filter(ImageFilter.GaussianBlur(min(W, H) * 0.05)))
    return big


def p_boundary(w, h, color, rng):
    big = to_big(noise(np.ones((h, w, 3)) * np.array((204, 206, 210), float), rng, 1.5), w, h)
    W, H = w * SS, h * SS
    ImageDraw.Draw(big, "RGBA").rectangle([2, 2, W - 3, H - 3], outline=(90, 170, 110, 200),
                                          width=max(2, int(0.012 * min(W, H))))
    return big


PAINTERS = {
    "wood": p_wood, "metal": p_metal, "plastic": p_plastic, "ceramic": p_ceramic,
    "fabric": p_fabric, "organic": p_organic, "glass": p_glass, "screen": p_screen,
    "paper": p_paper, "stone": p_stone, "button": p_button, "boundary": p_boundary,
}


# ---------------------------------------------------------------------------
# categorisation
# ---------------------------------------------------------------------------
# Whole-token keyword sets (matched against the name's underscore tokens with
# trailing digits stripped), so "cup" != "cupboard", "plate" != "topplate",
# "dish" != "dishwasher". First category wins; order matters.
SEMANTIC = [
    ("boundary", {"boundary", "spawn"}),
    ("glass", {"glass", "window", "mirror", "lens", "bulb", "windshield", "pane", "windowpane",
               "bottle", "jar", "flask", "decanter", "wineglass", "tumbler"}),
    ("ceramic", {"plate", "saucer", "bowl", "cup", "mug", "dish", "teapot", "toilet", "urinal",
                 "sink", "basin", "bathtub", "porcelain", "ceramic", "vase", "jug", "crockery"}),
    ("paper", {"paper", "money", "bill", "banknote", "card", "page", "note", "newspaper",
               "cardboard", "envelope", "ticket", "letter"}),
    ("stone", {"wall", "concrete", "brick", "plaster", "marble", "granite", "tile", "stone", "floor"}),
    ("button", {"button", "switch", "knob", "dial", "keycap"}),
    ("screen", {"screen", "display", "monitor", "television"}),
]


def _tokens(name: str) -> set[str]:
    out = set()
    for t in name.lower().removesuffix("_visual").removesuffix("_vis").split("_"):
        t = t.rstrip("0123456789")
        if t:
            out.add(t)
    return out


def category_for(name: str, style: str) -> str:
    toks = _tokens(name)
    if "tv" in toks:                                 # tv_frame -> screen, tv_remote -> plastic
        return "screen" if "remote" not in toks else "plastic"
    for cat, keys in SEMANTIC:
        if toks & keys:
            return cat
    return {"smooth": "plastic", "speckle": "organic"}.get(style, style)  # wood/metal/fabric pass through


# ---------------------------------------------------------------------------
def res_and_aspect(ext):
    """(width, height) for the texture from an extent [ex,ey,ez] cm (or None)."""
    if not ext:
        return 512, 512
    a, b, _ = sorted(ext, reverse=True)
    longest = max(ext)
    long_px = 1024 if longest > 40 else 768 if longest > 15 else 512 if longest > 5 else 256
    if b < 1e-6 or a / b < 1.25:
        return long_px, long_px
    h = max(96, long_px // 8, round(long_px * b / a))
    return long_px, int(h)


def main():
    cfg = json.loads(CONFIG.read_text())
    ext_cache = json.loads(EXTENTS.read_text()) if EXTENTS.is_file() else {}
    only = set(sys.argv[1:])
    tasks = [t for t in cfg if not t.startswith("_") and t not in SKIP_TASKS]
    if only:
        tasks = [t for t in tasks if t in only]

    counts: dict[str, int] = {}
    total = 0
    for task in tasks:
        task_ext = ext_cache.get(task, {})
        for obj, spec in cfg[task].items():
            if obj.startswith("_"):
                continue
            tex_rel = spec.get("texture")
            if not tex_rel or not tex_rel.startswith(f"textures/{task}/"):
                continue
            color = tuple(int(round(c * 255)) for c in spec.get("color", [0.8, 0.8, 0.8]))
            _, style, _ = appearance_for_name(obj)
            cat = category_for(obj, style)
            ext = task_ext.get(obj.lower())
            w, h = res_and_aspect(ext)
            rng = np.random.default_rng(_seed(task, obj))
            img = PAINTERS[cat](w, h, color, rng).resize((w, h), Image.LANCZOS).convert("RGB")
            out = HERE / tex_rel
            out.parent.mkdir(parents=True, exist_ok=True)
            img.save(out)
            counts[cat] = counts.get(cat, 0) + 1
            total += 1
        print(f"  {task:42s} done")
    print(f"\nPainted {total} textures across {len(tasks)} tasks.")
    print("by category:", dict(sorted(counts.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
