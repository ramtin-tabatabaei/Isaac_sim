"""
paint_change_channel_textures.py

Hand-painted, object-specific diffuse textures for the ``change_channel`` task,
replacing the generic procedural noise produced by ``generate_task_configs.py``.

Each texture is rendered at the *true aspect ratio* of the face it maps onto, so
nothing looks stretched. The runtime UV (``usd_uv.generate_uvs``) projects a
planar/box object onto its two largest-extent axes and normalises each to 0..1,
so a square texture on a long thin remote would smear 4x. The aspect ratios below
were measured from the baked USDs (extent in cm, "UV axes" = the two largest):

    tv_remote                21.6 x 5.12   X x Y   -> 4.22 : 1  (long thin slab)
    tv_frame                 61.6 x 39.5   Y x Z   -> 1.56 : 1  (~16:10 screen)
    target_button_topPlate*   3.31 x 3.31  X x Y   -> 1 : 1     (round cap top)
    target_button_wrap*       3.31 x 3.31  X x Y   -> 1 : 1     (side collar)
    plus/minus/power_visual    2.1 x 2.1   X x Y   -> 1 : 1     (glyph decal)
    spawn_boundary           45.5 x 32.5   Y x X   -> 1.40 : 1  (debug overlay)

Run with the env that has Pillow (env_isaacsim51):
    python scripts/aha_in_isaac/paint_change_channel_textures.py

Shapes/glyphs are drawn supersampled then downsampled (LANCZOS) for crisp edges.
Specular sheen and shading are baked into the image so the look survives even if
the appearance config's roughness is ever reset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

OUT = Path(__file__).with_name("textures") / "change_channel"
SS = 3  # supersample factor for crisp shapes/glyphs

# Deterministic per-object noise so re-runs are byte-stable.
SEEDS = {
    "tv_remote": 11, "tv_frame": 22,
    "target_button_topPlate0": 30, "target_button_topPlate1": 31, "target_button_topPlate2": 32,
    "target_button_wrap0": 40, "target_button_wrap1": 41, "target_button_wrap2": 42,
    "plus_visual": 50, "minus_visual": 51, "power_visual": 52,
    "spawn_boundary": 60,
}


# ---------------------------------------------------------------------------
# numpy gradient / noise helpers (work in float, return uint8 HxWx3)
# ---------------------------------------------------------------------------
def vgrad(w, h, top, bot):
    """Vertical gradient from ``top`` (row 0) to ``bot`` (last row)."""
    t = np.linspace(0.0, 1.0, h)[:, None, None]
    top = np.array(top, float); bot = np.array(bot, float)
    return (top * (1 - t) + bot * t) * np.ones((h, w, 1))


def add_noise(img, rng, sigma):
    return np.clip(img + rng.normal(0.0, sigma, img.shape), 0, 255)


def radial(w, h, inner, outer, power=1.0):
    """Radial gradient: ``inner`` at the centre fading to ``outer`` at the corner."""
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    r = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
    r = np.clip(r / np.sqrt(2.0), 0, 1)[..., None] ** power
    inner = np.array(inner, float); outer = np.array(outer, float)
    return inner * (1 - r) + outer * r


def to_big(arr_uint8, w, h):
    """uint8 HxWx3 -> supersampled RGBA PIL canvas for drawing/compositing."""
    pim = Image.fromarray(arr_uint8.astype("uint8")).convert("RGBA")
    return pim.resize((w * SS, h * SS), Image.LANCZOS)


def finish(big, w, h, path):
    big.resize((w, h), Image.LANCZOS).convert("RGB").save(path)
    print(f"  wrote {path.name:32s} {w}x{h}")


def rrect(d, box, radius, **kw):
    d.rounded_rectangle(box, radius=radius, **kw)


# ---------------------------------------------------------------------------
# Per-object painters
# ---------------------------------------------------------------------------
def remote_body(w, h, rng):
    """Sleek dark-plastic remote body. The 3 real buttons (+/-/power) are separate
    meshes that sit on top, so the body stays clean: vertical sheen + rounded-edge
    vignette + a thin brushed pinstripe."""
    img = vgrad(w, h, (60, 63, 72), (32, 33, 39))      # plastic, lit from top
    img = add_noise(img, rng, 2.5)
    # Darken the two long edges so the slab reads as rounded.
    v = np.abs(np.linspace(-1, 1, h))[:, None, None] ** 2.2
    img = img * (1 - 0.30 * v)
    big = to_big(img, w, h)
    W, H = w * SS, h * SS
    d = ImageDraw.Draw(big, "RGBA")
    # Soft specular sheen band near the top edge.
    sheen = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(sheen)
    sd.rounded_rectangle([int(0.02 * W), int(0.10 * H), int(0.98 * W), int(0.34 * H)],
                         radius=int(0.10 * H), fill=(255, 255, 255, 34))
    big.alpha_composite(sheen.filter(ImageFilter.GaussianBlur(H * 0.04)))
    # Thin brushed-metal pinstripe accent low on the body.
    d.line([(int(0.04 * W), int(0.74 * H)), (int(0.96 * W), int(0.74 * H))],
           fill=(150, 154, 165, 120), width=max(1, int(0.012 * H)))
    return big


def tv_screen(w, h, rng):
    """Flat-screen TV: black bezel + a faintly-lit dark screen with a soft diagonal
    reflection, vignette, and a small standby LED."""
    bezel = (16, 16, 19)
    img = np.ones((h, w, 3)) * np.array(bezel, float)
    img = add_noise(img, rng, 2.0)
    big = to_big(img, w, h)
    W, H = w * SS, h * SS
    d = ImageDraw.Draw(big, "RGBA")
    # Bezel outer rounding highlight.
    rrect(d, [2, 2, W - 3, H - 3], radius=int(0.045 * H), outline=(60, 62, 70, 180),
          width=max(1, int(0.006 * H)))
    # Screen rectangle (inset bezel).
    mx, my = int(0.075 * W), int(0.10 * H)
    sx0, sy0, sx1, sy1 = mx, my, W - mx, H - my
    sw, sh = sx1 - sx0, sy1 - sy0
    scr = radial(sw, sh, (16, 26, 52), (4, 6, 16), power=1.3)   # cool glow -> dark edges
    scr_img = Image.fromarray(scr.astype("uint8")).convert("RGBA")
    big.paste(scr_img, (sx0, sy0))
    # Diagonal reflection streak on the glass.
    refl = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    rd = ImageDraw.Draw(refl)
    rd.polygon([(sx0, sy0 + int(0.55 * sh)), (sx0 + int(0.42 * sw), sy0),
                (sx0 + int(0.70 * sw), sy0), (sx0 + int(0.18 * sw), sy0 + sh),
                (sx0, sy0 + sh)], fill=(255, 255, 255, 26))
    big.alpha_composite(refl.filter(ImageFilter.GaussianBlur(H * 0.02)))
    # Inner bezel lip (thin dark frame around the screen).
    rrect(d, [sx0 - 2, sy0 - 2, sx1 + 1, sy1 + 1], radius=int(0.02 * H),
          outline=(8, 8, 10, 255), width=max(1, int(0.010 * H)))
    # Standby LED, lower-centre of the bezel.
    lx, ly, lr = W // 2, H - int(0.045 * H), max(2, int(0.010 * H))
    d.ellipse([lx - lr, ly - lr, lx + lr, ly + lr], fill=(220, 40, 40, 255))
    big.alpha_composite(_glow(W, H, (lx, ly), lr * 4, (220, 40, 40, 120)))
    return big


def button_cap(w, h, rng):
    """Light silver plastic cap with a soft radial highlight and bevel ring, so it
    reads as a physical round button. Identity comes from the glyph decal on top."""
    img = radial(w, h, (228, 230, 236), (168, 171, 180), power=1.4)
    img = add_noise(img, rng, 2.0)
    big = to_big(img, w, h)
    W, H = w * SS, h * SS
    d = ImageDraw.Draw(big, "RGBA")
    # Bevel ring + outer contact shadow.
    d.ellipse([int(0.05 * W), int(0.05 * H), int(0.95 * W), int(0.95 * H)],
              outline=(120, 123, 132, 160), width=max(1, int(0.025 * H)))
    spec = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sd = ImageDraw.Draw(spec)
    sd.ellipse([int(0.24 * W), int(0.16 * H), int(0.62 * W), int(0.44 * H)],
               fill=(255, 255, 255, 90))
    big.alpha_composite(spec.filter(ImageFilter.GaussianBlur(H * 0.05)))
    return big


def button_wrap(w, h, rng):
    """Dark plastic side collar. The box UV smears this onto the ring's wall, so a
    near-solid charcoal with faint shading is the robust choice."""
    img = radial(w, h, (66, 68, 76), (40, 41, 47), power=1.2)
    img = add_noise(img, rng, 3.0)
    return to_big(img, w, h)


def _glow(W, H, center, radius, rgba):
    g = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(g)
    cx, cy = center
    gd.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=rgba)
    return g.filter(ImageFilter.GaussianBlur(radius * 0.6))


def glyph_decal(w, h, rng, kind, color):
    """A printed +/-/power symbol on a neutral cap-coloured ground, so the decal
    sits flush on the button cap and only the coloured symbol reads."""
    img = radial(w, h, (222, 224, 229), (198, 200, 206), power=1.3)
    img = add_noise(img, rng, 1.5)
    big = to_big(img, w, h)
    W, H = w * SS, h * SS
    S = min(W, H)
    cx, cy = W / 2.0, H / 2.0
    t = 0.15 * S          # bar thickness
    L = 0.30 * S          # bar half-length
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sh = ImageDraw.Draw(shadow)
    glyph = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glyph)
    col = tuple(color) + (255,)
    off = 0.018 * S       # emboss shadow offset

    def hbar(drw, dx, dy, c):
        drw.rounded_rectangle([cx - L + dx, cy - t / 2 + dy, cx + L + dx, cy + t / 2 + dy],
                              radius=t / 2, fill=c)

    def vbar(drw, dx, dy, c):
        drw.rounded_rectangle([cx - t / 2 + dx, cy - L + dy, cx + t / 2 + dx, cy + L + dy],
                              radius=t / 2, fill=c)

    if kind == "minus":
        hbar(sh, off, off, (0, 0, 0, 70)); hbar(gd, 0, 0, col)
    elif kind == "plus":
        hbar(sh, off, off, (0, 0, 0, 70)); vbar(sh, off, off, (0, 0, 0, 70))
        hbar(gd, 0, 0, col); vbar(gd, 0, 0, col)
    elif kind == "power":
        r = 0.30 * S
        lw = int(0.13 * S)
        # Broken ring (gap at top) + vertical stem through the gap = IEC power mark.
        for drw, c, dx, dy in ((sh, (0, 0, 0, 70), off, off), (gd, col, 0, 0)):
            drw.arc([cx - r + dx, cy - r + dy, cx + r + dx, cy + r + dy],
                    start=292, end=248, fill=c, width=lw)
            drw.rounded_rectangle([cx - lw / 2 + dx, cy - r - 0.10 * S + dy,
                                   cx + lw / 2 + dx, cy + 0.04 * S + dy],
                                  radius=lw / 2, fill=c)
    big.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(S * 0.012)))
    big.alpha_composite(glyph)
    return big


def spawn_boundary(w, h, rng):
    """Unobtrusive debug overlay: faint light fill + a thin soft-green border."""
    img = np.ones((h, w, 3)) * np.array((204, 206, 210), float)
    img = add_noise(img, rng, 1.5)
    big = to_big(img, w, h)
    W, H = w * SS, h * SS
    d = ImageDraw.Draw(big, "RGBA")
    d.rectangle([2, 2, W - 3, H - 3], outline=(90, 170, 110, 200), width=max(2, int(0.012 * H)))
    return big


# ---------------------------------------------------------------------------
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = lambda name: np.random.default_rng(SEEDS[name])  # noqa: E731
    print(f"Painting change_channel textures -> {OUT}")

    finish(remote_body(1024, 243, rng("tv_remote")), 1024, 243, OUT / "tv_remote.png")
    finish(tv_screen(1024, 656, rng("tv_frame")), 1024, 656, OUT / "tv_frame.png")

    for i in range(3):
        n = f"target_button_topPlate{i}"
        finish(button_cap(512, 512, rng(n)), 512, 512, OUT / f"{n}.png")
        n = f"target_button_wrap{i}"
        finish(button_wrap(512, 512, rng(n)), 512, 512, OUT / f"{n}.png")

    finish(glyph_decal(512, 512, rng("plus_visual"), "plus", (40, 96, 205)),
           512, 512, OUT / "plus_visual.png")
    finish(glyph_decal(512, 512, rng("minus_visual"), "minus", (40, 150, 78)),
           512, 512, OUT / "minus_visual.png")
    finish(glyph_decal(512, 512, rng("power_visual"), "power", (205, 52, 48)),
           512, 512, OUT / "power_visual.png")

    finish(spawn_boundary(1024, 731, rng("spawn_boundary")), 1024, 731, OUT / "spawn_boundary.png")
    print("Done.")


if __name__ == "__main__":
    main()
