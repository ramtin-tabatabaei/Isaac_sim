"""paint_change_clock_textures.py

Task-specific diffuse texture for the change_clock dial. The generic
``paint_task_textures.py`` paints the clock face as flat green noise (it has no
clock category), which reads as a blank green disc in-sim. This paints a proper
analog clock dial - cream face, dark rim, hour/minute ticks, numerals 1..12 and
a centre hub - onto ``textures/change_clock/clock_visual.png`` (the visible face;
the hour/minute HANDS are separate needle objects, so the dial carries no hands).

Only this task's textures are touched; no shared code or other task is affected.

Run with the Pillow env (env_isaacsim51):
    python scripts/aha_in_isaac/paint_change_clock_textures.py
"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).with_name("textures") / "change_clock"
SS = 4               # supersample factor (drawn big, downsized for crisp edges)
SIZE = 1024          # final square texture size

FACE = (244, 240, 228)     # cream dial
RIM = (38, 38, 42)         # dark case rim
INK = (28, 28, 32)         # ticks + numerals
HUB = (60, 60, 66)         # centre hub


def _font(px: int):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, px)
        except Exception:
            continue
    return ImageFont.load_default()


def paint_dial(path: Path):
    n = SIZE * SS
    img = Image.new("RGB", (n, n), FACE)
    d = ImageDraw.Draw(img)
    c = n / 2
    r = n * 0.47                      # dial radius
    # case rim
    d.ellipse([c - r, c - r, c + r, c + r], outline=RIM, width=int(n * 0.035))
    inner = r * 0.90
    # minute ticks (60) + bold hour ticks (12)
    for i in range(60):
        a = math.radians(i * 6 - 90)
        hour = (i % 5 == 0)
        t_out = inner
        t_in = inner - (n * 0.05 if hour else n * 0.022)
        w = int(n * 0.012 if hour else n * 0.005)
        d.line([c + t_in * math.cos(a), c + t_in * math.sin(a),
                c + t_out * math.cos(a), c + t_out * math.sin(a)], fill=INK, width=w)
    # numerals 1..12
    font = _font(int(n * 0.085))
    num_r = inner * 0.78
    for h in range(1, 13):
        a = math.radians(h * 30 - 90)
        x, y = c + num_r * math.cos(a), c + num_r * math.sin(a)
        s = str(h)
        bb = d.textbbox((0, 0), s, font=font)
        d.text((x - (bb[2] - bb[0]) / 2, y - (bb[3] - bb[1]) / 2 - bb[1]), s, fill=INK, font=font)
    # centre hub
    hub = n * 0.03
    d.ellipse([c - hub, c - hub, c + hub, c + hub], fill=HUB)

    img = img.resize((SIZE, SIZE), Image.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    print(f"[paint] wrote clock dial -> {path}")


if __name__ == "__main__":
    paint_dial(OUT / "clock_visual.png")
