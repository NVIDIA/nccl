# SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regenerate the logo (needs matplotlib, for the wordmark's outlines): python3 docs/figures/logo.py

An 8x8 grid of tiles holding a two-layer tide, next to "TIDE" in DejaVu Sans Mono Bold, the font matplotlib
ships, so every host draws the same outlines. CSS in the SVG moves the waves and the tide; where it does not
run, or reduced motion is requested, the picture stays at its first frame.
"""
import math
import pathlib

import matplotlib
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path
from matplotlib.textpath import TextPath

HERE = pathlib.Path(__file__).resolve().parent
FONT = pathlib.Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSansMono-Bold.ttf"
DRY, BACK, FRONT, INK = "#e8eefc", "#9ec5fe", "#1f5fbf", "#1b1f24"
N, GAP = 8, 0.1                    # tiles per side; gap as a fraction of the tile pitch
S, PAD = 100.0, 12.0               # grid side (one wavelength) and margin, in viewBox units
LEVEL, AMP, LIFT = 0.45, 1.1, 0.6  # mean tide as a fraction of S; amplitude and back-wave lift in tile pitches
PHASE, BACK_LAG = math.pi / 2, 0.35
LOOP_S, SWING = 6.0, 0.2           # one rise and fall of the tide; its swing as a fraction of S
CMD = {Path.MOVETO: "M", Path.LINETO: "L", Path.CURVE3: "Q", Path.CURVE4: "C"}

# The tide starts a quarter period in (mid level, rising), so the first frame is the still picture.
STYLE = (
    "@keyframes drift{to{transform:translateX(%(S)gpx)}}"
    "@keyframes tide{0%%,100%%{transform:translateY(%(swing)gpx)}50%%{transform:translateY(-%(swing)gpx)}}"
    ".front{animation:drift %(front)gs linear infinite}.back{animation:drift %(back)gs linear infinite}"
    ".tide{animation:tide %(loop)gs cubic-bezier(.37,0,.63,1) -%(quarter)gs infinite}"
    "@media (prefers-reduced-motion:reduce){.front,.back,.tide{animation:none}}"
) % dict(S=S, swing=SWING * S, front=LOOP_S / 2, back=LOOP_S, loop=LOOP_S, quarter=LOOP_S / 4)


def grid():
    u = S / (N - GAP)
    side = u * (1 - GAP)
    tiles = "".join(f'<rect x="{PAD + c * u:.2f}" y="{PAD + r * u:.2f}" width="{side:.2f}" height="{side:.2f}"/>'
                    for r in range(N) for c in range(N))
    bottom = PAD + S * (1 + SWING) + u  # still below the grid at high tide

    def water(y, phase):
        # Two wavelengths from PAD - S: shifted right by one wavelength, the wave looks the same.
        pts = " ".join(f"{PAD + x:.2f},{y - AMP * u * math.sin(2 * math.pi * x / S - phase):.2f}"
                       for x in (S * (i / 100 - 1) for i in range(201)))
        return f"M{PAD - S:.2f},{bottom:.2f} L{pts} L{PAD + S:.2f},{bottom:.2f}Z"

    y0 = PAD + S * (1 - LEVEL)
    return (f'<defs><clipPath id="tide-tiles">{tiles}</clipPath></defs><g clip-path="url(#tide-tiles)">'
            f'<rect x="{PAD:.2f}" y="{PAD:.2f}" width="{S:.2f}" height="{S:.2f}" fill="{DRY}"/><g class="tide">'
            f'<path class="back" d="{water(y0 - LIFT * u, PHASE + 2 * math.pi * BACK_LAG)}" fill="{BACK}"/>'
            f'<path class="front" d="{water(y0, PHASE)}" fill="{FRONT}"/></g></g>')


def wordmark(x, cap):
    """The word TIDE as outlines, cap height `cap` centred on the grid; returns the path data and right ink edge."""
    size = cap / TextPath((0, 0), "H", size=1, prop=FontProperties(fname=FONT)).get_extents().height
    prop = FontProperties(fname=FONT, size=size)

    def extent(s):
        return TextPath((0, 0), s, prop=prop).get_extents()

    step = extent("MM").x1 - extent("M").x1 + 0.04 * size
    base = PAD + S / 2 + cap / 2
    d = []
    for i, ch in enumerate("TIDE"):
        for verts, code in TextPath((0, 0), ch, prop=prop).iter_segments(curves=True, simplify=False):
            if code == Path.CLOSEPOLY:
                d.append("Z")
            else:
                d.append(CMD[code] + " ".join(f"{x + i * step + verts[j]:.2f},{base - verts[j + 1]:.2f}"
                                              for j in range(0, len(verts), 2)))
    return "".join(d), x + 3 * step + extent("E").x1


def main():
    d, right = wordmark(PAD + S + 0.2 * S, 0.6 * S)
    w, h = right + PAD, S + 2 * PAD
    # White like the other figures, so the dark wordmark stays readable in dark mode.
    (HERE / "logo.svg").write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.2f} {h:.2f}" width="{w:.0f}" height="{h:.0f}" '
        f'role="img" aria-label="TIDE"><title>TIDE</title><style>{STYLE}</style>'
        f'<rect width="{w:.2f}" height="{h:.2f}" fill="#fff"/>{grid()}<path d="{d}" fill="{INK}"/></svg>\n')


if __name__ == "__main__":
    main()
