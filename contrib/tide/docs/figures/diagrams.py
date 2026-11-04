# SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regenerate the data-flow diagrams in pure Ink style (standard library only): python3 docs/figures/diagrams.py"""
import pathlib

HERE = pathlib.Path(__file__).resolve().parent

# Standalone SVGs in one style:
# - 11px monospace labels, 12px bold panel titles, sharp 1px #444 boxes, orthogonal connectors
# - Single label <= 8 units (16 ASCII chars), total per figure <= 80 units, zero sentences in SVG
# - Unified role palette across all diagrams:
#     blue / blue-l     : copy engine (CE) & parameter/staging window slots
#     green / green-l   : RDMA / rail NIC & inbox
#     purple            : NVSwitch (multicast & multimem.ld_reduce)
#     orange / orange-l : SM work (compute, FSDP2 glue, TIDE sum/land CTAs) & outbox
#     gray / pad / gone : storage shards, stream wait, removed step
INK_CSS = (
    'svg{background:#fff}'
    'text{font:11px ui-monospace,"SF Mono",Menlo,Consolas,monospace;fill:#1b1f24;text-anchor:middle;'
    'dominant-baseline:central}text.l{text-anchor:start}text.r{text-anchor:end}'
    'text.t{font-size:12px;font-weight:bold;text-anchor:start}text.mute{fill:#555}'
    'text.blue{fill:#1f5fbf}text.green{fill:#1e6b2e}text.orange{fill:#c96a00}text.red{fill:#c0392b}'
    '.box{fill:#fff;stroke:#444;stroke-width:1}'
    '.box.blue{fill:#9ec5fe}.box.green{fill:#a6e3a1}.box.orange{fill:#ffd08a}.box.purple{fill:#d9b8ff}'
    '.box.gray{fill:#e6e6e6}.box.pad{fill:#f4f4f4}'
    '.box.blue-l{fill:#e8eefc}.box.green-l{fill:#eaf7e8}.box.orange-l{fill:#fff0d6}'
    '.box.gone{fill:#fafafa;stroke:#888;stroke-dasharray:4 3}'
    '.hl{fill:none;stroke:#c0392b;stroke-width:2}'
    '.grid{fill:none;stroke:#bbb;stroke-width:1}'
    '.e{fill:none;stroke:#333;stroke-width:1.4;marker-end:url(#ah)}'
    '.e.blue{stroke:#1f5fbf;marker-end:url(#ah-blue)}.e.green{stroke:#1e6b2e;marker-end:url(#ah-green)}'
    '.e.orange{stroke:#c96a00;marker-end:url(#ah-orange)}.e.purple{stroke:#7b4bc4;marker-end:url(#ah-purple)}'
    '.e.red{stroke:#c0392b;marker-end:url(#ah-red)}.e.dash{stroke-dasharray:4 3}.e.bare{marker-end:none}'
    '#ah path{fill:#333}#ah-blue path{fill:#1f5fbf}#ah-green path{fill:#1e6b2e}'
    '#ah-orange path{fill:#c96a00}#ah-purple path{fill:#7b4bc4}#ah-red path{fill:#c0392b}'
)
INK_MARKERS = "".join(
    f'<marker id="ah{n}" viewBox="0 0 8 8" refX="8" refY="4" markerWidth="7" markerHeight="7" orient="auto">'
    '<path d="M0,0 L8,4 L0,8 z"/></marker>'
    for n in ("", "-blue", "-green", "-orange", "-purple", "-red")
)


class Ink:
    def __init__(self, w, h):
        self.w, self.h, self.parts = w, h, []

    def text(self, x, y, s, cls=""):
        c = f' class="{cls}"' if cls else ""
        self.parts.append(f'<text{c} x="{x:g}" y="{y:g}">{s}</text>')

    def box(self, x, y, w, h, cls="", *lines):
        c = f"box {cls}".strip()
        self.parts.append(f'<rect class="{c}" x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}"/>')
        if len(lines) == 1:
            self.text(x + w / 2, y + h / 2, lines[0], "mute" if cls == "gone" else "")
        elif len(lines) == 2:
            self.text(x + w / 2, y + h / 2 - 9, lines[0])
            self.text(x + w / 2, y + h / 2 + 9, lines[1], "mute")

    def hl(self, x, y, w, h):
        self.parts.append(f'<rect class="hl" x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}"/>')

    def line(self, d, cls="e"):
        self.parts.append(f'<path class="{cls}" d="{d}"/>')

    def save(self, out):
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" '
                f'height="{self.h}"><style>{INK_CSS}</style><defs>{INK_MARKERS}</defs>')
        (HERE / out).write_text(head + "".join(self.parts) + "</svg>\n")


def glue(out="glue.svg"):
    """FSDP2 vs TIDE around one parameter group: copies and packing removed, transport off SMs."""
    s = Ink(960, 224)
    s.text(90, 20, "forward", "t")
    s.text(500, 20, "backward", "t")
    s.line("M475 30V176", "grid")
    s.text(20, 64, "FSDP2", "t")
    s.text(20, 144, "TIDE", "t")

    s.box(90, 40, 86, 48, "orange", "copy-in")
    s.box(198, 40, 144, 48, "orange", "all-gather", "16-32 CTAs")
    s.box(364, 40, 86, 48, "orange", "copy-out")
    s.box(500, 40, 100, 48, "orange", "_chunk_cat")
    s.box(622, 40, 318, 48, "orange", "reduce-scatter", "16-32 CTAs")
    for x0, x1 in ((176, 198), (342, 364), (600, 622)):
        s.line(f"M{x0} 64H{x1}")

    for x, w in ((90, 86), (364, 86), (500, 100)):
        s.box(x, 120, w, 48, "gone")
    s.box(198, 120, 144, 48, "blue", "all-gather")
    s.box(622, 120, 56, 48, "blue", "push")
    s.box(694, 120, 70, 48, "orange", "sum", "4 CTAs")
    s.box(780, 120, 56, 48, "green", "RDMA")
    s.box(852, 120, 88, 48, "purple", "NVLS land", "1-4 CTAs")
    for x0, x1 in ((678, 694), (764, 780), (836, 852)):
        s.line(f"M{x0} 144H{x1}")

    for x, cls, name in ((90, "orange", "SM work"), (200, "blue", "CE"), (266, "green", "RDMA"),
                         (342, "purple", "NVSwitch"), (446, "gone", "removed")):
        s.box(x, 196, 14, 14, cls)
        s.text(x + 22, 203, name, "l")
    s.save(out)


def ag_flow(out="ag_flow.svg"):
    """All-gather layout (rank-major vs direct param-major landing) and multi-node data path."""
    s = Ink(960, 296)
    s.text(16, 20, "① direct landing", "t")

    sizes, N, u = (2, 3, 1), 4, 26
    offs = (0, 2, 5)
    shard = 6
    sx, sy = 76, 86
    s.text(sx, sy - 14, "shard r1", "l")
    for p, w in enumerate(sizes):
        s.box(sx + offs[p] * u, sy, w * u, 32, "blue" if p == 1 else "blue-l", f"p{p}")

    x0, y_f, y_t = 312, 42, 130
    s.text(x0 - 12, y_f + 16, "FSDP2", "r")
    for r in range(N):
        for p, w in enumerate(sizes):
            x = x0 + (r * shard + offs[p]) * u
            s.box(x, y_f, w * u, 32, "orange" if (p, r) == (1, 1) else "pad")
        s.text(x0 + (r + 0.5) * shard * u, y_f - 12, f"r{r}", "mute")

    s.text(x0 - 12, y_t + 16, "TIDE", "r")
    for p, w in enumerate(sizes):
        for r in range(N):
            x = x0 + (offs[p] * N + r * w) * u
            s.box(x, y_t, w * u, 32, "blue" if (p, r) == (1, 1) else "blue-l")
        s.text(x0 + (offs[p] * N + w * N / 2) * u, y_t + 46, f"p{p}")
    s.hl(x0 + (offs[1] * N + 1 * sizes[1]) * u, y_t, sizes[1] * u, 32)

    mid_f = x0 + (1 * shard + offs[1] + 1.5) * u
    dst_t = x0 + (offs[1] * N + 1 * sizes[1] + 1.5) * u
    src_x = sx + (offs[1] + 1.5) * u
    s.line(f"M{src_x} {sy}V{y_f + 39}H{mid_f - 24}V{y_f + 32}")
    s.line(f"M{mid_f} {y_f + 32}V{y_t}", "e orange dash")
    s.text(mid_f + 42, (y_f + 32 + y_t) / 2, "copy-out", "orange")
    s.line(f"M{src_x} {sy + 32}V{y_t - 12}H{dst_t}V{y_t}", "e blue")

    s.line("M16 196H944", "grid")
    s.text(16, 214, "② multi-node path", "t")
    stages = (
        (76, "blue-l", "own shard", "in window"),
        (256, "green", "RDMA rounds", "log2 M"),
        (436, "green-l", "recv stage", "shared"),
        (616, "blue", "CE multicast", "NVSwitch"),
        (796, "blue-l", "param slot", "in place"),
    )
    for x, cls, l1, l2 in stages:
        s.box(x, 232, 140, 48, cls, l1, l2)
    for x, cls in ((216, "e green"), (396, "e green"), (576, "e blue"), (756, "e blue")):
        s.line(f"M{x} 256H{x + 40}", cls)
    s.save(out)


def rs_flow(out="rs_flow.svg"):
    """Reduce-scatter two-phase pipeline: intra-node CE push + TMA sum, inter-node rail + NVSwitch reduce."""
    s = Ink(960, 256)
    s.line("M476 16V240", "grid")

    # Left: Phase 1 (read phase, inside a node)
    s.text(16, 20, "① read phase", "t")
    gx = [24 + i * 54 for i in range(8)]
    for i, x in enumerate(gx):
        s.box(x, 40, 46, 28, "gray", f"G{i}")
    s.line("M101 68V88 M155 68V88 M209 68V88 M263 68V88 M317 68V88 M371 68V88 M425 68V88 M101 88H425", "e blue bare")
    s.line("M236 88V112", "e blue")
    s.text(290, 100, "CE push", "blue")
    s.box(146, 112, 180, 44, "blue-l", "staging slot", "2 slots")
    s.line("M236 156V184", "e blue")
    s.line("M47 68V206H146", "e")
    s.text(92, 194, "in place")
    s.box(146, 184, 140, 48, "orange", "fp32 sum", "4-8 CTAs")
    s.line("M286 208H318", "e orange")
    s.box(318, 184, 134, 48, "orange-l", "outbox", "2 banks")

    # Right: Phase 2 (network phase, across nodes + switch landing)
    s.text(496, 20, "② network phase", "t")
    s.box(644, 36, 120, 32, "green", "k-way RDMA")
    s.line("M704 68V88", "e green")
    rx = [496 + i * 56 for i in range(8)]
    for i, x in enumerate(rx):
        s.box(x, 88, 48, 32, "green-l", f"c{i + 1}")
    s.line("M520 120V140 M576 120V140 M632 120V140 M688 120V140 M744 120V140 M800 120V140 M856 120V140 M912 120V140 M520 140H912",
           "e purple bare")
    s.line("M640 140V184", "e purple")
    s.text(694, 162, "ld_reduce", "purple")
    s.box(570, 184, 140, 48, "purple", "NVSwitch", "fp32 add")
    s.line("M710 208H742", "e orange")
    s.box(742, 184, 134, 48, "gray", "fp32 shard")
    s.save(out)


if __name__ == "__main__":
    glue()
    ag_flow()
    rs_flow()
