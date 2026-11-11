# SPDX-FileCopyrightText: Copyright (c) 2026 Kuaishou Technology. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regenerate the benchmark charts from data.json (needs matplotlib): python3 docs/figures/plot.py"""
import json
import logging
import pathlib

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
# The diagrams' look (diagrams.py): monospace text, pastel fills with thin dark edges. Colors by path in every
# chart; orange is FSDP2's layout work around NCCL (the copy-out, the _chunk_cat packing) and what it pays.
BLUE, GREEN, ORANGE, PURPLE, GRAY = "#9ec5fe", "#a6e3a1", "#ffd08a", "#d9b8ff", "#e6e6e6"
EDGE, INK, TIDE_LINE, PURPLE_EDGE = "#444444", "#1b1f24", "#1f5fbf", "#7b4bc4"
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # the SVG names fonts this host lacks
plt.rcParams.update({
    "font.family": "monospace", "font.monospace": ["SF Mono", "Menlo", "Consolas", "DejaVu Sans Mono"],
    "font.size": 8.25, "axes.labelsize": 8.25, "axes.titlesize": 8.25, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
    "ytick.color": INK, "axes.edgecolor": EDGE, "axes.linewidth": 0.75, "axes.spines.top": False,
    "axes.spines.right": False, "xtick.major.size": 0, "ytick.major.size": 3, "ytick.major.width": 0.75,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#eeeeee", "grid.linewidth": 0.75,
    "axes.axisbelow": True, "legend.frameon": False, "legend.handlelength": 1.0, "legend.handleheight": 1.0,
    "svg.fonttype": "none", "svg.hashsalt": "tide", "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
})


def save(fig, name):
    fig.savefig(HERE / name, metadata={"Date": None, "Creator": None})
    plt.close(fig)


def box(ax, x, h, w, face, bottom=0.0, edge=EDGE, ls="-"):
    ax.bar(x, h, w, bottom=bottom, color=face, edgecolor=edge, linewidth=0.75, linestyle=ls)


def value(ax, x, y, s):
    ax.annotate(s, (x, y), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7)


def legend(target, items, ncol, columnspacing=1.6, **kw):
    """Square swatches like the diagrams' legends: (label, face, edge, line style)."""
    handles = [Patch(facecolor=f, edgecolor=e, linewidth=0.75, linestyle=s, label=lab) for lab, f, e, s in items]
    target.legend(handles=handles, ncol=ncol, handletextpad=0.5, columnspacing=columnspacing, **kw)


def interference(d):
    it = d["interference"]
    style = {"TIDE (4 sum CTAs)": ("TIDE, 4 sum CTAs", BLUE, "-"), "TIDE (16 sum CTAs)": ("TIDE, 16 sum CTAs", BLUE, "--"),
             "_chunk_cat + NCCL (what FSDP2 pays)": ("what FSDP2 pays", ORANGE, "-"),
             "FSDP2 with symmetric memory": ("FSDP2 + symmetric memory", GREEN, "-")}
    arms = [a for a in it["arms"] if a["name"] in style]  # the bare NCCL arms are in the table only
    n, w = len(arms), 0.8 / len(arms)
    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    for i, a in enumerate(arms):
        _, face, ls = style[a["name"]]
        for g, v in enumerate(a["pct"]):
            x = g + (i - (n - 1) / 2) * w
            box(ax, x, v, w * 0.86, face, ls=ls)
            value(ax, x, v, f"{v:.0f}" if v >= 10 else f"{v:.1f}")  # four bars a group: short labels
    ax.set_xticks(range(3), ["large GEMM", "one-wave GEMM", "memory-bound"])
    ax.set_xlabel("8 nodes (64 GPUs), 32 MiB reduce-scatter")
    ax.set_ylabel("compute slowdown (%)")
    ax.set_ylim(0, 80)
    legend(ax, [(lab, face, EDGE, ls) for lab, face, ls in style.values()], 2, loc="lower left",
           bbox_to_anchor=(0, 1.0), columnspacing=0.9)
    save(fig, "interference.svg")


def ag_fsdp2(d):
    f = d["all_gather_fsdp2"]
    ms = {a["name"]: a["ms"] for a in f["arms"]}
    paths = (("NCCL ring", "ring + copy-out", GRAY),
             ("NCCL copy engines (FSDP2 symmetric memory)", "copy engines + copy-out", GREEN))
    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    for j, (wire, total, face) in enumerate(paths):
        for g, t in enumerate(f["tide_ms"]):
            x, w, a = g + (j - 0.5) * 0.36, ms[wire][g] / t, ms[total][g] / t
            box(ax, x, w, 0.32, face)
            box(ax, x, a - w, 0.32, ORANGE, bottom=w)
            value(ax, x, a, f"{a:.2f}")
    ax.axhline(1.0, color=TIDE_LINE, lw=1.0, ls=(0, (4, 3)), zorder=3)
    ax.annotate("TIDE", (1.0, 1.0), xycoords=("axes fraction", "data"), xytext=(4, 0), textcoords="offset points",
                va="center", color=TIDE_LINE)
    ax.set_xticks(range(len(f["nodes"])), [f"{n} node{'s' if n > 1 else ''}" for n in f["nodes"]])
    ax.set_xlabel("128 MiB per rank, 30 tensors")
    ax.set_ylabel("latency / TIDE's")
    ax.set_ylim(0, 1.65)
    legend(ax, [("NCCL ring", GRAY, EDGE, "-"), ("NCCL copy engines", GREEN, EDGE, "-"),
                ("copy-out", ORANGE, EDGE, "-")], 3, loc="lower left", bbox_to_anchor=(0, 1.0), columnspacing=0.9)
    save(fig, "ag_fsdp2.svg")


def rs_scaling(d):
    r = d["reduce_scatter"]
    style = {  # legend labels must not start with "_": matplotlib hides those
        "TIDE": ("TIDE", BLUE, EDGE, "-"),
        "_chunk_cat + NCCL (what FSDP2 pays)": ("what FSDP2 pays (_chunk_cat + NCCL)", ORANGE, EDGE, "-"),
    }
    n, w = len(r["arms"]), 0.64 / len(r["arms"])
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5))
    for ax, size in zip(axes, map(str, r["sizes_mib"])):
        for i, a in enumerate(r["arms"]):
            _, face, edge, ls = style[a["name"]]
            xs = [g + (i - (n - 1) / 2) * w for g in range(len(r["nodes"]))]
            box(ax, xs, a["min"][size], w * 0.88, face, edge=edge, ls=ls)
            ax.vlines(xs, a["min"][size], a["median"][size], color=EDGE, lw=0.75)  # up to the median
            ax.hlines(a["median"][size], [x - w * 0.22 for x in xs], [x + w * 0.22 for x in xs], color=EDGE, lw=0.75)
        ax.set_xticks(range(len(r["nodes"])), [str(v) for v in r["nodes"]])
        ax.set_xlabel("nodes")
        ax.set_ylabel("latency (ms)")
        ax.set_title(f"{size} MiB per rank", loc="left")
    legend(fig, list(style.values()), 2, loc="lower center", bbox_to_anchor=(0.5, 0.97))
    fig.tight_layout()
    save(fig, "rs_scaling.svg")


def main():
    d = json.loads((HERE / "data.json").read_text())
    interference(d)
    ag_fsdp2(d)
    rs_scaling(d)


if __name__ == "__main__":
    main()
