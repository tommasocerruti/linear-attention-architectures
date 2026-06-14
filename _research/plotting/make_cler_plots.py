#!/usr/bin/env python3
"""Generate CLER plots without external plotting libraries.

The default Clariden login Python used in this project does not have
matplotlib, so this script writes lightweight SVGs directly. It also writes a
README manifest with source runs and interpretation notes.
"""

import json
import math
import os
import re
from pathlib import Path
from statistics import mean
from xml.sax.saxutils import escape


ROOT = Path("/users/course_00252/cler")
SCRATCH_RESULTS = Path(
    os.environ.get(
        "CLER_RESULTS_DIR",
        "/iopsstor/scratch/cscs/course_00252/cler/_research/results",
    )
)
PERF = Path(os.environ.get("CLER_PERF_DIR", str(SCRATCH_RESULTS / "performance")))
SCRATCH_RUNS = Path(os.environ.get("CLER_RUNS_DIR", str(SCRATCH_RESULTS / "runs")))
OUT = Path(os.environ.get("CLER_PLOTS_DIR", str(ROOT / "tracker/week4/plots")))
CLER_40B_PERF_DIR = Path(
    os.environ.get(
        "CLER_40B_PERF_DIR",
        str(PERF),
    )
)
SEQ_LEN = 4096
GLOBAL_BATCH = 128
TOKENS_PER_STEP = SEQ_LEN * GLOBAL_BATCH
CLER_40B_STEM = "1.3B-CLER-DN-MUON-970296930-2355856"
CLER_40B_TOKENS_PER_STEP = 512 * SEQ_LEN


COLORS = {
    "gdn": "#2f6f5e",
    "cler": "#b24646",
    "head": "#7b5dbb",
    "channel": "#d08a2e",
    "delta": "#2f5d9b",
    "linear": "#5e7f3d",
    "softmax": "#5f6672",
    "sdpa": "#c7cbd1",
    "grid": "#e8eaee",
    "axis": "#30343b",
    "text": "#22252a",
    "muted": "#686f7a",
    "bg": "#ffffff",
}


RUNS = {
    "gdn_adamw": PERF / "gdn-pytorch-350m-llama2-fwe1b-adamw-compile10h-2236100.jsonl",
    "cler_scalar_adamw": PERF / "cler-gated-v1-carry-350m-llama2-fwe1b-adamw-compile10h-2236101.jsonl",
    "cler_head_adamw": PERF / "cler-gated-v1-headgamma-350m-llama2-fwe1b-adamw-gamma1e-2-compile10h-2282970.jsonl",
    "cler_channel_adamw": PERF / "cler-gated-v1-channelgamma-350m-llama2-fwe1b-adamw-gamma1e-2-compile10h-2291777.jsonl",
    "gdn_muon_log": SCRATCH_RUNS / "350m-fwe1b-gdn-muon-c10h-2019280.log",
    "cler_gated_muon": SCRATCH_RUNS / "cler-g-350m-2059790.log",
    "deltanet_muon": SCRATCH_RUNS / "dn-350m-fwe1b-2065614.log",
    "cler_deltanet_muon": SCRATCH_RUNS / "cler-dn-350m-2059791.log",
}

GAMMA_FILES = {
    "scalar": PERF / "cler-gated-v1-carry-350m-llama2-fwe1b-adamw-compile10h-2236101.cler_gamma.jsonl",
    "head": PERF / "cler-gated-v1-headgamma-350m-llama2-fwe1b-adamw-gamma1e-2-compile10h-2282970.cler_gamma.jsonl",
    "channel": PERF / "cler-gated-v1-channelgamma-350m-llama2-fwe1b-adamw-gamma1e-2-compile10h-2291777.cler_gamma.jsonl",
}

RESIDUAL_FILES = {
    "scalar": PERF / "cler-gated-v1-carry-350m-llama2-fwe1b-adamw-compile10h-2236101.cler_residual.jsonl",
    "head": PERF / "cler-gated-v1-headgamma-350m-llama2-fwe1b-adamw-gamma1e-2-compile10h-2282970.cler_residual.jsonl",
    "channel": PERF / "cler-gated-v1-channelgamma-350m-llama2-fwe1b-adamw-gamma1e-2-compile10h-2291777.cler_residual.jsonl",
}

VALIDATION = {
    "GDN": {"final": 3.255381, "best": 3.251918, "color": COLORS["gdn"]},
    "CLER scalar": {"final": 3.256151, "best": 3.252717, "color": COLORS["cler"]},
    "CLER head": {"final": 3.258292, "best": 3.254543, "color": COLORS["head"]},
    "CLER channel": {"final": 3.257861, "best": 3.253723, "color": COLORS["channel"]},
}

MUON_LADDER = {
    "GDN": {"final": 2.831940, "best": 2.828173, "color": COLORS["gdn"]},
    "CLER-G": {"final": 2.833288, "best": 2.829291, "color": COLORS["cler"]},
    "DeltaNet": {"final": 2.851128, "best": 2.847952, "color": COLORS["delta"]},
    "CLER-DN": {"final": 2.850695, "best": 2.846753, "color": COLORS["head"]},
    "Linear": {"final": 2.880643, "best": 2.875981, "color": COLORS["linear"]},
    "Softmax": {"final": 2.846277, "best": 2.843195, "color": COLORS["softmax"]},
}

ADAMW_LADDER = {
    "GDN": {"final": 3.255381, "best": 3.251918, "color": COLORS["gdn"]},
    "CLER-G": {"final": 3.256151, "best": 3.252717, "color": COLORS["cler"]},
    "CLER-G head": {"final": 3.258292, "best": 3.254543, "color": COLORS["head"]},
    "CLER-G channel": {"final": 3.257861, "best": 3.253723, "color": COLORS["channel"]},
    "DeltaNet": {"final": 3.406819, "best": 3.403113, "color": COLORS["delta"]},
    "CLER-DN": {"final": 3.425838, "best": 3.424568, "color": COLORS["head"]},
}


def read_jsonl(path: Path):
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_megatron_train_log(path: Path):
    """Parse Megatron training iteration lines from a full stdout log."""
    rows = []
    pat = re.compile(
        r"iteration\s+(\d+)/\s*\d+.*?"
        r"consumed samples:\s*(\d+).*?"
        r"elapsed time per iteration \(ms\):\s*([0-9.]+).*?"
        r"learning rate:\s*([0-9.E+-]+).*?"
        r"lm loss:\s*([0-9.E+-]+).*?"
        r"grad norm:\s*([0-9.E+-]+).*?"
        r"params norm:\s*([0-9.E+-]+)"
    )
    with path.open(errors="replace") as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            step = int(m.group(1))
            elapsed_ms = float(m.group(3))
            rows.append(
                {
                    "step": step,
                    "consumed_samples": int(m.group(2)),
                    "iteration_time": elapsed_ms / 1000.0,
                    "lr": float(m.group(4)),
                    "train_loss": float(m.group(5)),
                    "grad_norm": float(m.group(6)),
                    "params_norm": float(m.group(7)),
                }
            )
    return rows


def final_jsonl(path: Path):
    last = None
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                last = json.loads(line)
    if last is None:
        raise ValueError(f"empty file: {path}")
    return last


def smooth(values, window=25):
    if window <= 1:
        return values
    out = []
    acc = []
    for v in values:
        acc.append(v)
        if len(acc) > window:
            acc.pop(0)
        out.append(sum(acc) / len(acc))
    return out


def layer_from_name(name: str) -> int:
    m = re.search(r"layers\.(\d+)\.", name)
    if not m:
        raise ValueError(f"could not parse layer from {name}")
    return int(m.group(1))


def gamma_index_from_name(name: str) -> int:
    m = re.search(r"cler_gamma\.(\d+)$", name)
    if not m:
        return 0
    return int(m.group(1))


def svg_doc(width, height, body, title=None):
    title_el = f"<title>{escape(title)}</title>" if title else ""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        f"{title_el}\n"
        f'<rect width="{width}" height="{height}" fill="{COLORS["bg"]}"/>\n'
        f"{body}\n</svg>\n"
    )


def text(x, y, s, size=16, weight="400", fill=None, anchor="start", family="Arial, sans-serif"):
    fill = fill or COLORS["text"]
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="{family}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{escape(str(s))}</text>'
    )


def line(x1, y1, x2, y2, stroke=None, width=2, dash=None, marker=False):
    stroke = stroke or COLORS["axis"]
    dash_s = f' stroke-dasharray="{dash}"' if dash else ""
    marker_s = ' marker-end="url(#arrow)"' if marker else ""
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{stroke}" stroke-width="{width}"{dash_s}{marker_s}/>'
    )


def rect(x, y, w, h, fill, stroke="#ffffff", sw=1, rx=6):
    return (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
        f'rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    )


def circle(x, y, r, fill, stroke="#ffffff", sw=1):
    return (
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}"/>'
    )


def polyline(points, stroke, width=2, fill="none", dash=None):
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    dash_s = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<polyline points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"{dash_s}/>'


def color_lerp(c1, c2, t):
    def h(c):
        c = c.lstrip("#")
        return tuple(int(c[i : i + 2], 16) for i in (0, 2, 4))

    a = h(c1)
    b = h(c2)
    rgb = tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))
    return "#" + "".join(f"{v:02x}" for v in rgb)


def diverging_color(v, vmax):
    if vmax <= 0:
        return "#ffffff"
    t = max(-1.0, min(1.0, v / vmax))
    if t < 0:
        return color_lerp("#2e6fbb", "#ffffff", 1 + t)
    return color_lerp("#ffffff", "#b24646", t)


def sequential_color(v, vmin, vmax, c0="#f1f5f1", c1="#2f6f5e"):
    if vmax <= vmin:
        t = 0.0
    else:
        t = max(0.0, min(1.0, (v - vmin) / (vmax - vmin)))
    return color_lerp(c0, c1, t)


def save_svg(name, width, height, body, title=None):
    path = OUT / f"{name}.svg"
    path.write_text(svg_doc(width, height, body, title), encoding="utf-8")
    return path


def plot_architecture():
    w, h = 1500, 640
    body = []
    body.append(text(60, 60, "Hybrid stack vs pure linear-memory stack", 34, "700"))
    body.append(text(60, 96, "350M setup: 20 transformer layers, linear_attention_freq controls where SDPA is inserted", 18, fill=COLORS["muted"]))

    def row(y, title, pattern, note):
        body.append(text(60, y + 28, title, 22, "700"))
        x0, box_w, gap = 270, 48, 8
        for i, kind in enumerate(pattern):
            x = x0 + i * (box_w + gap)
            color = COLORS["gdn"] if kind == "GDN" else COLORS["sdpa"]
            fg = "#ffffff" if kind == "GDN" else COLORS["text"]
            body.append(rect(x, y, box_w, 58, color, stroke="#ffffff", rx=5))
            body.append(text(x + box_w / 2, y + 34, kind, 12, "700", fill=fg, anchor="middle"))
            body.append(text(x + box_w / 2, y + 83, i, 11, fill=COLORS["muted"], anchor="middle"))
            if i < len(pattern) - 1:
                body.append(line(x + box_w + 2, y + 29, x + box_w + gap - 2, y + 29, stroke=COLORS["muted"], width=1.5))
        body.append(text(270, y + 124, note, 16, fill=COLORS["muted"]))

    hybrid = ["GDN" if i % 3 != 2 else "SDPA" for i in range(20)]
    pure = ["GDN" for _ in range(20)]
    row(150, "Hybrid freq=3", hybrid, "CLER-capable receiver layers: 14/20; SDPA layers do not consume the CLER side-channel.")
    row(375, "Pure freq=21", pure, "CLER-capable receiver layers: 20/20; removes the periodic softmax mixer loophole.")
    body.append(rect(1040, 535, 400, 74, "#f6f7f9", stroke="#dadde3", rx=8))
    body.append(text(1060, 560, "Parser detail:", 16, "700"))
    body.append(text(1170, 560, "freq=3 -> GDN,GDN,SDPA; freq=21 -> all GDN", 13))
    body.append(text(1170, 585, "freq=1 means all SDPA, not pure linear", 13, fill=COLORS["cler"]))
    return save_svg("01_architecture_hybrid_vs_pure", w, h, "\n".join(body), "Hybrid vs pure linear stack")


def plot_mechanism():
    w, h = 1500, 760
    body = []
    body.append(
        '<defs><marker id="arrow" markerWidth="12" markerHeight="8" refX="10" refY="4" '
        'orient="auto" markerUnits="strokeWidth"><path d="M0,0 L10,4 L0,8 z" '
        f'fill="{COLORS["axis"]}"/></marker></defs>'
    )
    body.append(text(60, 60, "CLER v1 side-channel: receiver-side residual injection", 34, "700"))
    body.append(text(60, 96, "The normal hidden state still flows through every Transformer layer; CLER carries a separate write-residual signal between linear-memory layers.", 18, fill=COLORS["muted"]))

    x1, x2, x3 = 120, 570, 1030
    y = 175
    body.append(rect(x1, y, 330, 155, "#e8f3ee", stroke="#bed7cd", rx=10))
    body.append(text(x1 + 165, y + 40, "Layer p(l)", 22, "700", anchor="middle"))
    body.append(text(x1 + 165, y + 78, "DeltaNet / GDN update", 17, anchor="middle"))
    body.append(text(x1 + 165, y + 112, "compute residual r[p(l), t]", 17, anchor="middle", fill=COLORS["gdn"]))

    body.append(rect(x2, y, 360, 155, "#fff4e4", stroke="#e3c693", rx=10))
    body.append(text(x2 + 180, y + 40, "CLER injection", 22, "700", anchor="middle"))
    body.append(text(x2 + 180, y + 78, "v + Gamma * rho(r_prev)", 18, anchor="middle", fill=COLORS["channel"]))
    body.append(text(x2 + 180, y + 112, "forms v_tilde for receiver", 17, anchor="middle"))

    body.append(rect(x3, y, 330, 155, "#e8f3ee", stroke="#bed7cd", rx=10))
    body.append(text(x3 + 165, y + 40, "Layer l", 22, "700", anchor="middle"))
    body.append(text(x3 + 165, y + 78, "DeltaNet / GDN update", 17, anchor="middle"))
    body.append(text(x3 + 165, y + 112, "compute new residual r[l, t]", 17, anchor="middle", fill=COLORS["gdn"]))

    body.append(line(x1 + 330, y + 78, x2, y + 78, width=3, marker=True))
    body.append(line(x2 + 360, y + 78, x3, y + 78, width=3, marker=True))
    body.append(text(455, y + 62, "cross-layer residual", 14, fill=COLORS["muted"], anchor="middle"))
    body.append(text(970, y + 62, "modified target", 14, fill=COLORS["muted"], anchor="middle"))

    eq_y = 430
    body.append(text(90, eq_y, "Equations used in the implementation", 25, "700"))
    body.append(rect(90, eq_y + 25, 1320, 220, "#f8f9fb", stroke="#dde1e7", rx=10))
    body.append(text(130, eq_y + 75, "v_tilde[l,t,h,d] = v[l,t,h,d] + Gamma[l,h,d] * rho(r[prev(l),t,h,d])", 24, "600", family="Courier New, monospace"))
    body.append(text(130, eq_y + 122, "r[l,t] = v_tilde[l,t] - W[l,t-1] phi(k[l,t])", 24, "600", family="Courier New, monospace"))
    body.append(text(130, eq_y + 169, "Gamma is static receiver-side routing; GDN gates are token-dependent write/forget/output controls.", 19, fill=COLORS["muted"]))

    body.append(rect(90, 675, 1320, 1, COLORS["grid"], stroke=COLORS["grid"], rx=0))
    body.append(text(90, 713, "Key distinction:", 18, "700"))
    body.append(text(230, 713, "per-channel CLER has V's head/channel axes [8,64], but is broadcast over batch and time.", 18))
    return save_svg("02_cler_mechanism_diagram", w, h, "\n".join(body), "CLER mechanism")


def plot_cler_transformer_style():
    """Transformer-style CLER architecture diagram for the technical report."""

    w, h = 1500, 1180
    body = []
    body.append(
        '<defs><marker id="arrow" markerWidth="12" markerHeight="8" refX="10" refY="4" '
        'orient="auto" markerUnits="strokeWidth"><path d="M0,0 L10,4 L0,8 z" '
        f'fill="{COLORS["axis"]}"/></marker>'
        '<marker id="arrow_cler" markerWidth="12" markerHeight="8" refX="10" refY="4" '
        'orient="auto" markerUnits="strokeWidth"><path d="M0,0 L10,4 L0,8 z" '
        f'fill="{COLORS["cler"]}"/></marker></defs>'
    )
    body.append(text(60, 62, "Cross-Layer Error Residuals (CLER)", 34, "700"))
    body.append(
        text(
            60,
            96,
            "CLER routes the delta-rule write residual between linear-memory layers; the normal Transformer residual stream is unchanged.",
            17,
            fill=COLORS["muted"],
        )
    )

    def arrow(x1, y1, x2, y2, stroke=None, width=2, dash=None, cler=False):
        marker_name = "arrow_cler" if cler else "arrow"
        stroke = stroke or (COLORS["cler"] if cler else COLORS["axis"])
        dash_s = f' stroke-dasharray="{dash}"' if dash else ""
        return (
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{stroke}" stroke-width="{width}" marker-end="url(#{marker_name})"{dash_s}/>'
        )

    def add_block(x, y, width, height, label, fill, stroke, size=16, weight="700"):
        body.append(rect(x, y, width, height, fill, stroke=stroke, sw=1.4, rx=6))
        for idx, part in enumerate(str(label).split("\n")):
            body.append(text(x + width / 2, y + height / 2 - 6 + idx * 20, part, size, weight, anchor="middle"))

    def layer_column(x, y, title, kind, receiver=False, producer=False):
        col_w = 320
        body.append(text(x + col_w / 2, y - 28, title, 21, "700", anchor="middle"))
        add_block(x, y, col_w, 54, "RMSNorm", "#f7f8fa", "#d8dce3", 15)
        mixer_label = "GDN / DeltaNet\nlinear-memory mixer" if kind == "linear" else "SDPA / softmax\nattention mixer"
        mixer_fill = "#e8f3ee" if kind == "linear" else "#eef1f5"
        mixer_stroke = "#a8ccb9" if kind == "linear" else "#c4c9d2"
        add_block(x, y + 78, col_w, 94, mixer_label, mixer_fill, mixer_stroke, 15)
        add_block(x, y + 202, col_w, 46, "Add", "#ffffff", "#d8dce3", 15)
        add_block(x, y + 278, col_w, 54, "RMSNorm", "#f7f8fa", "#d8dce3", 15)
        add_block(x, y + 356, col_w, 80, "MLP", "#f7f8fa", "#d8dce3", 17)
        add_block(x, y + 466, col_w, 46, "Add", "#ffffff", "#d8dce3", 15)
        # Normal hidden-state stream.
        cx = x + col_w / 2
        for y1, y2 in [(y + 54, y + 78), (y + 172, y + 202), (y + 248, y + 278), (y + 332, y + 356), (y + 436, y + 466)]:
            body.append(arrow(cx, y1, cx, y2, width=2))
        # Residual bypass hints, Transformer-figure style.
        body.append(polyline([(x - 26, y + 25), (x - 26, y + 225), (x, y + 225)], COLORS["muted"], width=1.8, dash="5 4"))
        body.append(polyline([(x - 26, y + 305), (x - 26, y + 490), (x, y + 490)], COLORS["muted"], width=1.8, dash="5 4"))
        if kind == "linear":
            if producer:
                body.append(text(x + col_w + 18, y + 118, "emit", 13, "700", fill=COLORS["cler"]))
                body.append(text(x + col_w + 18, y + 140, "r_{p(l),t}", 18, "700", fill=COLORS["cler"], family="Times New Roman, serif"))
            if receiver:
                body.append(text(x - 130, y + 86, "receive", 13, "700", fill=COLORS["cler"]))
                body.append(text(x - 130, y + 108, "Gamma_l rho(r)", 15, "700", fill=COLORS["cler"], family="Times New Roman, serif"))

    y0 = 220
    x0, x1, x2 = 120, 590, 1060
    layer_column(x0, y0, "Layer p(l)", "linear", producer=True)
    layer_column(x1, y0, "Optional intervening layer", "sdpa")
    layer_column(x2, y0, "Layer l", "linear", receiver=True, producer=True)

    # Normal hidden-state connections between layers.
    body.append(arrow(x0 + 320, y0 + 490, x1, y0 + 26, stroke=COLORS["axis"], width=2))
    body.append(arrow(x1 + 320, y0 + 490, x2, y0 + 26, stroke=COLORS["axis"], width=2))
    body.append(text(585, y0 + 545, "normal hidden-state stream h_l", 15, fill=COLORS["muted"], anchor="middle"))

    # CLER side-channel path: it bypasses SDPA and is consumed by the receiver.
    side_y = y0 + 124
    body.append(
        polyline(
            [
                (x0 + 320, side_y),
                (x0 + 390, side_y),
                (x0 + 390, y0 - 55),
                (x2 - 60, y0 - 55),
                (x2 - 60, side_y),
                (x2, side_y),
            ],
            COLORS["cler"],
            width=4,
        ).replace("/>", ' marker-end="url(#arrow_cler)"/>')
    )
    body.append(text(710, y0 - 78, "CLER side-channel carries write residual across depth", 17, "700", fill=COLORS["cler"], anchor="middle"))

    # Equation panel.
    eq_y = 805
    body.append(rect(90, eq_y, 1320, 300, "#f8f9fb", stroke="#dde1e7", sw=1.4, rx=8))
    body.append(text(125, eq_y + 42, "Receiver-side CLER update", 23, "700"))
    body.append(text(125, eq_y + 88, "r_prev = v_prev - W_prev phi(k_prev)", 23, "600", family="Times New Roman, serif"))
    body.append(text(125, eq_y + 138, "v_tilde[l,t,h,d] = v[l,t,h,d] + Gamma[l,h,d] rho(r_prev[t,h,d])", 23, "600", fill=COLORS["cler"], family="Times New Roman, serif"))
    body.append(text(125, eq_y + 188, "r[l,t] = v_tilde[l,t] - W[l,t-1] phi(k[l,t])", 23, "600", family="Times New Roman, serif"))
    body.append(text(125, eq_y + 242, "Gamma can be scalar, per-head, or per-channel; it is static over tokens, unlike GDN gates.", 17, fill=COLORS["muted"]))

    # Legend.
    body.append(rect(1015, 720, 360, 64, "#ffffff", stroke="#dde1e7", rx=6))
    body.append(line(1038, 744, 1080, 744, stroke=COLORS["axis"], width=3))
    body.append(text(1095, 749, "hidden state / Transformer residuals", 13))
    body.append(line(1038, 768, 1080, 768, stroke=COLORS["cler"], width=4))
    body.append(text(1095, 773, "CLER write-residual side-channel", 13))

    return save_svg(
        "20_cler_architecture_transformer_style",
        w,
        h,
        "\n".join(body),
        "CLER architecture",
    )


def plot_line_chart(
    name,
    title,
    series,
    width=1500,
    height=760,
    y_label="train loss",
    x_label="consumed tokens (B)",
    baseline_zero_label=None,
    subtitle=None,
    y_min_override=None,
    y_max_override=None,
):
    margin = dict(l=95, r=40, t=115, b=85)
    x_min = min(min(xs) for xs, _, _ in series)
    x_max = max(max(xs) for xs, _, _ in series)
    y_min = min(min(ys) for _, ys, _ in series)
    y_max = max(max(ys) for _, ys, _ in series)
    pad = (y_max - y_min) * 0.06
    y_min = y_min - pad if y_min_override is None else y_min_override
    y_max = y_max + pad if y_max_override is None else y_max_override
    pw = width - margin["l"] - margin["r"]
    ph = height - margin["t"] - margin["b"]

    def sx(x):
        return margin["l"] + (x - x_min) / (x_max - x_min) * pw

    def sy(y):
        return margin["t"] + (y_max - y) / (y_max - y_min) * ph

    body = [text(60, 60, title, 32, "700")]
    body.append(text(60, 92, subtitle or "Curves are locally smoothed training loss from local JSONL/stdout logs; validation conclusions use the separate final/best validation bars.", 16, fill=COLORS["muted"]))
    y_fmt = "{:.3f}" if (y_max - y_min) < 0.2 else "{:.2f}"
    for i in range(6):
        yv = y_min + (y_max - y_min) * i / 5
        y = sy(yv)
        body.append(line(margin["l"], y, width - margin["r"], y, stroke=COLORS["grid"], width=1))
        body.append(text(margin["l"] - 12, y + 5, y_fmt.format(yv), 13, fill=COLORS["muted"], anchor="end"))
    for i in range(6):
        xv = x_min + (x_max - x_min) * i / 5
        x = sx(xv)
        body.append(line(x, margin["t"], x, height - margin["b"], stroke=COLORS["grid"], width=1))
        body.append(text(x, height - margin["b"] + 26, f"{xv:.1f}", 13, fill=COLORS["muted"], anchor="middle"))
    body.append(line(margin["l"], margin["t"], margin["l"], height - margin["b"], width=2))
    body.append(line(margin["l"], height - margin["b"], width - margin["r"], height - margin["b"], width=2))
    if baseline_zero_label and y_min < 0 < y_max:
        y0 = sy(0)
        body.append(line(margin["l"], y0, width - margin["r"], y0, stroke=COLORS["axis"], width=2, dash="7 5"))
    body.append(text(width / 2, height - 28, x_label, 16, fill=COLORS["muted"], anchor="middle"))
    body.append(text(26, height / 2, y_label, 16, fill=COLORS["muted"], anchor="middle", family="Arial, sans-serif") .replace("<text ", '<text transform="rotate(-90 26 {:.2f})" '.format(height / 2), 1))

    lx, ly = 1080, 138
    legend_idx = 0
    if baseline_zero_label:
        body.append(line(lx, ly, lx + 34, ly, stroke=COLORS["axis"], width=3, dash="7 5"))
        body.append(text(lx + 45, ly + 5, baseline_zero_label, 15))
        legend_idx = 1
    for idx, (xs, ys, meta) in enumerate(series):
        pts = [(sx(x), sy(y)) for x, y in zip(xs, ys)]
        body.append(polyline(pts, meta["color"], width=3, dash=meta.get("dash")))
        body.append(circle(pts[-1][0], pts[-1][1], 4, meta["color"], stroke=meta["color"]))
        yleg = ly + (idx + legend_idx) * 28
        body.append(line(lx, yleg, lx + 34, yleg, stroke=meta["color"], width=4, dash=meta.get("dash")))
        body.append(text(lx + 45, yleg + 5, meta["label"], 15))
    return save_svg(name, width, height, "\n".join(body), title)


def plot_training_curves():
    series = []
    labels = [
        ("GDN AdamW", "gdn_adamw", COLORS["gdn"]),
        ("CLER scalar", "cler_scalar_adamw", COLORS["cler"]),
        ("CLER head", "cler_head_adamw", COLORS["head"]),
        ("CLER channel", "cler_channel_adamw", COLORS["channel"]),
    ]
    for label, key, color in labels:
        rows = read_jsonl(RUNS[key])
        xs = [r["step"] * TOKENS_PER_STEP / 1e9 for r in rows if "train_loss" in r]
        ys = smooth([r["train_loss"] for r in rows if "train_loss" in r], 25)
        series.append((xs, ys, {"label": label, "color": color}))
    return plot_line_chart(
        "03_adamw_train_loss_capacity_curves",
        "AdamW train loss: GDN vs static-gamma CLER variants",
        series,
    )


def plot_training_delta_vs_gdn():
    rows_gdn = read_jsonl(RUNS["gdn_adamw"])
    gdn_by_step = {
        r["step"]: v
        for r, v in zip(rows_gdn, smooth([r["train_loss"] for r in rows_gdn], 25))
        if "train_loss" in r
    }
    series = []
    labels = [
        ("CLER scalar - GDN", "cler_scalar_adamw", COLORS["cler"]),
        ("CLER head - GDN", "cler_head_adamw", COLORS["head"]),
        ("CLER channel - GDN", "cler_channel_adamw", COLORS["channel"]),
    ]
    for label, key, color in labels:
        rows = read_jsonl(RUNS[key])
        smoothed = smooth([r["train_loss"] for r in rows if "train_loss" in r], 25)
        xs, ys = [], []
        for r, y in zip([r for r in rows if "train_loss" in r], smoothed):
            step = r["step"]
            if step < 50 or step not in gdn_by_step:
                continue
            xs.append(step * TOKENS_PER_STEP / 1e9)
            ys.append(y - gdn_by_step[step])
        series.append((xs, ys, {"label": label, "color": color}))
    return plot_line_chart(
        "03b_adamw_train_loss_delta_vs_gdn",
        "AdamW smoothed train-loss delta vs GDN",
        series,
        y_label="train loss delta",
        baseline_zero_label="GDN baseline",
    )


def plot_actual_train_loss_zoom(name, title, run_specs, start_tokens_b=0.15):
    """Plot actual smoothed training loss after warmup, including baseline."""
    series = []
    for label, rows, color, dash in run_specs:
        train_rows = [r for r in rows if "train_loss" in r]
        smoothed = smooth([r["train_loss"] for r in train_rows], 25)
        xs, ys = [], []
        for r, y in zip(train_rows, smoothed):
            x = r["step"] * TOKENS_PER_STEP / 1e9
            if x < start_tokens_b:
                continue
            xs.append(x)
            ys.append(y)
        series.append((xs, ys, {"label": label, "color": color, "dash": dash}))
    return plot_line_chart(
        name,
        title,
        series,
        y_label="smoothed train loss",
        x_label="consumed tokens (B)",
    )


def plot_adamw_train_loss_zoom_with_baseline():
    specs = [
        ("CLER scalar", read_jsonl(RUNS["cler_scalar_adamw"]), COLORS["cler"], None),
        ("CLER head", read_jsonl(RUNS["cler_head_adamw"]), COLORS["head"], None),
        ("CLER channel", read_jsonl(RUNS["cler_channel_adamw"]), COLORS["channel"], None),
        ("GDN baseline", read_jsonl(RUNS["gdn_adamw"]), COLORS["axis"], "7 5"),
    ]
    return plot_actual_train_loss_zoom(
        "03c_adamw_train_loss_zoom_with_gdn_baseline",
        "AdamW smoothed train loss zoom with GDN baseline",
        specs,
    )


def plot_muon_delta_vs_gdn():
    rows_gdn = read_megatron_train_log(RUNS["gdn_muon_log"])
    gdn_smoothed = smooth([r["train_loss"] for r in rows_gdn], 25)
    gdn_by_step = {r["step"]: v for r, v in zip(rows_gdn, gdn_smoothed)}

    rows = read_megatron_train_log(RUNS["cler_gated_muon"])
    cler_smoothed = smooth([r["train_loss"] for r in rows if "train_loss" in r], 25)
    xs, ys = [], []
    for r, y in zip([r for r in rows if "train_loss" in r], cler_smoothed):
        step = r["step"]
        if step < 50 or step not in gdn_by_step:
            continue
        xs.append(step * TOKENS_PER_STEP / 1e9)
        ys.append(y - gdn_by_step[step])
    return plot_line_chart(
        "12_muon_train_loss_delta_vs_gdn",
        "Muon smoothed train-loss delta: CLER-Gated vs GDN",
        [(xs, ys, {"label": "CLER-Gated - GDN", "color": COLORS["cler"]})],
        y_label="train loss delta",
        baseline_zero_label="GDN baseline",
    )


def plot_muon_train_loss_zoom_with_gdn_baseline():
    specs = [
        ("CLER-Gated", read_megatron_train_log(RUNS["cler_gated_muon"]), COLORS["cler"], None),
        ("GDN baseline", read_megatron_train_log(RUNS["gdn_muon_log"]), COLORS["axis"], "7 5"),
    ]
    return plot_actual_train_loss_zoom(
        "12b_muon_train_loss_zoom_with_gdn_baseline",
        "Muon smoothed train loss zoom with GDN baseline",
        specs,
    )


def plot_muon_delta_vs_deltanet():
    rows_dn = read_megatron_train_log(RUNS["deltanet_muon"])
    dn_smoothed = smooth([r["train_loss"] for r in rows_dn if "train_loss" in r], 25)
    dn_by_step = {r["step"]: v for r, v in zip([r for r in rows_dn if "train_loss" in r], dn_smoothed)}

    rows = read_megatron_train_log(RUNS["cler_deltanet_muon"])
    cler_smoothed = smooth([r["train_loss"] for r in rows if "train_loss" in r], 25)
    xs, ys = [], []
    for r, y in zip([r for r in rows if "train_loss" in r], cler_smoothed):
        step = r["step"]
        if step < 50 or step not in dn_by_step:
            continue
        xs.append(step * TOKENS_PER_STEP / 1e9)
        ys.append(y - dn_by_step[step])
    return plot_line_chart(
        "13_muon_train_loss_delta_vs_deltanet",
        "Muon smoothed train-loss delta: CLER-DeltaNet vs DeltaNet",
        [(xs, ys, {"label": "CLER-DN - DeltaNet", "color": COLORS["head"]})],
        y_label="train loss delta",
        baseline_zero_label="DeltaNet baseline",
    )


def plot_muon_train_loss_zoom_with_deltanet_baseline():
    specs = [
        ("CLER-DeltaNet", read_megatron_train_log(RUNS["cler_deltanet_muon"]), COLORS["head"], None),
        ("DeltaNet baseline", read_megatron_train_log(RUNS["deltanet_muon"]), COLORS["axis"], "7 5"),
    ]
    return plot_actual_train_loss_zoom(
        "13b_muon_train_loss_zoom_with_deltanet_baseline",
        "Muon smoothed train loss zoom with DeltaNet baseline",
        specs,
    )


def plot_muon_validation_delta():
    pairs = [
        ("CLER-G vs GDN", 2.833288 - 2.831940, 2.829291 - 2.828173, COLORS["cler"]),
        ("CLER-DN vs DN", 2.850695 - 2.851128, 2.846753 - 2.847952, COLORS["head"]),
    ]
    w, h = 1180, 680
    vals = [v for _, a, b, _ in pairs for v in (a, b)]
    ymax = max(abs(v) for v in vals) * 1.35
    margin = dict(l=110, r=40, t=120, b=110)
    pw = w - margin["l"] - margin["r"]
    ph = h - margin["t"] - margin["b"]

    def sy(v):
        return margin["t"] + (ymax - v) / (2 * ymax) * ph

    body = [text(60, 60, "Muon validation-loss delta for matched CLER pairs", 32, "700")]
    body.append(text(60, 92, "Positive is worse than the matched baseline; negative is better. Effects are tiny in both cases.", 16, fill=COLORS["muted"]))
    for i in range(7):
        v = -ymax + 2 * ymax * i / 6
        y = sy(v)
        body.append(line(margin["l"], y, w - margin["r"], y, stroke=COLORS["grid"], width=1))
        body.append(text(margin["l"] - 12, y + 5, f"{v:.4f}", 13, fill=COLORS["muted"], anchor="end"))
    body.append(line(margin["l"], sy(0), w - margin["r"], sy(0), stroke=COLORS["axis"], width=2, dash="7 5"))
    group_w = pw / len(pairs)
    bar_w = 62
    for i, (label, final, best, color) in enumerate(pairs):
        cx = margin["l"] + group_w * (i + 0.5)
        for j, (v, shade) in enumerate([(final, color), (best, color_lerp(color, "#ffffff", 0.35))]):
            x = cx + (j - 0.5) * (bar_w + 12) - bar_w / 2
            y0 = sy(0)
            y = sy(v)
            body.append(rect(x, min(y, y0), bar_w, abs(y0 - y), shade, stroke="#ffffff", rx=4))
            body.append(text(x + bar_w / 2, y - 8 if v >= 0 else y + 20, f"{v:+.4f}", 13, "700", anchor="middle"))
        body.append(text(cx, h - margin["b"] + 32, label, 16, "700", anchor="middle"))
    body.append(text(935, 145, "dark: final", 13))
    body.append(text(935, 168, "light: best", 13))
    body.append(line(margin["l"], margin["t"], margin["l"], h - margin["b"], width=2))
    body.append(line(margin["l"], h - margin["b"], w - margin["r"], h - margin["b"], width=2))
    body.append(text(28, h / 2, "val loss delta", 16, fill=COLORS["muted"], anchor="middle").replace("<text ", f'<text transform="rotate(-90 28 {h / 2:.2f})" ', 1))
    return save_svg("14_muon_validation_delta_matched_pairs", w, h, "\n".join(body), "Muon validation delta")


def plot_validation_delta():
    w, h = 1320, 720
    base = VALIDATION["GDN"]
    modes = ["CLER scalar", "CLER head", "CLER channel"]
    final = [VALIDATION[m]["final"] - base["final"] for m in modes]
    best = [VALIDATION[m]["best"] - base["best"] for m in modes]
    vals = final + best
    ymax = max(vals) * 1.18
    margin = dict(l=110, r=40, t=120, b=110)
    pw = w - margin["l"] - margin["r"]
    ph = h - margin["t"] - margin["b"]

    def sy(v):
        return margin["t"] + (ymax - v) / ymax * ph

    body = [text(60, 60, "Capacity ablation: validation loss delta vs GDN AdamW", 32, "700")]
    body.append(text(60, 92, "Positive values are worse than the matched GDN baseline. Per-channel > per-head, but scalar remains least-worse.", 16, fill=COLORS["muted"]))
    for i in range(6):
        v = ymax * i / 5
        y = sy(v)
        body.append(line(margin["l"], y, w - margin["r"], y, stroke=COLORS["grid"], width=1))
        body.append(text(margin["l"] - 12, y + 5, f"{v:.4f}", 13, fill=COLORS["muted"], anchor="end"))
    group_w = pw / len(modes)
    bar_w = 58
    for i, mode in enumerate(modes):
        cx = margin["l"] + group_w * (i + 0.5)
        color = VALIDATION[mode]["color"]
        for j, (v, kind, shade) in enumerate([(final[i], "final", color), (best[i], "best", color_lerp(color, "#ffffff", 0.35))]):
            x = cx + (j - 0.5) * (bar_w + 10) - bar_w / 2
            y = sy(v)
            body.append(rect(x, y, bar_w, h - margin["b"] - y, shade, stroke="#ffffff", rx=4))
            body.append(text(x + bar_w / 2, y - 8, f"+{v:.4f}", 13, "700", fill=COLORS["text"], anchor="middle"))
        body.append(text(cx, h - margin["b"] + 30, mode.replace("CLER ", ""), 16, "700", anchor="middle"))
    body.append(line(margin["l"], margin["t"], margin["l"], h - margin["b"], width=2))
    body.append(line(margin["l"], h - margin["b"], w - margin["r"], h - margin["b"], width=2))
    body.append(text(26, h / 2, "val loss delta vs GDN", 16, fill=COLORS["muted"], anchor="middle").replace("<text ", f'<text transform="rotate(-90 26 {h / 2:.2f})" ', 1))
    body.append(rect(930, 128, 290, 74, "#f8f9fb", stroke="#dde1e7", rx=8))
    body.append(rect(955, 150, 24, 16, COLORS["cler"], stroke=COLORS["cler"], rx=2))
    body.append(text(990, 164, "final validation", 14))
    body.append(rect(955, 178, 24, 16, color_lerp(COLORS["cler"], "#ffffff", 0.35), stroke=color_lerp(COLORS["cler"], "#ffffff", 0.35), rx=2))
    body.append(text(990, 192, "best validation", 14))
    return save_svg("04_validation_delta_vs_gdn_adamw", w, h, "\n".join(body), "Validation delta")


def plot_muon_ladder():
    w, h = 1380, 720
    names = list(MUON_LADDER.keys())
    vals = [MUON_LADDER[n]["best"] for n in names]
    ymin, ymax = min(vals) - 0.006, max(vals) + 0.006
    margin = dict(l=105, r=40, t=115, b=115)
    pw = w - margin["l"] - margin["r"]
    ph = h - margin["t"] - margin["b"]

    def sy(v):
        return margin["t"] + (ymax - v) / (ymax - ymin) * ph

    body = [text(60, 60, "Muon architecture ladder: best validation loss", 32, "700")]
    body.append(text(60, 92, "This plot is not a CLER win claim; the matched tests are CLER-G vs GDN and CLER-DN vs DeltaNet.", 16, fill=COLORS["muted"]))
    for i in range(6):
        v = ymin + (ymax - ymin) * i / 5
        y = sy(v)
        body.append(line(margin["l"], y, w - margin["r"], y, stroke=COLORS["grid"], width=1))
        body.append(text(margin["l"] - 12, y + 5, f"{v:.3f}", 13, fill=COLORS["muted"], anchor="end"))
    group_w = pw / len(names)
    bar_w = 90
    for i, name in enumerate(names):
        cx = margin["l"] + group_w * (i + 0.5)
        v = MUON_LADDER[name]["best"]
        y = sy(v)
        body.append(rect(cx - bar_w / 2, y, bar_w, h - margin["b"] - y, MUON_LADDER[name]["color"], stroke="#ffffff", rx=5))
        body.append(text(cx, y - 9, f"{v:.4f}", 13, "700", anchor="middle"))
        body.append(text(cx, h - margin["b"] + 30, name, 15, "700", anchor="middle"))
    body.append(line(margin["l"], margin["t"], margin["l"], h - margin["b"], width=2))
    body.append(line(margin["l"], h - margin["b"], w - margin["r"], h - margin["b"], width=2))
    body.append(text(26, h / 2, "best validation loss", 16, fill=COLORS["muted"], anchor="middle").replace("<text ", f'<text transform="rotate(-90 26 {h / 2:.2f})" ', 1))
    return save_svg("05_muon_architecture_ladder_best_val", w, h, "\n".join(body), "Muon ladder")


def plot_adamw_ladder_best():
    w, h = 1500, 720
    names = list(ADAMW_LADDER.keys())
    vals = [ADAMW_LADDER[n]["best"] for n in names]
    ymin, ymax = min(vals) - 0.018, max(vals) + 0.018
    margin = dict(l=105, r=40, t=115, b=125)
    pw = w - margin["l"] - margin["r"]
    ph = h - margin["t"] - margin["b"]

    def sy(v):
        return margin["t"] + (ymax - v) / (ymax - ymin) * ph

    body = [text(60, 60, "AdamW architecture ladder: best validation loss", 32, "700")]
    body.append(text(60, 92, "Absolute best validation loss. Matched comparisons are CLER-G vs GDN and CLER-DN vs DeltaNet.", 16, fill=COLORS["muted"]))
    for i in range(6):
        v = ymin + (ymax - ymin) * i / 5
        y = sy(v)
        body.append(line(margin["l"], y, w - margin["r"], y, stroke=COLORS["grid"], width=1))
        body.append(text(margin["l"] - 12, y + 5, f"{v:.3f}", 13, fill=COLORS["muted"], anchor="end"))
    group_w = pw / len(names)
    bar_w = 92
    for i, name in enumerate(names):
        cx = margin["l"] + group_w * (i + 0.5)
        v = ADAMW_LADDER[name]["best"]
        y = sy(v)
        body.append(rect(cx - bar_w / 2, y, bar_w, h - margin["b"] - y, ADAMW_LADDER[name]["color"], stroke="#ffffff", rx=5))
        body.append(text(cx, y - 9, f"{v:.4f}", 13, "700", anchor="middle"))
        label = name.replace("CLER-G ", "CLER-G\n")
        if "\n" in label:
            a, b = label.split("\n", 1)
            body.append(text(cx, h - margin["b"] + 28, a, 14, "700", anchor="middle"))
            body.append(text(cx, h - margin["b"] + 48, b, 13, "700", anchor="middle"))
        else:
            body.append(text(cx, h - margin["b"] + 34, label, 14, "700", anchor="middle"))
    body.append(line(margin["l"], margin["t"], margin["l"], h - margin["b"], width=2))
    body.append(line(margin["l"], h - margin["b"], w - margin["r"], h - margin["b"], width=2))
    body.append(text(26, h / 2, "best validation loss", 16, fill=COLORS["muted"], anchor="middle").replace("<text ", f'<text transform="rotate(-90 26 {h / 2:.2f})" ', 1))
    return save_svg("15_adamw_architecture_ladder_best_val", w, h, "\n".join(body), "AdamW ladder")


def plot_adamw_ladder_final_best():
    w, h = 1500, 720
    names = list(ADAMW_LADDER.keys())
    vals = [ADAMW_LADDER[n][k] for n in names for k in ("final", "best")]
    ymin, ymax = min(vals) - 0.018, max(vals) + 0.018
    margin = dict(l=105, r=40, t=115, b=125)
    pw = w - margin["l"] - margin["r"]
    ph = h - margin["t"] - margin["b"]

    def sy(v):
        return margin["t"] + (ymax - v) / (ymax - ymin) * ph

    body = [text(60, 60, "AdamW architecture ladder: final and best validation loss", 32, "700")]
    body.append(text(60, 92, "Dark bars are final validation; light bars are best validation observed during the run.", 16, fill=COLORS["muted"]))
    for i in range(6):
        v = ymin + (ymax - ymin) * i / 5
        y = sy(v)
        body.append(line(margin["l"], y, w - margin["r"], y, stroke=COLORS["grid"], width=1))
        body.append(text(margin["l"] - 12, y + 5, f"{v:.3f}", 13, fill=COLORS["muted"], anchor="end"))
    group_w = pw / len(names)
    bar_w = 48
    for i, name in enumerate(names):
        cx = margin["l"] + group_w * (i + 0.5)
        color = ADAMW_LADDER[name]["color"]
        for j, key in enumerate(["final", "best"]):
            v = ADAMW_LADDER[name][key]
            y = sy(v)
            shade = color if key == "final" else color_lerp(color, "#ffffff", 0.35)
            x = cx + (j - 0.5) * (bar_w + 8) - bar_w / 2
            body.append(rect(x, y, bar_w, h - margin["b"] - y, shade, stroke="#ffffff", rx=4))
            body.append(text(x + bar_w / 2, y - 8, f"{v:.4f}", 11, "700", anchor="middle"))
        label = name.replace("CLER-G ", "CLER-G\n")
        if "\n" in label:
            a, b = label.split("\n", 1)
            body.append(text(cx, h - margin["b"] + 28, a, 14, "700", anchor="middle"))
            body.append(text(cx, h - margin["b"] + 48, b, 13, "700", anchor="middle"))
        else:
            body.append(text(cx, h - margin["b"] + 34, label, 14, "700", anchor="middle"))
    body.append(line(margin["l"], margin["t"], margin["l"], h - margin["b"], width=2))
    body.append(line(margin["l"], h - margin["b"], w - margin["r"], h - margin["b"], width=2))
    body.append(text(26, h / 2, "validation loss", 16, fill=COLORS["muted"], anchor="middle").replace("<text ", f'<text transform="rotate(-90 26 {h / 2:.2f})" ', 1))
    return save_svg("16_adamw_architecture_ladder_final_best_val", w, h, "\n".join(body), "AdamW final/best ladder")


def plot_adamw_gdn_family_best_zoom():
    names = ["GDN", "CLER-G", "CLER-G head", "CLER-G channel"]
    w, h = 1260, 680
    vals = [ADAMW_LADDER[n]["best"] for n in names]
    ymin, ymax = min(vals) - 0.0007, max(vals) + 0.0007
    margin = dict(l=105, r=40, t=115, b=120)
    pw = w - margin["l"] - margin["r"]
    ph = h - margin["t"] - margin["b"]

    def sy(v):
        return margin["t"] + (ymax - v) / (ymax - ymin) * ph

    body = [text(60, 60, "AdamW GDN-family best validation loss zoom", 32, "700")]
    body.append(text(60, 92, "Same numbers as the full AdamW ladder, zoomed to show the tiny CLER-G differences.", 16, fill=COLORS["muted"]))
    for i in range(6):
        v = ymin + (ymax - ymin) * i / 5
        y = sy(v)
        body.append(line(margin["l"], y, w - margin["r"], y, stroke=COLORS["grid"], width=1))
        body.append(text(margin["l"] - 12, y + 5, f"{v:.4f}", 13, fill=COLORS["muted"], anchor="end"))
    group_w = pw / len(names)
    bar_w = 92
    for i, name in enumerate(names):
        cx = margin["l"] + group_w * (i + 0.5)
        v = ADAMW_LADDER[name]["best"]
        y = sy(v)
        body.append(rect(cx - bar_w / 2, y, bar_w, h - margin["b"] - y, ADAMW_LADDER[name]["color"], stroke="#ffffff", rx=5))
        body.append(text(cx, y - 9, f"{v:.4f}", 13, "700", anchor="middle"))
        label = name.replace("CLER-G ", "CLER-G\n")
        if "\n" in label:
            a, b = label.split("\n", 1)
            body.append(text(cx, h - margin["b"] + 28, a, 14, "700", anchor="middle"))
            body.append(text(cx, h - margin["b"] + 48, b, 13, "700", anchor="middle"))
        else:
            body.append(text(cx, h - margin["b"] + 34, label, 14, "700", anchor="middle"))
    body.append(line(margin["l"], margin["t"], margin["l"], h - margin["b"], width=2))
    body.append(line(margin["l"], h - margin["b"], w - margin["r"], h - margin["b"], width=2))
    body.append(text(26, h / 2, "best validation loss", 16, fill=COLORS["muted"], anchor="middle").replace("<text ", f'<text transform="rotate(-90 26 {h / 2:.2f})" ', 1))
    return save_svg("19_adamw_gdn_family_best_val_zoom", w, h, "\n".join(body), "AdamW GDN family zoom")


def plot_validation_delta_muon_adamw_matched():
    pairs = [
        ("Muon\nCLER-G vs GDN", 2.833288 - 2.831940, 2.829291 - 2.828173, COLORS["cler"]),
        ("AdamW\nCLER-G vs GDN", 3.256151 - 3.255381, 3.252717 - 3.251918, COLORS["cler"]),
        ("Muon\nCLER-DN vs DN", 2.850695 - 2.851128, 2.846753 - 2.847952, COLORS["head"]),
        ("AdamW\nCLER-DN vs DN", 3.425838 - 3.406819, 3.424568 - 3.403113, COLORS["head"]),
    ]
    w, h = 1500, 760
    vals = [v for _, a, b, _ in pairs for v in (a, b)]
    ymax = max(abs(v) for v in vals) * 1.22
    margin = dict(l=120, r=40, t=120, b=130)
    pw = w - margin["l"] - margin["r"]
    ph = h - margin["t"] - margin["b"]

    def sy(v):
        return margin["t"] + (ymax - v) / (2 * ymax) * ph

    body = [text(60, 60, "Validation-loss deltas for matched CLER pairs: Muon and AdamW", 32, "700")]
    body.append(text(60, 92, "Positive is worse than the matched baseline; negative is better. Dark bars are final, light bars are best.", 16, fill=COLORS["muted"]))
    for i in range(7):
        v = -ymax + 2 * ymax * i / 6
        y = sy(v)
        body.append(line(margin["l"], y, w - margin["r"], y, stroke=COLORS["grid"], width=1))
        body.append(text(margin["l"] - 12, y + 5, f"{v:.3f}", 13, fill=COLORS["muted"], anchor="end"))
    body.append(line(margin["l"], sy(0), w - margin["r"], sy(0), stroke=COLORS["axis"], width=2, dash="7 5"))
    group_w = pw / len(pairs)
    bar_w = 58
    for i, (label, final, best, color) in enumerate(pairs):
        cx = margin["l"] + group_w * (i + 0.5)
        for j, (v, shade) in enumerate([(final, color), (best, color_lerp(color, "#ffffff", 0.35))]):
            x = cx + (j - 0.5) * (bar_w + 10) - bar_w / 2
            y0 = sy(0)
            y = sy(v)
            body.append(rect(x, min(y, y0), bar_w, abs(y0 - y), shade, stroke="#ffffff", rx=4))
            label_y = y - 8 if v >= 0 else y + 20
            body.append(text(x + bar_w / 2, label_y, f"{v:+.4f}", 12, "700", anchor="middle"))
        lines = label.split("\n")
        body.append(text(cx, h - margin["b"] + 30, lines[0], 14, "700", anchor="middle"))
        body.append(text(cx, h - margin["b"] + 52, lines[1], 14, "700", anchor="middle"))
    body.append(line(margin["l"], margin["t"], margin["l"], h - margin["b"], width=2))
    body.append(line(margin["l"], h - margin["b"], w - margin["r"], h - margin["b"], width=2))
    body.append(text(28, h / 2, "val loss delta", 16, fill=COLORS["muted"], anchor="middle").replace("<text ", f'<text transform="rotate(-90 28 {h / 2:.2f})" ', 1))
    return save_svg("17_validation_delta_matched_pairs_muon_adamw", w, h, "\n".join(body), "Muon and AdamW validation delta")


def plot_validation_delta_muon_adamw_small_zoom():
    pairs = [
        ("Muon\nCLER-G vs GDN", 2.833288 - 2.831940, 2.829291 - 2.828173, COLORS["cler"]),
        ("AdamW\nCLER-G vs GDN", 3.256151 - 3.255381, 3.252717 - 3.251918, COLORS["cler"]),
        ("Muon\nCLER-DN vs DN", 2.850695 - 2.851128, 2.846753 - 2.847952, COLORS["head"]),
    ]
    w, h = 1300, 720
    vals = [v for _, a, b, _ in pairs for v in (a, b)]
    ymax = max(abs(v) for v in vals) * 1.45
    margin = dict(l=120, r=40, t=120, b=130)
    pw = w - margin["l"] - margin["r"]
    ph = h - margin["t"] - margin["b"]

    def sy(v):
        return margin["t"] + (ymax - v) / (2 * ymax) * ph

    body = [text(60, 60, "Matched validation deltas: small-effect zoom", 32, "700")]
    body.append(text(60, 92, "Zoom excludes AdamW CLER-DN because that failure case is much larger. Dark bars are final, light bars are best.", 16, fill=COLORS["muted"]))
    for i in range(7):
        v = -ymax + 2 * ymax * i / 6
        y = sy(v)
        body.append(line(margin["l"], y, w - margin["r"], y, stroke=COLORS["grid"], width=1))
        body.append(text(margin["l"] - 12, y + 5, f"{v:.4f}", 13, fill=COLORS["muted"], anchor="end"))
    body.append(line(margin["l"], sy(0), w - margin["r"], sy(0), stroke=COLORS["axis"], width=2, dash="7 5"))
    group_w = pw / len(pairs)
    bar_w = 62
    for i, (label, final, best, color) in enumerate(pairs):
        cx = margin["l"] + group_w * (i + 0.5)
        for j, (v, shade) in enumerate([(final, color), (best, color_lerp(color, "#ffffff", 0.35))]):
            x = cx + (j - 0.5) * (bar_w + 12) - bar_w / 2
            y0 = sy(0)
            y = sy(v)
            body.append(rect(x, min(y, y0), bar_w, abs(y0 - y), shade, stroke="#ffffff", rx=4))
            body.append(text(x + bar_w / 2, y - 8 if v >= 0 else y + 20, f"{v:+.4f}", 13, "700", anchor="middle"))
        lines = label.split("\n")
        body.append(text(cx, h - margin["b"] + 30, lines[0], 14, "700", anchor="middle"))
        body.append(text(cx, h - margin["b"] + 52, lines[1], 14, "700", anchor="middle"))
    body.append(line(margin["l"], margin["t"], margin["l"], h - margin["b"], width=2))
    body.append(line(margin["l"], h - margin["b"], w - margin["r"], h - margin["b"], width=2))
    body.append(text(28, h / 2, "val loss delta", 16, fill=COLORS["muted"], anchor="middle").replace("<text ", f'<text transform="rotate(-90 28 {h / 2:.2f})" ', 1))
    return save_svg("18_validation_delta_small_effects_zoom", w, h, "\n".join(body), "Small validation deltas")


def plot_gamma_shape_magnitude():
    stats = {}
    for mode, path in GAMMA_FILES.items():
        row = final_jsonl(path)
        stats[mode] = {
            "count": row["count"],
            "abs_mean": row["abs_mean"],
            "max_abs": row["max_abs"],
        }
    w, h = 1420, 720
    body = [text(60, 60, "Gamma parameterization: count and final magnitudes", 32, "700")]
    body.append(text(60, 92, "Final sidecar stats at step 1900. More static gamma capacity did not translate into lower validation loss.", 16, fill=COLORS["muted"]))
    panels = [
        ("parameter count (log10)", "count", True, 90, 130, 580, 480),
        ("abs-mean / max-abs", "mag", False, 770, 130, 560, 480),
    ]
    modes = ["scalar", "head", "channel"]
    colors = [COLORS["cler"], COLORS["head"], COLORS["channel"]]
    for title, kind, logscale, x0, y0, pw, ph in panels:
        body.append(text(x0, y0 - 20, title, 20, "700"))
        if kind == "count":
            vals = [math.log10(stats[m]["count"]) for m in modes]
            vmax = max(vals) * 1.12
            label = lambda v: f"{10 ** v:.0f}" if v < 4 else f"{10 ** v / 1000:.0f}k"
            bars_per = 1
        else:
            vals = [stats[m]["abs_mean"] for m in modes] + [stats[m]["max_abs"] for m in modes]
            vmax = max(vals) * 1.18
            label = lambda v: f"{v:.3f}"
            bars_per = 2
        if kind == "count":
            tick_values = [1, 10, 100, 1000, 10000]
            for count_tick in tick_values:
                v = math.log10(count_tick)
                y = y0 + ph - v / vmax * ph
                body.append(line(x0, y, x0 + pw, y, stroke=COLORS["grid"], width=1))
                body.append(text(x0 - 10, y + 5, f"{count_tick}", 12, fill=COLORS["muted"], anchor="end"))
        else:
            for i in range(5):
                v = vmax * i / 4
                y = y0 + ph - v / vmax * ph
                body.append(line(x0, y, x0 + pw, y, stroke=COLORS["grid"], width=1))
                body.append(text(x0 - 10, y + 5, label(v), 12, fill=COLORS["muted"], anchor="end"))
        group_w = pw / len(modes)
        for i, m in enumerate(modes):
            cx = x0 + group_w * (i + 0.5)
            if kind == "count":
                v = math.log10(stats[m]["count"])
                bh = v / vmax * ph
                body.append(rect(cx - 35, y0 + ph - bh, 70, bh, colors[i], rx=5))
                body.append(text(cx, y0 + ph - bh - 8, f'{stats[m]["count"]}', 13, "700", anchor="middle"))
            else:
                for j, key in enumerate(["abs_mean", "max_abs"]):
                    v = stats[m][key]
                    bh = v / vmax * ph
                    shade = colors[i] if key == "abs_mean" else color_lerp(colors[i], "#ffffff", 0.35)
                    body.append(rect(cx + (j - 0.5) * 44 - 18, y0 + ph - bh, 36, bh, shade, rx=4))
                    body.append(text(cx + (j - 0.5) * 44, y0 + ph - bh - 8, f"{v:.3f}", 11, "700", anchor="middle"))
            body.append(text(cx, y0 + ph + 30, m, 15, "700", anchor="middle"))
        body.append(line(x0, y0, x0, y0 + ph, width=2))
        body.append(line(x0, y0 + ph, x0 + pw, y0 + ph, width=2))
    body.append(rect(610, 160, 145, 62, "#f8f9fb", stroke="#dde1e7", rx=8))
    body.append(text(625, 184, "dark: abs-mean", 13))
    body.append(text(625, 207, "light: max-abs", 13))
    return save_svg("06_gamma_shape_magnitude", w, h, "\n".join(body), "Gamma shape and magnitude")


def plot_gamma_evolution():
    series = []
    labels = [("scalar", COLORS["cler"]), ("head", COLORS["head"]), ("channel", COLORS["channel"])]
    for mode, color in labels:
        rows = read_jsonl(GAMMA_FILES[mode])
        xs = [r["step"] * TOKENS_PER_STEP / 1e9 for r in rows]
        ys = [r["abs_mean"] for r in rows]
        series.append((xs, ys, {"label": mode, "color": color}))
    return plot_line_chart(
        "07_gamma_abs_mean_evolution",
        "Gamma abs-mean evolution",
        series,
        y_label="abs-mean(Gamma)",
    )


def plot_residual_abs_mean():
    row = final_jsonl(RESIDUAL_FILES["channel"])
    layer_items = sorted(
        ((layer_from_name(k), v["abs_mean"], v["max_abs"]) for k, v in row["layers"].items()),
        key=lambda x: x[0],
    )
    w, h = 1380, 720
    max_abs_mean = max(v for _, v, _ in layer_items) * 1.18
    margin = dict(l=105, r=40, t=115, b=115)
    pw = w - margin["l"] - margin["r"]
    ph = h - margin["t"] - margin["b"]

    def sy(v):
        return margin["t"] + (max_abs_mean - v) / max_abs_mean * ph

    body = [text(60, 60, "Per-channel CLER: residual abs-mean by layer", 32, "700")]
    body.append(text(60, 92, "Final step 1900 sidecar. Residuals are nonzero; the no-effect result is not residual disappearance.", 16, fill=COLORS["muted"]))
    for i in range(6):
        v = max_abs_mean * i / 5
        y = sy(v)
        body.append(line(margin["l"], y, w - margin["r"], y, stroke=COLORS["grid"], width=1))
        body.append(text(margin["l"] - 12, y + 5, f"{v:.3f}", 13, fill=COLORS["muted"], anchor="end"))
    n = len(layer_items)
    group_w = pw / n
    bar_w = min(52, group_w * 0.62)
    for i, (layer, v, maxv) in enumerate(layer_items):
        cx = margin["l"] + group_w * (i + 0.5)
        y = sy(v)
        color = sequential_color(v, 0, max_abs_mean, "#edf4ef", COLORS["gdn"])
        body.append(rect(cx - bar_w / 2, y, bar_w, h - margin["b"] - y, color, rx=4))
        body.append(text(cx, y - 8, f"{v:.3f}", 11, "700", anchor="middle"))
        body.append(text(cx, h - margin["b"] + 26, str(layer), 13, "700", anchor="middle"))
    body.append(line(margin["l"], margin["t"], margin["l"], h - margin["b"], width=2))
    body.append(line(margin["l"], h - margin["b"], w - margin["r"], h - margin["b"], width=2))
    body.append(text(w / 2, h - 35, "CLER-capable layer index", 16, fill=COLORS["muted"], anchor="middle"))
    body.append(text(28, h / 2, "residual abs-mean", 16, fill=COLORS["muted"], anchor="middle").replace("<text ", f'<text transform="rotate(-90 28 {h / 2:.2f})" ', 1))
    return save_svg("08_residual_abs_mean_by_layer_channelgamma", w, h, "\n".join(body), "Residual abs-mean")


def collect_gamma_values(path: Path):
    row = final_jsonl(path)
    by_layer = {}
    for name, val in row["values"].items():
        layer = layer_from_name(name)
        idx = gamma_index_from_name(name)
        by_layer.setdefault(layer, {})[idx] = float(val)
    return row, by_layer


def plot_heatmap(name, title, matrix, row_labels, col_labels=None, width=1400, height=760, diverging=True, note=""):
    rows = len(matrix)
    cols = len(matrix[0]) if rows else 0
    margin = dict(l=135, r=110, t=125, b=90)
    pw = width - margin["l"] - margin["r"]
    ph = height - margin["t"] - margin["b"]
    cell_w = pw / cols
    cell_h = ph / rows
    flat = [v for r in matrix for v in r]
    vmax = max(abs(v) for v in flat) if flat else 1
    vmin, vmax_seq = (min(flat), max(flat)) if flat else (0, 1)
    body = [text(60, 60, title, 32, "700")]
    if note:
        body.append(text(60, 92, note, 16, fill=COLORS["muted"]))
    for r, row in enumerate(matrix):
        for c, v in enumerate(row):
            x = margin["l"] + c * cell_w
            y = margin["t"] + r * cell_h
            fill = diverging_color(v, vmax) if diverging else sequential_color(v, vmin, vmax_seq)
            body.append(rect(x, y, cell_w + 0.5, cell_h + 0.5, fill, stroke=fill, sw=0, rx=0))
    for r, lab in enumerate(row_labels):
        y = margin["t"] + (r + 0.5) * cell_h + 5
        body.append(text(margin["l"] - 10, y, lab, 12, fill=COLORS["muted"], anchor="end"))
    if col_labels:
        for c, lab in enumerate(col_labels):
            if c % max(1, cols // 8) == 0:
                x = margin["l"] + (c + 0.5) * cell_w
                body.append(text(x, height - margin["b"] + 25, lab, 12, fill=COLORS["muted"], anchor="middle"))
    body.append(line(margin["l"], margin["t"], margin["l"] + pw, margin["t"], width=1))
    body.append(line(margin["l"], margin["t"] + ph, margin["l"] + pw, margin["t"] + ph, width=1))
    body.append(line(margin["l"], margin["t"], margin["l"], margin["t"] + ph, width=1))
    body.append(line(margin["l"] + pw, margin["t"], margin["l"] + pw, margin["t"] + ph, width=1))
    # simple color legend
    leg_x, leg_y = width - 78, margin["t"]
    for i in range(120):
        t = i / 119
        v = vmax * (1 - 2 * t)
        fill = diverging_color(v, vmax) if diverging else sequential_color(vmin + (vmax_seq - vmin) * (1 - t), vmin, vmax_seq)
        body.append(rect(leg_x, leg_y + i * 3, 22, 3.2, fill, stroke=fill, sw=0, rx=0))
    body.append(text(leg_x + 34, leg_y + 8, f"+{vmax:.3f}" if diverging else f"{vmax_seq:.3f}", 12, fill=COLORS["muted"]))
    body.append(text(leg_x + 34, leg_y + 182, "0.000" if diverging else f"{(vmin+vmax_seq)/2:.3f}", 12, fill=COLORS["muted"]))
    body.append(text(leg_x + 34, leg_y + 358, f"-{vmax:.3f}" if diverging else f"{vmin:.3f}", 12, fill=COLORS["muted"]))
    return save_svg(name, width, height, "\n".join(body), title)


def plot_per_head_gamma():
    _, by_layer = collect_gamma_values(GAMMA_FILES["head"])
    layers = sorted(by_layer)
    matrix = []
    for layer in layers:
        vals = by_layer[layer]
        matrix.append([vals[i] for i in range(8)])
    return plot_heatmap(
        "09_per_head_gamma_final_heatmap",
        "Per-head CLER gamma at final sidecar step",
        matrix,
        [f"L{l}" for l in layers],
        [f"H{i}" for i in range(8)],
        width=980,
        height=760,
        note="Shape per CLER layer: [8]. Values are receiver-side static gamma, not GDN gates.",
    )


def plot_channel_gamma_heatmap():
    _, by_layer = collect_gamma_values(GAMMA_FILES["channel"])
    layer = 19 if 19 in by_layer else max(by_layer)
    vals = by_layer[layer]
    matrix = []
    for h in range(8):
        matrix.append([vals[h * 64 + d] for d in range(64)])
    return plot_heatmap(
        "10_channel_gamma_final_heatmap_layer19",
        f"Per-channel CLER gamma heatmap, layer {layer}",
        matrix,
        [f"H{h}" for h in range(8)],
        [str(i) for i in range(64)],
        width=1450,
        height=620,
        note="Shape per CLER layer: [8 heads, 64 value channels], broadcast over batch and sequence.",
    )


def plot_channel_layer_summary():
    _, by_layer = collect_gamma_values(GAMMA_FILES["channel"])
    layer_items = []
    for layer, vals in sorted(by_layer.items()):
        arr = list(vals.values())
        layer_items.append((layer, mean(arr), min(arr), max(arr), mean(abs(v) for v in arr)))
    w, h = 1380, 720
    margin = dict(l=105, r=40, t=115, b=115)
    pw = w - margin["l"] - margin["r"]
    ph = h - margin["t"] - margin["b"]
    vals = [x for _, a, b, c, d in layer_items for x in (a, b, c)]
    ymax = max(abs(v) for v in vals) * 1.2

    def sx(i):
        return margin["l"] + pw * i / (len(layer_items) - 1)

    def sy(v):
        return margin["t"] + (ymax - v) / (2 * ymax) * ph

    body = [text(60, 60, "Per-channel gamma by layer: mean and min/max range", 32, "700")]
    body.append(text(60, 92, "Layer 0 has no incoming CLER residual but still has parameters; downstream layers show small positive and negative channel-wise values.", 16, fill=COLORS["muted"]))
    for i in range(7):
        v = -ymax + 2 * ymax * i / 6
        y = sy(v)
        body.append(line(margin["l"], y, w - margin["r"], y, stroke=COLORS["grid"], width=1))
        body.append(text(margin["l"] - 12, y + 5, f"{v:.3f}", 13, fill=COLORS["muted"], anchor="end"))
    body.append(line(margin["l"], sy(0), w - margin["r"], sy(0), stroke=COLORS["axis"], width=1.5))
    mean_pts = []
    abs_pts = []
    for i, (layer, m, lo, hi, am) in enumerate(layer_items):
        x = sx(i)
        body.append(line(x, sy(lo), x, sy(hi), stroke=color_lerp(COLORS["channel"], "#ffffff", 0.25), width=4))
        body.append(circle(x, sy(lo), 4, COLORS["channel"], stroke=COLORS["channel"]))
        body.append(circle(x, sy(hi), 4, COLORS["channel"], stroke=COLORS["channel"]))
        mean_pts.append((x, sy(m)))
        abs_pts.append((x, sy(am)))
        body.append(text(x, h - margin["b"] + 26, str(layer), 13, "700", anchor="middle"))
    body.append(polyline(mean_pts, COLORS["cler"], width=3))
    for x, y in mean_pts:
        body.append(circle(x, y, 5, COLORS["cler"], stroke=COLORS["cler"]))
    body.append(polyline(abs_pts, COLORS["head"], width=2))
    body.append(rect(985, 130, 300, 85, "#f8f9fb", stroke="#dde1e7", rx=8))
    body.append(line(1010, 158, 1045, 158, stroke=COLORS["cler"], width=3))
    body.append(text(1060, 163, "mean gamma", 14))
    body.append(line(1010, 190, 1045, 190, stroke=COLORS["head"], width=2))
    body.append(text(1060, 195, "abs-mean gamma", 14))
    body.append(text(w / 2, h - 35, "CLER-capable layer index", 16, fill=COLORS["muted"], anchor="middle"))
    body.append(text(28, h / 2, "gamma value", 16, fill=COLORS["muted"], anchor="middle").replace("<text ", f'<text transform="rotate(-90 28 {h / 2:.2f})" ', 1))
    body.append(line(margin["l"], margin["t"], margin["l"], h - margin["b"], width=2))
    body.append(line(margin["l"], h - margin["b"], w - margin["r"], h - margin["b"], width=2))
    return save_svg("11_channel_gamma_layer_summary", w, h, "\n".join(body), "Channel gamma layer summary")


def load_40b_rows():
    main_path = CLER_40B_PERF_DIR / f"{CLER_40B_STEM}.jsonl"
    gamma_path = CLER_40B_PERF_DIR / f"{CLER_40B_STEM}.cler_gamma.jsonl"
    residual_path = CLER_40B_PERF_DIR / f"{CLER_40B_STEM}.cler_residual.jsonl"
    meta_path = CLER_40B_PERF_DIR / f"{CLER_40B_STEM}.meta.json"
    if not (main_path.exists() and gamma_path.exists() and residual_path.exists() and meta_path.exists()):
        raise FileNotFoundError(
            f"missing 40B CLER data under {CLER_40B_PERF_DIR}; set CLER_40B_PERF_DIR to the directory containing the sidecars"
        )
    main = read_jsonl(main_path)
    gamma = read_jsonl(gamma_path)
    residual = read_jsonl(residual_path)
    meta = json.loads(meta_path.read_text())
    return main, gamma, residual, meta


def closest_row(rows, target_tokens):
    target_step = target_tokens / CLER_40B_TOKENS_PER_STEP
    return min(rows, key=lambda row: abs(row["step"] - target_step))


def plot_40b_train_loss():
    rows, _, _, _ = load_40b_rows()
    loss_rows = [r for r in rows if "train_loss" in r]
    xs = [r["step"] * CLER_40B_TOKENS_PER_STEP / 1e9 for r in loss_rows]
    ys = smooth([r["train_loss"] for r in loss_rows], 100)
    return plot_line_chart(
        "21_40b_cler_dn_train_loss",
        "1.3B CLER-DeltaNet Muon: train loss over 40B tokens",
        [(xs, ys, {"label": "CLER-DN train loss", "color": COLORS["delta"]})],
        y_label="train loss",
        subtitle="100-step moving average from the transferred JSONL; the matched DeltaNet rerun is not included here.",
    )


def plot_40b_gamma_evolution():
    _, rows, _, _ = load_40b_rows()
    xs = [r["step"] * CLER_40B_TOKENS_PER_STEP / 1e9 for r in rows]
    return plot_line_chart(
        "22_40b_cler_dn_gamma_evolution",
        "CLER gamma stays small over 40B tokens",
        [
            (xs, [r["abs_mean"] for r in rows], {"label": "gamma abs mean", "color": COLORS["cler"]}),
            (xs, [r["max_abs"] for r in rows], {"label": "gamma max abs", "color": COLORS["head"]}),
        ],
        y_label="gamma magnitude",
        subtitle="Scalar receiver gamma from the 1.3B pure CLER-DeltaNet run; layer 0 is included in aggregate stats.",
        y_min_override=0.0,
    )


def plot_40b_residual_absmean_evolution():
    _, _, rows, _ = load_40b_rows()
    xs = [r["step"] * CLER_40B_TOKENS_PER_STEP / 1e9 for r in rows]
    return plot_line_chart(
        "23_40b_cler_dn_residual_absmean_evolution",
        "Routed residual mean magnitude over 40B tokens",
        [(xs, [r["abs_mean_mean"] for r in rows], {"label": "residual abs mean", "color": COLORS["channel"]})],
        y_label="residual abs mean",
        subtitle="Mean over layer-level residual absolute means; residuals are raw, not RMS-normalized.",
        y_min_override=0.0,
    )


def plot_40b_residual_maxabs_evolution():
    _, _, rows, _ = load_40b_rows()
    xs = [r["step"] * CLER_40B_TOKENS_PER_STEP / 1e9 for r in rows]
    return plot_line_chart(
        "24_40b_cler_dn_residual_maxabs_evolution",
        "Routed residual max-abs spikes over 40B tokens",
        [(xs, [r["max_abs"] for r in rows], {"label": "residual max abs", "color": COLORS["channel"]})],
        y_label="residual max abs",
        subtitle="Max across layer residual tensors; this reflects rare large components, not typical residual scale.",
        y_min_override=0.0,
    )


def plot_40b_final_gamma_by_layer():
    _, gamma_rows, _, _ = load_40b_rows()
    last = gamma_rows[-1]
    vals = {layer_from_name(k): float(v) for k, v in last["values"].items()}
    layers = sorted(vals)
    w, h = 1500, 760
    margin = dict(l=95, r=45, t=125, b=105)
    pw = w - margin["l"] - margin["r"]
    ph = h - margin["t"] - margin["b"]
    ymax = max(abs(v) for v in vals.values()) * 1.22

    def sx(i):
        return margin["l"] + pw * (i + 0.5) / len(layers)

    def sy(v):
        return margin["t"] + (ymax - v) / (2 * ymax) * ph

    body = [text(60, 60, "Final scalar CLER gamma by layer after 40B tokens", 32, "700")]
    body.append(text(60, 92, "Signed receiver gamma values; layer 0 remains zero because it has no lower CLER residual to consume.", 16, fill=COLORS["muted"]))
    for i in range(7):
        v = -ymax + 2 * ymax * i / 6
        y = sy(v)
        body.append(line(margin["l"], y, w - margin["r"], y, stroke=COLORS["grid"], width=1))
        body.append(text(margin["l"] - 12, y + 5, f"{v:.4f}", 13, fill=COLORS["muted"], anchor="end"))
    body.append(line(margin["l"], sy(0), w - margin["r"], sy(0), stroke=COLORS["axis"], width=1.8, dash="7 5"))
    bar_w = min(36, pw / len(layers) * 0.62)
    for i, layer in enumerate(layers):
        v = vals[layer]
        x = sx(i) - bar_w / 2
        y = sy(max(0, v))
        body.append(rect(x, y, bar_w, abs(sy(v) - sy(0)), COLORS["cler"] if v >= 0 else COLORS["head"], stroke="#ffffff", rx=4))
        body.append(text(sx(i), h - margin["b"] + 27, str(layer), 12, fill=COLORS["muted"], anchor="middle"))
    body.append(text(w / 2, h - 30, "layer", 16, fill=COLORS["muted"], anchor="middle"))
    body.append(text(27, h / 2, "gamma", 16, fill=COLORS["muted"], anchor="middle").replace("<text ", f'<text transform="rotate(-90 27 {h / 2:.2f})" ', 1))
    body.append(text(1060, 145, f"final abs mean = {last['abs_mean']:.4f}", 16, "700"))
    body.append(text(1060, 172, f"final max abs = {last['max_abs']:.4f}", 16, "700"))
    return save_svg("25_40b_cler_dn_final_gamma_by_layer", w, h, "\n".join(body), "40B final gamma by layer")


def plot_40b_injected_signal_proxy_by_layer():
    _, gamma_rows, residual_rows, _ = load_40b_rows()
    gamma_last = gamma_rows[-1]
    residual_last = residual_rows[-1]
    gammas = {layer_from_name(k): float(v) for k, v in gamma_last["values"].items()}
    res_abs = {layer_from_name(k): float(v["abs_mean"]) for k, v in residual_last["layers"].items()}
    layers = [l for l in sorted(gammas) if l > 0]
    gamma_abs = [abs(gammas[l]) for l in layers]
    proxy = [abs(gammas[l]) * res_abs.get(l - 1, 0.0) for l in layers]
    w, h = 1500, 760
    margin = dict(l=95, r=45, t=125, b=105)
    pw = w - margin["l"] - margin["r"]
    ph = h - margin["t"] - margin["b"]
    ymax = max(gamma_abs + proxy) * 1.18

    def sx(i, j):
        group_w = pw / len(layers)
        return margin["l"] + group_w * (i + 0.5) + (j - 0.5) * 21

    def sy(v):
        return margin["t"] + (ymax - v) / ymax * ph

    body = [text(60, 60, "Final CLER injected-signal proxy by receiver layer", 32, "700")]
    body.append(text(60, 92, "Proxy = abs(gamma_l) times abs-mean residual from layer l-1; raw summaries cannot recover exact tensor products.", 16, fill=COLORS["muted"]))
    for i in range(6):
        v = ymax * i / 5
        y = sy(v)
        body.append(line(margin["l"], y, w - margin["r"], y, stroke=COLORS["grid"], width=1))
        body.append(text(margin["l"] - 12, y + 5, f"{v:.4f}", 13, fill=COLORS["muted"], anchor="end"))
    body.append(line(margin["l"], margin["t"], margin["l"], h - margin["b"], width=2))
    body.append(line(margin["l"], h - margin["b"], w - margin["r"], h - margin["b"], width=2))
    bar_w = 18
    for i, layer in enumerate(layers):
        for j, (v, color) in enumerate(((gamma_abs[i], COLORS["head"]), (proxy[i], COLORS["gdn"]))):
            x = sx(i, j) - bar_w / 2
            y = sy(v)
            body.append(rect(x, y, bar_w, h - margin["b"] - y, color, stroke="#ffffff", rx=3))
        body.append(text(margin["l"] + pw * (i + 0.5) / len(layers), h - margin["b"] + 27, str(layer), 12, fill=COLORS["muted"], anchor="middle"))
    body.append(line(1050, 145, 1088, 145, stroke=COLORS["head"], width=6))
    body.append(text(1100, 150, "abs(gamma_l)", 15))
    body.append(line(1050, 174, 1088, 174, stroke=COLORS["gdn"], width=6))
    body.append(text(1100, 179, "abs(gamma_l) * E abs(r prev)", 15))
    body.append(text(w / 2, h - 30, "receiver layer", 16, fill=COLORS["muted"], anchor="middle"))
    body.append(text(27, h / 2, "magnitude", 16, fill=COLORS["muted"], anchor="middle").replace("<text ", f'<text transform="rotate(-90 27 {h / 2:.2f})" ', 1))
    summary = {
        "run": CLER_40B_STEM,
        "final_step": gamma_last["step"],
        "final_gamma_abs_mean": gamma_last["abs_mean"],
        "final_gamma_max_abs": gamma_last["max_abs"],
        "final_residual_abs_mean_mean": residual_last["abs_mean_mean"],
        "final_residual_max_abs": residual_last["max_abs"],
        "final_injected_signal_proxy_mean": mean(proxy),
        "final_injected_signal_proxy_max": max(proxy),
    }
    (OUT / "40b_cler_dn_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return save_svg("26_40b_cler_dn_injected_signal_proxy_by_layer", w, h, "\n".join(body), "40B CLER injected signal proxy")


def plot_40b_residual_by_layer_1b_vs_40b():
    _, _, residual_rows, _ = load_40b_rows()
    early = closest_row(residual_rows, 1e9)
    final = residual_rows[-1]
    early_vals = {layer_from_name(k): float(v["abs_mean"]) for k, v in early["layers"].items()}
    final_vals = {layer_from_name(k): float(v["abs_mean"]) for k, v in final["layers"].items()}
    layers = sorted(final_vals)
    w, h = 1500, 760
    margin = dict(l=95, r=45, t=125, b=105)
    pw = w - margin["l"] - margin["r"]
    ph = h - margin["t"] - margin["b"]
    ymax = max(list(early_vals.values()) + list(final_vals.values())) * 1.15

    def sx(i, j):
        group_w = pw / len(layers)
        return margin["l"] + group_w * (i + 0.5) + (j - 0.5) * 21

    def sy(v):
        return margin["t"] + (ymax - v) / ymax * ph

    body = [text(60, 60, "CLER residual abs-mean by layer: 1B vs 40B tokens", 32, "700")]
    body.append(text(60, 92, "Same 1.3B pure CLER-DeltaNet run; compares the sidecar closest to 1B tokens with the final sidecar.", 16, fill=COLORS["muted"]))
    for i in range(6):
        v = ymax * i / 5
        y = sy(v)
        body.append(line(margin["l"], y, w - margin["r"], y, stroke=COLORS["grid"], width=1))
        body.append(text(margin["l"] - 12, y + 5, f"{v:.3f}", 13, fill=COLORS["muted"], anchor="end"))
    body.append(line(margin["l"], margin["t"], margin["l"], h - margin["b"], width=2))
    body.append(line(margin["l"], h - margin["b"], w - margin["r"], h - margin["b"], width=2))
    bar_w = 18
    for i, layer in enumerate(layers):
        for j, (v, color) in enumerate(((early_vals[layer], COLORS["channel"]), (final_vals[layer], COLORS["delta"]))):
            x = sx(i, j) - bar_w / 2
            y = sy(v)
            body.append(rect(x, y, bar_w, h - margin["b"] - y, color, stroke="#ffffff", rx=3))
        body.append(text(margin["l"] + pw * (i + 0.5) / len(layers), h - margin["b"] + 27, str(layer), 12, fill=COLORS["muted"], anchor="middle"))
    early_b = early["step"] * CLER_40B_TOKENS_PER_STEP / 1e9
    final_b = final["step"] * CLER_40B_TOKENS_PER_STEP / 1e9
    body.append(line(1030, 145, 1068, 145, stroke=COLORS["channel"], width=6))
    body.append(text(1080, 150, f"{early_b:.2f}B tokens", 15))
    body.append(line(1030, 174, 1068, 174, stroke=COLORS["delta"], width=6))
    body.append(text(1080, 179, f"{final_b:.2f}B tokens", 15))
    body.append(text(w / 2, h - 30, "layer", 16, fill=COLORS["muted"], anchor="middle"))
    body.append(text(27, h / 2, "residual abs mean", 16, fill=COLORS["muted"], anchor="middle").replace("<text ", f'<text transform="rotate(-90 27 {h / 2:.2f})" ', 1))
    return save_svg("27_40b_cler_dn_residual_by_layer_1b_vs_40b", w, h, "\n".join(body), "40B residual by layer 1B vs 40B")


def plot_40b_gamma_and_proxy_1b_vs_40b():
    _, gamma_rows, residual_rows, _ = load_40b_rows()
    gamma_early = closest_row(gamma_rows, 1e9)
    gamma_final = gamma_rows[-1]
    residual_early = closest_row(residual_rows, 1e9)
    residual_final = residual_rows[-1]

    ge = {layer_from_name(k): float(v) for k, v in gamma_early["values"].items()}
    gf = {layer_from_name(k): float(v) for k, v in gamma_final["values"].items()}
    re = {layer_from_name(k): float(v["abs_mean"]) for k, v in residual_early["layers"].items()}
    rf = {layer_from_name(k): float(v["abs_mean"]) for k, v in residual_final["layers"].items()}
    layers = [l for l in sorted(gf) if l > 0]
    gamma_abs_early = [abs(ge[l]) for l in layers]
    gamma_abs_final = [abs(gf[l]) for l in layers]
    proxy_early = [abs(ge[l]) * re.get(l - 1, 0.0) for l in layers]
    proxy_final = [abs(gf[l]) * rf.get(l - 1, 0.0) for l in layers]

    w, h = 1500, 840
    margin = dict(l=95, r=45, t=125, b=105)
    pw = w - margin["l"] - margin["r"]
    ph = h - margin["t"] - margin["b"]
    panel_gap = 54
    panel_h = (ph - panel_gap) / 2
    ymax_g = max(gamma_abs_early + gamma_abs_final) * 1.18
    ymax_p = max(proxy_early + proxy_final) * 1.18

    def sx(i, j):
        group_w = pw / len(layers)
        return margin["l"] + group_w * (i + 0.5) + (j - 0.5) * 21

    def sy(v, panel, ymax):
        top = margin["t"] + panel * (panel_h + panel_gap)
        return top + (ymax - v) / ymax * panel_h

    body = [text(60, 60, "CLER gamma and injected-signal proxy: 1B vs 40B", 32, "700")]
    body.append(text(60, 92, "Gamma is much larger near 1B tokens than at the end; the estimated injected side-channel also shrinks.", 16, fill=COLORS["muted"]))

    for panel, ymax, ylabel in [(0, ymax_g, "abs gamma"), (1, ymax_p, "proxy")]:
        top = margin["t"] + panel * (panel_h + panel_gap)
        for i in range(5):
            v = ymax * i / 4
            y = sy(v, panel, ymax)
            body.append(line(margin["l"], y, w - margin["r"], y, stroke=COLORS["grid"], width=1))
            body.append(text(margin["l"] - 12, y + 5, f"{v:.4f}", 13, fill=COLORS["muted"], anchor="end"))
        body.append(line(margin["l"], top, margin["l"], top + panel_h, width=2))
        body.append(line(margin["l"], top + panel_h, w - margin["r"], top + panel_h, width=2))
        body.append(text(27, top + panel_h / 2, ylabel, 15, fill=COLORS["muted"], anchor="middle").replace("<text ", f'<text transform="rotate(-90 27 {top + panel_h / 2:.2f})" ', 1))

    bar_w = 18
    for i, layer in enumerate(layers):
        for panel, early_values, final_values, ymax in [
            (0, gamma_abs_early, gamma_abs_final, ymax_g),
            (1, proxy_early, proxy_final, ymax_p),
        ]:
            for j, (v, color) in enumerate(((early_values[i], COLORS["channel"]), (final_values[i], COLORS["delta"]))):
                x = sx(i, j) - bar_w / 2
                y = sy(v, panel, ymax)
                bottom = margin["t"] + panel * (panel_h + panel_gap) + panel_h
                body.append(rect(x, y, bar_w, bottom - y, color, stroke="#ffffff", rx=3))
        body.append(text(margin["l"] + pw * (i + 0.5) / len(layers), h - margin["b"] + 27, str(layer), 12, fill=COLORS["muted"], anchor="middle"))

    early_b = gamma_early["step"] * CLER_40B_TOKENS_PER_STEP / 1e9
    final_b = gamma_final["step"] * CLER_40B_TOKENS_PER_STEP / 1e9
    body.append(line(1050, 145, 1088, 145, stroke=COLORS["channel"], width=6))
    body.append(text(1100, 150, f"{early_b:.2f}B tokens", 15))
    body.append(line(1050, 174, 1088, 174, stroke=COLORS["delta"], width=6))
    body.append(text(1100, 179, f"{final_b:.2f}B tokens", 15))
    body.append(text(w / 2, h - 30, "receiver layer", 16, fill=COLORS["muted"], anchor="middle"))
    return save_svg("28_40b_cler_dn_gamma_proxy_1b_vs_40b", w, h, "\n".join(body), "40B gamma proxy 1B vs 40B")


def write_manifest(paths, include_40b=False):
    rel = [p.relative_to(ROOT) for p in paths]
    notes = [
        ("01_architecture_hybrid_vs_pure", "Architecture diagram: hybrid linear/SDPA pattern versus the queued pure-GDN follow-up."),
        ("02_cler_mechanism_diagram", "Mechanism diagram and implementation formula for CLER residual routing."),
        ("03_adamw_train_loss_capacity_curves", "Offline train-loss curves from JSONL logs for GDN/scalar/head/channel AdamW."),
        ("03b_adamw_train_loss_delta_vs_gdn", "Offline smoothed train-loss delta versus GDN, with the matched GDN baseline shown as y=0."),
        ("03c_adamw_train_loss_zoom_with_gdn_baseline", "Offline smoothed actual train-loss zoom after warmup, with GDN baseline drawn as a dashed normal curve."),
        ("04_validation_delta_vs_gdn_adamw", "Derived validation-loss deltas versus the matched GDN AdamW baseline; positive is worse."),
        ("05_muon_architecture_ladder_best_val", "Muon architecture ladder best validation loss; useful context, not a CLER win claim."),
        ("06_gamma_shape_magnitude", "Gamma parameter count and final magnitude statistics from CLER sidecars."),
        ("07_gamma_abs_mean_evolution", "Gamma abs-mean evolution from sidecar logs."),
        ("08_residual_abs_mean_by_layer_channelgamma", "Per-channel CLER residual abs-mean per layer at final sidecar step."),
        ("09_per_head_gamma_final_heatmap", "Per-head gamma heatmap at final sidecar step."),
        ("10_channel_gamma_final_heatmap_layer19", "Per-channel gamma heatmap for layer 19 at final sidecar step."),
        ("11_channel_gamma_layer_summary", "Per-channel gamma mean/range by layer at final sidecar step."),
        ("12_muon_train_loss_delta_vs_gdn", "Muon smoothed train-loss delta for scalar CLER-Gated versus GDN, with baseline shown as y=0."),
        ("12b_muon_train_loss_zoom_with_gdn_baseline", "Muon smoothed actual train-loss zoom after warmup, with GDN baseline drawn as a dashed normal curve."),
        ("13_muon_train_loss_delta_vs_deltanet", "Muon smoothed train-loss delta for scalar CLER-DeltaNet versus DeltaNet, with baseline shown as y=0."),
        ("13b_muon_train_loss_zoom_with_deltanet_baseline", "Muon smoothed actual train-loss zoom after warmup, with DeltaNet baseline drawn as a dashed normal curve."),
        ("14_muon_validation_delta_matched_pairs", "Muon final/best validation-loss deltas for the matched CLER-G and CLER-DN comparisons."),
        ("15_adamw_architecture_ladder_best_val", "AdamW architecture ladder best validation loss, analogous to the Muon ladder plot."),
        ("16_adamw_architecture_ladder_final_best_val", "AdamW architecture ladder final and best validation loss side by side."),
        ("17_validation_delta_matched_pairs_muon_adamw", "Matched CLER validation-loss deltas for both Muon and AdamW in one plot."),
        ("18_validation_delta_small_effects_zoom", "Zoomed matched validation-loss deltas for the small effects, excluding the much larger AdamW CLER-DN failure case."),
        ("19_adamw_gdn_family_best_val_zoom", "Zoomed AdamW GDN-family best validation loss to make scalar/head/channel CLER-G differences readable."),
        ("20_cler_architecture_transformer_style", "Report-style CLER architecture diagram showing the normal Transformer stream, linear-memory residuals, and the cross-layer CLER side-channel."),
    ]
    if include_40b:
        notes.extend(
            [
                ("21_40b_cler_dn_train_loss", "1.3B CLER-DeltaNet Muon train-loss curve over the transferred 40B-token run."),
                ("22_40b_cler_dn_gamma_evolution", "Scalar CLER gamma abs-mean and max-abs over the 40B-token run."),
                ("23_40b_cler_dn_residual_absmean_evolution", "Mean routed-residual absolute magnitude over the 40B-token run."),
                ("24_40b_cler_dn_residual_maxabs_evolution", "Maximum routed-residual absolute value over the 40B-token run, showing rare large components."),
                ("25_40b_cler_dn_final_gamma_by_layer", "Final signed scalar CLER gamma value per layer after 40B tokens."),
                ("26_40b_cler_dn_injected_signal_proxy_by_layer", "Final per-layer injected-signal proxy: absolute gamma times previous-layer residual abs mean."),
                ("27_40b_cler_dn_residual_by_layer_1b_vs_40b", "Same-run residual abs-mean by layer near 1B tokens versus final 40B tokens."),
                ("28_40b_cler_dn_gamma_proxy_1b_vs_40b", "Same-run gamma magnitude and injected-signal proxy near 1B tokens versus final 40B tokens."),
            ]
        )
    md = [
        "# CLER Slide Plots",
        "",
        "Generated by `_research/plotting/make_cler_plots.py`.",
        "",
        "All plots use shared JSONL logs, sidecars, or tracked run summaries under the repository tracker. These are intended for slide material that is not directly available as W&B charts.",
        "",
        "## Files",
        "",
    ]
    existing = {p.stem for p in paths}
    for stem, note in notes:
        svg = f"{stem}.svg"
        png = f"{stem}.png"
        md.append(f"- `{svg}`: {note}")
        if (OUT / png).exists():
            md.append(f"- `{png}`: PNG copy of `{svg}` for slide tools.")
    md.extend(
        [
            "",
            "## Main Source Runs",
            "",
            "| purpose | run/job | local file |",
            "|---|---:|---|",
            "| GDN AdamW baseline | 2236100 | `gdn-pytorch-350m-llama2-fwe1b-adamw-compile10h-2236100.jsonl` |",
            "| scalar CLER-G AdamW | 2236101 | `cler-gated-v1-carry-350m-llama2-fwe1b-adamw-compile10h-2236101.*` |",
            "| per-head CLER-G AdamW | 2282970 | `cler-gated-v1-headgamma-350m-llama2-fwe1b-adamw-gamma1e-2-compile10h-2282970.*` |",
            "| per-channel CLER-G AdamW | 2291777 | `cler-gated-v1-channelgamma-350m-llama2-fwe1b-adamw-gamma1e-2-compile10h-2291777.*` |",
            "| DeltaNet AdamW baseline | 2236098 | `deltanet-pytorch-350m-llama2-fwe1b-adamw-compile10h-2236098.jsonl` |",
            "| scalar CLER-DN AdamW | 2236099 | `cler-deltanet-v1-carry-350m-llama2-fwe1b-adamw-compile10h-2236099.*` |",
            "| GDN Muon baseline | 2019280 | scratch log `350m-fwe1b-gdn-muon-c10h-2019280.log` |",
            "| scalar CLER-G Muon | 2059790 | scratch `350m-llama2-fwe1b-cler-gated-v1-carry-muon-compile10h-2059790.*` |",
            "| DeltaNet Muon baseline | 2065614 | scratch `350m-llama2-fwe1b-deltanet-pytorch-muon-compile10h-2065614.jsonl` |",
            "| scalar CLER-DN Muon | 2059791 | scratch `350m-llama2-fwe1b-cler-deltanet-pytorch-muon-compile10h-2059791.*` |",
        "",
            "## How To Regenerate",
            "",
            "```bash",
            "python3 /users/course_00252/cler/_research/plotting/make_cler_plots.py",
            "```",
            "",
        ]
    )
    if include_40b:
        md.extend(
            [
                "## 40B Shared Logs",
                "",
                "| purpose | run/job | shared file |",
                "|---|---:|---|",
                "| 1.3B scalar CLER-DN Muon 40B-token run | 2355856 | `CLER_40B_PERF_DIR/1.3B-CLER-DN-MUON-970296930-2355856.*` |",
                "",
            ]
        )
    (OUT / "README.md").write_text("\n".join(md), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    have_40b = True
    try:
        load_40b_rows()
    except FileNotFoundError:
        have_40b = False
    paths = [
        plot_architecture(),
        plot_mechanism(),
        plot_training_curves(),
        plot_training_delta_vs_gdn(),
        plot_adamw_train_loss_zoom_with_baseline(),
        plot_validation_delta(),
        plot_muon_ladder(),
        plot_gamma_shape_magnitude(),
        plot_gamma_evolution(),
        plot_residual_abs_mean(),
        plot_per_head_gamma(),
        plot_channel_gamma_heatmap(),
        plot_channel_layer_summary(),
        plot_muon_delta_vs_gdn(),
        plot_muon_train_loss_zoom_with_gdn_baseline(),
        plot_muon_delta_vs_deltanet(),
        plot_muon_train_loss_zoom_with_deltanet_baseline(),
        plot_muon_validation_delta(),
        plot_adamw_ladder_best(),
        plot_adamw_ladder_final_best(),
        plot_validation_delta_muon_adamw_matched(),
        plot_validation_delta_muon_adamw_small_zoom(),
        plot_adamw_gdn_family_best_zoom(),
        plot_cler_transformer_style(),
    ]
    if have_40b:
        paths.extend(
            [
                plot_40b_train_loss(),
                plot_40b_gamma_evolution(),
                plot_40b_residual_absmean_evolution(),
                plot_40b_residual_maxabs_evolution(),
                plot_40b_final_gamma_by_layer(),
                plot_40b_injected_signal_proxy_by_layer(),
                plot_40b_residual_by_layer_1b_vs_40b(),
                plot_40b_gamma_and_proxy_1b_vs_40b(),
            ]
        )
    write_manifest(paths, include_40b=have_40b)
    print("Wrote:")
    for p in paths:
        print(p)
    print(OUT / "README.md")


if __name__ == "__main__":
    main()
