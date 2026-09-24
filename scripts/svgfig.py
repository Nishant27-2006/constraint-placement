"""Minimal publication-grade SVG chart toolkit.

Design rules followed here, and why:

* **Encode by position first.** Every quantitative comparison in this paper is a
  position along a common scale (Cleveland & McGill 1984). No pie, no area, no 3D.
* **Never rely on colour alone.** Every series carries a distinct *marker shape*
  in addition to hue, and series are labelled directly on the plot rather than
  through a detached legend where possible.
* **Colourblind-safe palette.** Okabe-Ito, which is distinguishable under
  protanopia, deuteranopia and tritanopia.
* **SVG.** Every figure here is well under 1000 marks, so SVG is the right
  renderer: crisp at any zoom, DOM events, and ARIA for screen readers.
* **Annotate the insight.** Each figure carries a short annotation naming what
  the reader should take away, not merely what is plotted.
"""
from __future__ import annotations
import math
from typing import Iterable, Sequence

# Okabe-Ito, colourblind-safe.  Ordered so the first three are maximally distinct.
OKABE = {
    "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
    "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7", "black": "#000000",
}
SERIES = [OKABE["vermillion"], OKABE["blue"], OKABE["green"], OKABE["purple"],
          OKABE["sky"], OKABE["orange"]]


def esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


class Scale:
    """Linear or log scale mapping data -> pixels."""

    def __init__(self, d0, d1, r0, r1, log=False, pad=0.0):
        if log:
            d0, d1 = math.log10(max(d0, 1e-12)), math.log10(max(d1, 1e-12))
        span = (d1 - d0) or 1.0
        d0, d1 = d0 - span * pad, d1 + span * pad
        self.d0, self.d1, self.r0, self.r1, self.log = d0, d1, r0, r1, log

    def __call__(self, v):
        if self.log:
            v = math.log10(max(v, 1e-12))
        t = (v - self.d0) / ((self.d1 - self.d0) or 1.0)
        return self.r0 + t * (self.r1 - self.r0)

    def ticks(self, n=5):
        if self.log:
            lo, hi = math.ceil(self.d0), math.floor(self.d1)
            return [10 ** k for k in range(int(lo), int(hi) + 1)]
        return _nice_ticks(self.d0, self.d1, n)


def _nice_ticks(lo, hi, n=5):
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw / mag <= m:
            step = m * mag
            break
    else:
        step = 10 * mag
    start = math.ceil(lo / step) * step
    out, v = [], start
    while v <= hi + step * 1e-9:
        out.append(round(v, 10))
        v += step
    return out


def marker(shape: str, cx: float, cy: float, r: float, fill: str, stroke: str,
           sw: float = 1.8, title: str = "") -> str:
    """A data mark. Shape is the redundant encoding that survives colourblindness."""
    t = f"<title>{esc(title)}</title>" if title else ""
    a = f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"'
    if shape == "circle":
        return f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" {a}>{t}</circle>'
    if shape == "square":
        return (f'<rect x="{cx-r:.2f}" y="{cy-r:.2f}" width="{2*r:.2f}" '
                f'height="{2*r:.2f}" {a}>{t}</rect>')
    if shape == "triangle":
        p = f"{cx:.2f},{cy-r*1.15:.2f} {cx+r:.2f},{cy+r*0.8:.2f} {cx-r:.2f},{cy+r*0.8:.2f}"
        return f'<polygon points="{p}" {a}>{t}</polygon>'
    if shape == "diamond":
        p = (f"{cx:.2f},{cy-r*1.25:.2f} {cx+r*1.1:.2f},{cy:.2f} "
             f"{cx:.2f},{cy+r*1.25:.2f} {cx-r*1.1:.2f},{cy:.2f}")
        return f'<polygon points="{p}" {a}>{t}</polygon>'
    if shape == "cross":
        return (f'<path d="M{cx-r:.2f},{cy-r:.2f}L{cx+r:.2f},{cy+r:.2f}'
                f'M{cx-r:.2f},{cy+r:.2f}L{cx+r:.2f},{cy-r:.2f}" stroke="{stroke}" '
                f'stroke-width="{sw+0.6}" fill="none" stroke-linecap="round">{t}</path>')
    return f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" {a}>{t}</circle>'


SHAPES = ["circle", "square", "triangle", "diamond", "cross"]


class Figure:
    """An SVG figure with a plot frame, ticks, gridlines and ARIA metadata."""

    def __init__(self, w, h, ml, mr, mt, mb, title, desc, fid):
        self.w, self.h = w, h
        self.x0, self.x1 = ml, w - mr
        self.y0, self.y1 = mt, h - mb
        self.fid = fid
        self.o = [
            f'<svg viewBox="0 0 {w} {h}" class="fig" role="img" '
            f'aria-labelledby="{fid}-t {fid}-d" preserveAspectRatio="xMidYMid meet">',
            f'<title id="{fid}-t">{esc(title)}</title>',
            f'<desc id="{fid}-d">{esc(desc)}</desc>',
        ]

    def frame(self, fill="var(--plot-bg)"):
        self.o.append(f'<rect x="{self.x0}" y="{self.y0}" width="{self.x1-self.x0}" '
                      f'height="{self.y1-self.y0}" fill="{fill}" rx="4"/>')
        return self

    def grid_y(self, sy: Scale, ticks: Iterable[float], fmt=lambda v: f"{v:g}",
               label_every=1):
        for i, v in enumerate(ticks):
            y = sy(v)
            if not (self.y0 - 1 <= y <= self.y1 + 1):
                continue
            self.o.append(f'<line x1="{self.x0}" y1="{y:.2f}" x2="{self.x1}" '
                          f'y2="{y:.2f}" class="grid"/>')
            if i % label_every == 0:
                self.o.append(f'<text x="{self.x0-9}" y="{y+4:.2f}" class="tick" '
                              f'text-anchor="end">{esc(fmt(v))}</text>')
        return self

    def grid_x(self, sx: Scale, ticks: Iterable[float], fmt=lambda v: f"{v:g}",
               show_lines=True):
        for v in ticks:
            x = sx(v)
            if not (self.x0 - 1 <= x <= self.x1 + 1):
                continue
            if show_lines:
                self.o.append(f'<line x1="{x:.2f}" y1="{self.y0}" x2="{x:.2f}" '
                              f'y2="{self.y1}" class="grid"/>')
            self.o.append(f'<text x="{x:.2f}" y="{self.y1+20:.2f}" class="tick" '
                          f'text-anchor="middle">{esc(fmt(v))}</text>')
        return self

    def rule(self, sy: Scale, v: float, label: str = "", dash="5 4"):
        y = sy(v)
        self.o.append(f'<line x1="{self.x0}" y1="{y:.2f}" x2="{self.x1}" y2="{y:.2f}" '
                      f'class="rule" stroke-dasharray="{dash}"/>')
        if label:
            self.o.append(f'<text x="{self.x1-6}" y="{y-7:.2f}" class="rule-lbl" '
                          f'text-anchor="end">{esc(label)}</text>')
        return self

    def axis_titles(self, x: str = "", y: str = ""):
        if x:
            cx = (self.x0 + self.x1) / 2
            self.o.append(f'<text x="{cx:.0f}" y="{self.h-8}" class="axis" '
                          f'text-anchor="middle">{esc(x)}</text>')
        if y:
            cy = (self.y0 + self.y1) / 2
            self.o.append(f'<text x="15" y="{cy:.0f}" class="axis" text-anchor="middle" '
                          f'transform="rotate(-90 15 {cy:.0f})">{esc(y)}</text>')
        return self

    def path(self, pts: Sequence[tuple], color: str, width=2.0, dash=None, op=1.0):
        if len(pts) < 2:
            return self
        d = " ".join(f"{'M' if i == 0 else 'L'}{x:.2f},{y:.2f}"
                     for i, (x, y) in enumerate(pts))
        da = f' stroke-dasharray="{dash}"' if dash else ""
        self.o.append(f'<path d="{d}" fill="none" stroke="{color}" '
                      f'stroke-width="{width}"{da} opacity="{op}" '
                      f'stroke-linejoin="round" stroke-linecap="round"/>')
        return self

    def errbar_v(self, x, ylo, yhi, color, cap=4, width=1.6):
        self.o.append(f'<path d="M{x:.2f},{ylo:.2f}L{x:.2f},{yhi:.2f}'
                      f'M{x-cap:.2f},{ylo:.2f}L{x+cap:.2f},{ylo:.2f}'
                      f'M{x-cap:.2f},{yhi:.2f}L{x+cap:.2f},{yhi:.2f}" '
                      f'stroke="{color}" stroke-width="{width}" fill="none" '
                      f'opacity=".85"/>')
        return self

    def mark(self, shape, x, y, color, r=5.5, filled=True, title=""):
        self.o.append(marker(shape, x, y, r, color if filled else "var(--plot-bg)",
                             color, 1.9, title))
        return self

    def label(self, x, y, text, color="var(--fg)", anchor="start", cls="lbl", dy=0):
        self.o.append(f'<text x="{x:.2f}" y="{y+dy:.2f}" class="{cls}" fill="{color}" '
                      f'text-anchor="{anchor}">{esc(text)}</text>')
        return self

    def note(self, x, y, lines, anchor="start", cls="note"):
        """Short annotation naming the insight."""
        for i, ln in enumerate(lines):
            self.o.append(f'<text x="{x:.2f}" y="{y + i*14:.2f}" class="{cls}" '
                          f'text-anchor="{anchor}">{esc(ln)}</text>')
        return self

    def arrow(self, x1, y1, x2, y2, color="var(--muted)"):
        self.o.append(
            f'<defs><marker id="{self.fid}-ah" markerWidth="7" markerHeight="7" '
            f'refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 z" '
            f'fill="{color}"/></marker></defs>'
            f'<path d="M{x1:.2f},{y1:.2f}L{x2:.2f},{y2:.2f}" stroke="{color}" '
            f'stroke-width="1.4" fill="none" marker-end="url(#{self.fid}-ah)"/>')
        return self

    def legend(self, items, x, y, cols=1, dy=17):
        """Used only where direct labelling would overplot."""
        for i, (shape, color, txt, filled) in enumerate(items):
            r_, c_ = divmod(i, cols) if cols > 1 else (i, 0)
            lx, ly = x + c_ * 150, y + r_ * dy
            self.o.append(marker(shape, lx, ly - 4, 4.6,
                                 color if filled else "var(--plot-bg)", color, 1.8))
            self.o.append(f'<text x="{lx+11:.2f}" y="{ly:.2f}" class="legend-t">'
                          f'{esc(txt)}</text>')
        return self

    def done(self):
        self.o.append("</svg>")
        return "".join(self.o)


# ---------------------------------------------------------------- labelling
FONT_SIZE = {"tick": 12.0, "lbl": 12.0, "series-lbl": 12.5, "axis": 12.5,
             "note": 11.5, "legend-t": 11.5, "rule-lbl": 11.0}
CHAR_W = 0.56          # mean glyph width / font size for Noto Sans
LINE_H = 1.15


def text_box(x, y, s, cls="lbl", anchor="start"):
    fs = FONT_SIZE.get(cls, 12.0)
    w, h = len(s) * fs * CHAR_W, fs * LINE_H
    x0 = x if anchor == "start" else (x - w if anchor == "end" else x - w / 2)
    return [x0, y - h * 0.8, x0 + w, y + h * 0.2]


def _hit(a, b, pad=1.5):
    return not (a[2] + pad <= b[0] or b[2] + pad <= a[0]
                or a[3] + pad <= b[1] or b[3] + pad <= a[1])


class LabelPlacer:
    """Greedy point-feature label placement with leader lines.

    Standard approach: for each anchor try a ring of candidate offsets in
    preference order, take the first that collides with nothing already placed
    and stays inside the frame. If every candidate fails, push the label out to
    a free slot and draw a leader line to it. Labels that still cannot be placed
    are reported so the caller can fall back to a legend rather than emit a
    collision.
    """

    RING = [(10, 4, "start"), (-10, 4, "end"), (10, -9, "start"), (10, 15, "start"),
            (-10, -9, "end"), (-10, 15, "end"), (0, -12, "middle"), (0, 18, "middle"),
            (14, -20, "start"), (-14, -20, "end"), (14, 24, "start"), (-14, 24, "end"),
            (22, -32, "start"), (-22, -32, "end"), (22, 36, "start"), (-22, 36, "end")]

    def __init__(self, fig, bounds=None):
        self.f = fig
        self.taken = []
        self.bounds = bounds or (fig.x0 - 60, fig.y0 - 26, fig.x1 + 168, fig.y1 + 26)

    def reserve_box(self, box):
        self.taken.append(list(box))

    def reserve_mark(self, x, y, r=7.5):
        self.taken.append([x - r, y - r, x + r, y + r])

    def reserve_text(self, x, y, s, cls="lbl", anchor="start"):
        self.taken.append(text_box(x, y, s, cls, anchor))

    def _free(self, box):
        b = self.bounds
        if box[0] < b[0] or box[2] > b[2] or box[1] < b[1] or box[3] > b[3]:
            return False
        return not any(_hit(box, t) for t in self.taken)

    def place(self, ax, ay, s, color="var(--fg)", cls="lbl", leader=True,
              ring=None):
        """Place `s` near (ax, ay). Returns True if placed."""
        for dx, dy, anchor in (ring or self.RING):
            x, y = ax + dx, ay + dy
            box = text_box(x, y, s, cls, anchor)
            if self._free(box):
                far = abs(dx) > 16 or abs(dy) > 20
                if far and leader:
                    tx = box[0] - 3 if anchor == "start" else (
                        box[2] + 3 if anchor == "end" else (box[0] + box[2]) / 2)
                    ty = (box[1] + box[3]) / 2
                    self.f.o.append(
                        f'<path d="M{ax:.2f},{ay:.2f}L{tx:.2f},{ty:.2f}" '
                        f'stroke="{color}" stroke-width="0.9" fill="none" '
                        f'opacity=".55"/>')
                self.f.label(x, y, s, color, anchor, cls)
                self.taken.append(box)
                return True
        return False
