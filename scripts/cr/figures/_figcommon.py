"""Shared helpers for the camera-ready figure scripts (scripts/cr/figures/fig_*.py).

Run every figure script from the repository root, e.g. `python scripts/cr/figures/fig_fig3_interventional_forest.py`, or all
four with `bash scripts/cr/figures/make_figures.sh`. Scripts read only repository files (results/cr/*.csv, data/), make no
network or language-model calls, and are deterministic. Output: figures/<name>.png at 300 dpi plus a vector figures/<name>.pdf (the paper includes the PDF);
file names are those of the submitted version. Main-text figures are laid out at the 5.5 in NeurIPS text width so that
the font ladder below is the printed size.

Style: role-mapped font ladder (base 8 pt for titles/axis labels, 7 pt for legends/annotations, 6 pt for ticks), open frame,
frameless legends, sentence titles left-aligned. The focal colour (FOCAL) marks the arm the paper is about (language model
or intervention shown); registry/comparator arms are grey; the single alarm colour (ACC) is reserved for nominal/target
reference lines and the one operating-point marker.
"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
CR = os.path.join(ROOT, 'results', 'cr')
OUT = os.path.join(ROOT, 'figures')
TABLES = os.path.join(ROOT, 'scripts', 'cr', 'tables')

FOCAL, GREY, GREY2, ACC, BAR = '#2c5f8a', '#9a9a9a', '#8a8a8a', '#b5423b', '#3f6f99'
SIZES = (8, 7, 6)


def apply_style(sizes=SIZES):
    base, secondary, tick = sizes
    matplotlib.rcParams.update({
        'font.family': 'sans-serif', 'font.size': base, 'axes.labelsize': base, 'axes.titlesize': base,
        'legend.fontsize': secondary, 'xtick.labelsize': tick, 'ytick.labelsize': tick,
        'axes.linewidth': 0.6, 'xtick.direction': 'out', 'ytick.direction': 'out',
        'xtick.major.size': 3, 'ytick.major.size': 3, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': False, 'legend.frameon': False,
        'figure.dpi': 200, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'axes.titleweight': 'normal',
        'axes.titlelocation': 'left', 'axes.labelweight': 'normal', 'lines.linewidth': 1.2, 'patch.linewidth': 0.6,
        'pdf.fonttype': 42, 'ps.fonttype': 42})


def cr(label):
    return pd.read_csv(os.path.join(CR, f'{label}.csv'))


def letter(ax, s, x=-0.08, y=1.10):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, fontweight='bold', va='top', ha='right')


def interval(ax, y, est, lo, hi, color, ms=5.5, lw=2.2, marker='o', zorder=3, **kw):
    ax.plot([lo, hi], [y, y], color=color, lw=lw, solid_capstyle='round', zorder=zorder - 1)
    ax.plot(est, y, marker, color=color, ms=ms, zorder=zorder, **kw)


def check_layout(fig, name):
    """Geometric overlap check: no two visible text boxes may overlap, no text may cross a spine (tick labels on their own
    axes' spines excepted), and every text box must lie inside the figure canvas. Raises on failure so make_figures.sh stops."""
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    texts = [(t, t.get_window_extent(r)) for t in fig.findobj(matplotlib.text.Text) if t.get_text().strip() and t.get_visible()]
    texts = [(t, b) for t, b in texts if b.width > 0 and b.height > 0]
    tick = {ax: set(ax.get_xticklabels(which='both') + ax.get_yticklabels(which='both')) for ax in fig.axes}
    alltick = set().union(*tick.values())
    bad = []
    for i, (a, ba) in enumerate(texts):
        for b, bb in texts[i + 1:]:
            # rotated tick labels on one axis: the axis-aligned bbox of rotated text overstates its footprint, so adjacent
            # rotated tick labels are checked visually (Sec. 9.2) rather than geometrically.
            if a in alltick and b in alltick and a.axes is b.axes and (a.get_rotation() or b.get_rotation()):
                continue
            if ba.overlaps(bb):
                bad.append(('text/text', a.get_text(), b.get_text()))
    for ax in fig.axes:
        for s in ax.spines.values():
            if not s.get_visible():
                continue
            bs = s.get_window_extent(r)
            for t, bt in texts:
                if t.axes is ax and t in tick[ax]:
                    continue
                if bt.overlaps(bs):
                    bad.append(('text/spine', t.get_text(), ax.get_title() or 'axes'))
    if bad:
        for b in bad:
            print('LAYOUT', name, b)
        raise SystemExit(f'{name}: {len(bad)} layout problem(s)')


def fit_width(fig, width=5.5, iters=4):
    """Resize the canvas so that the tight bounding box (what is saved) is `width` inches wide. Text does not scale with the
    canvas, so a few iterations converge; the result prints at 100% when included at \\linewidth (5.5 in in the NeurIPS style)."""
    for _ in range(iters):
        fig.canvas.draw(); r = fig.canvas.get_renderer()
        bb = fig.get_tightbbox(r); w_now = bb.width
        w_fig, h_fig = fig.get_size_inches()
        if abs(w_now - width) < 0.02:
            break
        fig.set_size_inches(w_fig + (width - w_now), h_fig)
    fig.canvas.draw(); return fig.get_tightbbox(fig.canvas.get_renderer())


def save(fig, name, width=5.5):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    for t in fig.findobj(matplotlib.text.Text):
        assert '\u2014' not in t.get_text(), 'em dash in figure text'
    bb = fit_width(fig, width) if width else (fig.canvas.draw() or fig.get_tightbbox(fig.canvas.get_renderer()))
    check_layout(fig, name)
    fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.01)
    # vector copy for the paper (fonts embedded as TrueType; no timestamps, so the file is reproducible byte for byte)
    pdf = os.path.splitext(path)[0] + '.pdf'
    fig.savefig(pdf, bbox_inches='tight', pad_inches=0.01, metadata={'CreationDate': None, 'ModDate': None, 'Producer': None, 'Creator': None})
    plt.close(fig)
    print('wrote', os.path.relpath(path, ROOT), 'and', os.path.relpath(pdf, ROOT), f'({bb.width:.2f} x {bb.height:.2f} in)')
