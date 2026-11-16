"""Common figure style (IEEE-like serif fonts) and export in three formats.

Every figure is written as
* SVG -- master copy; text kept as editable text (``svg.fonttype = 'none'``);
* PDF -- vector copy for ``\\includegraphics`` with pdfLaTeX (TrueType fonts embedded);
* PNG -- 600 dpi raster copy (IEEE requires at least 300 dpi).
Photographic panels (generated / real images) are necessarily embedded as
raster images in the SVG and PDF at their native resolution.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STYLE = {
    "svg.fonttype": "none",
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "pdf.fonttype": 42,
}
FORMATS = ("svg", "pdf", "png")
PNG_DPI = 600


def apply_style():
    plt.rcParams.update(STYLE)


def save_figure(fig, out):
    """Save ``fig`` as <stem>.svg, <stem>.pdf and <stem>.png (any extension of ``out`` is ignored)."""
    stem = Path(out).with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in FORMATS:
        path = stem.with_suffix(f".{fmt}")
        fig.savefig(path, format=fmt, bbox_inches="tight", dpi=PNG_DPI if fmt == "png" else 300)
        paths.append(path)
    print("saved " + ", ".join(str(p) for p in paths))
    plt.close(fig)
    return paths
