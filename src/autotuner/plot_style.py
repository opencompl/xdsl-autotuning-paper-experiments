"""Shared matplotlib style for the paper figures.

All figures are sized for a single column of the two-column LaTeX template
(~3.335 in wide) and are rendered without a title: the caption lives in the
LaTeX ``\\caption``, not in the image.
"""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, LinearSegmentedColormap, to_rgb
from matplotlib.figure import Figure

# Width of one column in the paper template, in inches.
COLUMN_WIDTH = 3.335
# Width of a figure* spanning both columns, in inches.
PAGE_WIDTH = 7.0

# Ink and grid colors, kept recessive so the data stays in front.
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#dcdbd6"

# A dash pattern, either a named style or an (offset, on/off sequence) pair.
LineStyle = str | tuple[int, tuple[int, ...]]

# The two variants the paper puts head to head, ours and the baseline.
OURS = "xdsl_libxsmm"
BASELINE = "libxsmm"

_STYLES: dict[str, tuple[str, str, str, LineStyle]] = {
    # variant: (label, color, marker, linestyle)
    OURS: ("x86 dialect (ours)", "#b2df8a", "none", "-"),
    BASELINE: ("LIBXSMM", "#a6cee3", "none", "-"),
    "compxsmm": ("CompXSMM", "#4a3aa7", "D", "-."),
    "compxsmm_manual": ("CompXSMM (no regalloc)", "#cab2d6", "d", (0, (2, 1))),
    "aocl": ("AOCL", "#ff7f00", "s", (0, (5, 2))),
    "mkl": ("MKL", "#008300", "^", ":"),
    "naive_c": ("naive C", "#e34948", "v", (0, (3, 1, 1, 1))),
    "llvm_intrinsics": ("LLVM intrinsics", "#1baf7a", "P", (0, (5, 1))),
    "tvm": ("TVM", "#eda100", "X", (0, (1, 1))),
    "transform_xdsl": ("xDSL transform", "#e87ba4", "*", (0, (4, 1, 1, 1))),
    "transform_mlir": ("MLIR transform", "#52514e", "<", (0, (2, 1))),
    "naive_mlir": ("naive MLIR", "#0d366b", ">", (0, (6, 2))),
    "vector_intrinsic": ("vector intrinsics", "#184f95", "h", (0, (3, 2))),
}

# One-hue sequential ramp (blue, light to dark) for the heatmaps.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue_sequential",
    [
        "#e8f1fd",
        "#cde2fb",
        "#9ec5f4",
        "#6da7ec",
        "#3987e5",
        "#256abf",
        "#184f95",
        "#0d366b",
    ],
)


# --- CIELAB / CIELCh, so ramps can be built in a perceptual space ----------

# sRGB primaries -> CIE XYZ (D65) and back.
_RGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)
_XYZ_TO_RGB = np.linalg.inv(_RGB_TO_XYZ)
# CIE standard illuminant D65, normalised to Y = 1.
_WHITE = np.array([0.95047, 1.0, 1.08883])
# The CIELAB kink, at (6/29)**3, below which the transfer is linear.
_DELTA = 6.0 / 29.0


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB in [0, 1] to CIELAB (L*, a*, b*)."""
    rgb = np.asarray(rgb, dtype=float)
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = linear @ _RGB_TO_XYZ.T / _WHITE
    f = np.where(
        xyz > _DELTA**3,
        np.cbrt(np.clip(xyz, 0.0, None)),
        xyz / (3 * _DELTA**2) + 4.0 / 29.0,
    )
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    return np.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], axis=-1)


def _lab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """Convert CIELAB to sRGB in [0, 1], clipping colors outside the gamut."""
    lab = np.asarray(lab, dtype=float)
    fy = (lab[..., 0] + 16) / 116
    fx = fy + lab[..., 1] / 500
    fz = fy - lab[..., 2] / 200
    f = np.stack([fx, fy, fz], axis=-1)
    xyz = np.where(f > _DELTA, f**3, 3 * _DELTA**2 * (f - 4.0 / 29.0)) * _WHITE
    linear = xyz @ _XYZ_TO_RGB.T
    srgb = np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.clip(linear, 0.0, None) ** (1 / 2.4) - 0.055,
    )
    return np.clip(srgb, 0.0, 1.0)


def diverging_colormap(
    low: str,
    high: str,
    *,
    name: str = "diverging",
    center_lightness: float = 97.5,
    center_chroma: float = 1.5,
    samples: int = 256,
) -> Colormap:
    """A two-armed diverging ramp from ``low`` through near-white to ``high``."""
    center = np.array([center_lightness, 0.0, 0.0])
    arms = []
    for end_hex in (low, high):
        end = _srgb_to_lab(np.array(to_rgb(end_hex)))
        chroma = float(np.hypot(end[1], end[2]))
        hue = float(np.arctan2(end[2], end[1]))
        # Ramp lightness and chroma together, hue fixed:
        t = np.linspace(0.0, 1.0, samples // 2)
        lightness = center[0] + t * (end[0] - center[0])
        chromas = center_chroma + t * (chroma - center_chroma)
        arm = np.stack(
            [lightness, chromas * np.cos(hue), chromas * np.sin(hue)], axis=-1
        )
        arms.append(_lab_to_srgb(arm))

    low_arm, high_arm = arms
    return LinearSegmentedColormap.from_list(
        name, np.concatenate([low_arm[::-1], high_arm[1:]])
    )


DIVERGING = diverging_colormap("#1f78b4", "#33a02c", name="libxsmm_vs_ours")


def contrasting_ink(color: Any) -> str:
    """Black or white text, whichever stays legible on ``color``."""
    lightness = float(_srgb_to_lab(np.array(to_rgb(color)))[0])
    return "white" if lightness < 60.0 else INK


def variant_label(variant: str) -> str:
    """Human-readable name of a kernel variant."""
    style = _STYLES.get(variant)
    return style[0] if style else variant


def variant_style(variant: str) -> dict[str, Any]:
    """Plot kwargs (color, marker, linestyle, label) for a kernel variant."""
    style = _STYLES.get(variant)
    if style is None:
        return {"label": variant, "color": INK_MUTED, "marker": "o", "linestyle": "-"}
    label, color, marker, linestyle = style
    return {"label": label, "color": color, "marker": marker, "linestyle": linestyle}


def sorted_variants(variants: Iterable[str]) -> list[str]:
    """Order variants by their palette slot, unknown ones last."""
    order = list(_STYLES)

    def key(variant: str) -> tuple[int, str]:
        return (order.index(variant) if variant in order else len(order), variant)

    return sorted(set(variants), key=key)


def use_paper_style() -> None:
    """Apply the rcParams shared by every figure."""
    mpl.rcParams.update(
        {
            # Helvetica, with metric-compatible fallbacks.  TrueType faces come
            # before the URW OpenType clones: they embed cleanly as the Type 42
            # fonts that paper templates expect.
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Helvetica",
                "Arimo",
                "Liberation Sans",
                "Nimbus Sans",
                "DejaVu Sans",
            ],
            "mathtext.fontset": "stixsans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.title_fontsize": 7,
            "figure.titlesize": 8,
            # Thin marks, recessive axes.
            "lines.linewidth": 1.0,
            "lines.markersize": 3.0,
            "lines.markeredgewidth": 0.0,
            "axes.linewidth": 0.5,
            "axes.edgecolor": INK_MUTED,
            "axes.labelcolor": INK,
            "axes.labelpad": 2.0,
            "text.color": INK,
            "grid.color": GRID,
            "grid.linewidth": 0.4,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK,
            "ytick.labelcolor": INK,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.0,
            "ytick.major.size": 2.0,
            "xtick.major.pad": 1.5,
            "ytick.major.pad": 1.5,
            # Compact legend that fits inside a column.
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.edgecolor": GRID,
            "legend.borderpad": 0.3,
            "legend.labelspacing": 0.25,
            "legend.handlelength": 1.8,
            "legend.handletextpad": 0.5,
            "legend.borderaxespad": 0.3,
            "legend.columnspacing": 1.0,
            # Output: vector text in PDFs, tight crop, no wasted margin.
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.01,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def column_figure(
    *,
    aspect: float = 0.62,
    width: float = COLUMN_WIDTH,
    nrows: int = 1,
    ncols: int = 1,
    **kwargs: Any,
) -> tuple[Figure, Any]:
    """Create a figure that is exactly one column wide.

    ``aspect`` is the height of one subplot as a fraction of its own width, so a
    grid keeps the panel shape of a single-panel figure.
    """
    use_paper_style()
    return plt.subplots(
        nrows,
        ncols,
        figsize=(width, width * aspect * nrows / ncols),
        **kwargs,
    )


def sized_figure(width: float, height: float) -> Figure:
    """An empty figure of an exact size, for axes placed by hand."""
    use_paper_style()
    return plt.figure(figsize=(width, height))


def tidy_axes(ax: Axes) -> None:
    """Drop the top/right spines and put a recessive grid behind the data."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linewidth=0.4, color=GRID)
    ax.set_axisbelow(True)


def integer_ticks(values: Iterable[float], max_ticks: int = 8) -> list[int]:
    """Tick positions taken from ``values``, thinned to at most ``max_ticks``."""
    unique = sorted({int(v) for v in values})
    step = max(1, -(-len(unique) // max_ticks))
    return unique[::step]


def save(fig: Figure, output_path: Path | None, *, tight: bool = True) -> None:
    """Write the figure to ``output_path``, or show it when that is None.

    ``tight`` crops the output to the ink, which is what a figure laid out by
    ``tight_layout`` wants.  A figure whose margins were sized by hand must
    pass ``tight=False``, or the crop takes its exact width away again.
    """
    if output_path is None:
        plt.show()
        return

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # ``bbox_inches=None`` falls back to the rcParam, which this style sets to
    # "tight", so an untight save has to clear the param itself.
    with mpl.rc_context({} if tight else {"savefig.bbox": None}):
        fig.savefig(path)
    plt.close(fig)


def legend_inside(fig: Figure, ax: Axes, *, loc: str = "lower right") -> None:
    """Put the legend in a corner of the axes, where the data leaves room."""
    ax.legend(loc=loc)
    fig.tight_layout(pad=0.1)


def legend_below(fig: Figure, ax: Axes, *, ncol: int = 3) -> None:
    """Put the legend under the axes so it never covers the data."""
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=ncol,
    )
    fig.tight_layout(pad=0.1, rect=(0, 0.06, 1, 1))
