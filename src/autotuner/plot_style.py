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
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
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

# Categorical hues in fixed slot order; a variant always keeps its color and
# marker across figures.  Line style and marker are a second encoding, so the
# figures survive grayscale printing.
_STYLES: dict[str, tuple[str, str, str, LineStyle]] = {
    # variant: (label, color, marker, linestyle)
    "xdsl_libxsmm": ("xDSL (ours)", "#2a78d6", "o", "-"),
    "libxsmm": ("LIBXSMM", "#eb6834", "s", "--"),
    "compxsmm": ("CompXSMM", "#4a3aa7", "D", "-."),
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


def save(fig: Figure, output_path: Path | None) -> None:
    """Write the figure to ``output_path``, or show it when that is None."""
    if output_path is None:
        plt.show()
        return

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


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
