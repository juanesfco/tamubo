"""Create a compact ExactBO-style partition figure for a poster."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
import numpy as np


CM_TO_INCH = 1 / 2.54
FIGURE_WIDTH_CM = 7.0
FIGURE_HEIGHT_CM = 6.0

RED = "#d62728"
GREEN = "#008837"

CMAP = LinearSegmentedColormap.from_list(
    "pale_red_white_green",
    [(0.0, "#f96e6e"), (0.5, "white"), (1.0, "#6adc86")],
)
NORM = Normalize(vmin=0.0, vmax=1.0)


def split_direct(rectangle):
    """Split a rectangle into five children using the DIRECT layout."""
    x, y, width, height = rectangle

    if width >= height:
        third_width = width / 3
        third_height = height / 3
        return [
            (x, y, third_width, height),
            (x + 2 * third_width, y, third_width, height),
            (x + third_width, y, third_width, third_height),
            (
                x + third_width,
                y + third_height,
                third_width,
                third_height,
            ),
            (
                x + third_width,
                y + 2 * third_height,
                third_width,
                third_height,
            ),
        ]

    third_width = width / 3
    third_height = height / 3
    return [
        (x, y, width, third_height),
        (x, y + 2 * third_height, width, third_height),
        (x, y + third_height, third_width, third_height),
        (
            x + third_width,
            y + third_height,
            third_width,
            third_height,
        ),
        (
            x + 2 * third_width,
            y + third_height,
            third_width,
            third_height,
        ),
    ]


def distance_to_rectangle(point, rectangle):
    """Return the shortest Euclidean distance from a point to a rectangle."""
    point_x, point_y = point
    x, y, width, height = rectangle
    dx = max(x - point_x, 0.0, point_x - (x + width))
    dy = max(y - point_y, 0.0, point_y - (y + height))
    return np.hypot(dx, dy)


def build_direct_partition(
    rectangle,
    depth,
    max_depth,
    focus_point,
    refinement_width,
):
    """Build a locally refined partition from repeated DIRECT-style splits."""
    distance = distance_to_rectangle(focus_point, rectangle)
    local_scale = max(rectangle[2], rectangle[3])
    should_split = (
        depth < max_depth
        and distance <= refinement_width * local_scale
    )

    if not should_split:
        return [rectangle]

    leaves = []
    for child in split_direct(rectangle):
        leaves.extend(
            build_direct_partition(
                child,
                depth + 1,
                max_depth,
                focus_point,
                refinement_width,
            )
        )
    return leaves


def draw_partition(ax, rectangles, seed=7):
    """Draw tiles with colors that trend greener as their area decreases."""
    rng = np.random.default_rng(seed)
    areas = np.array([width * height for _, _, width, height in rectangles])
    log_areas = np.log(areas)
    log_span = log_areas.max() - log_areas.min()

    if np.isclose(log_span, 0):
        smallness = np.full_like(log_areas, 0.5)
    else:
        smallness = (log_areas.max() - log_areas) / log_span

    values = np.clip(
        0.08 + 0.84 * smallness + rng.normal(0, 0.055, len(areas)),
        0,
        1,
    )

    for (x, y, width, height), value in zip(rectangles, values):
        edge_color = RED if value < 0.5 else GREEN
        ax.add_patch(
            Rectangle(
                (x, y),
                width,
                height,
                facecolor=CMAP(NORM(value)),
                edgecolor=edge_color,
                linewidth=0.8,
            )
        )


def create_figure():
    """Return the poster figure and its partition axis."""
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 500,
            "font.size": 10,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Computer Modern Sans Serif",
                "CMU Sans Serif",
                "CMU Sans",
                "cmss10",
                "cmss",
            ],
            "mathtext.fontset": "cm",
            "mathtext.rm": "cm",
            "mathtext.sf": "cmss",
            "axes.labelsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    fig = plt.figure(
        figsize=(FIGURE_WIDTH_CM * CM_TO_INCH, FIGURE_HEIGHT_CM * CM_TO_INCH)
    )
    # Manual placement maximizes the plotting area while preserving the exact
    # physical figure size and leaving just enough room for labels.
    ax = fig.add_axes([0.06, 0.08, 0.78, 0.84])
    colorbar_axis = fig.add_axes([0.87, 0.08, 0.035, 0.84])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    rectangles = build_direct_partition(
        rectangle=(0.0, 0.0, 1.0, 1.0),
        depth=0,
        max_depth=6,
        focus_point=(0.7, 0.3),
        refinement_width=0.14,
    )
    draw_partition(ax, rectangles)

    colorbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=NORM, cmap=CMAP),
        cax=colorbar_axis,
        orientation="vertical",
    )
    colorbar.set_ticks([])
    colorbar.ax.tick_params(length=0)
    colorbar.set_label(r"$\overline{\mathrm{EI}}$", rotation=0, labelpad=7)
    colorbar.ax.yaxis.set_label_position("right")

    legend_handles = [
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor="none",
            edgecolor=GREEN,
            linewidth=0.8,
            label="Active",
        ),
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor="none",
            edgecolor=RED,
            linewidth=0.8,
            label="Inactive",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.45, 0.995),
        ncol=2,
        frameon=False,
        fontsize=10,
        handlelength=0.7,
        handleheight=0.4,
        handletextpad=0.35,
        columnspacing=0.8,
        borderpad=0.0,
    )

    return fig, ax


def main():
    """Create and save the figure next to the other development figures."""
    output_directory = Path(__file__).resolve().parent / "figures"
    output_directory.mkdir(parents=True, exist_ok=True)

    fig, _ = create_figure()
    fig.savefig(output_directory / "poster_figure.png", dpi=500)
    fig.savefig(output_directory / "poster_figure.pdf")
    plt.show()


if __name__ == "__main__":
    main()
