#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

PT_TO_INCH = 1 / 72.27
TEXT_WIDTH_PT = 542.02501
SUBFIGURE_WIDTH_PT = 0.99 * TEXT_WIDTH_PT / 3
SUBFIGURE_HEIGHT_PT = 1.0 * SUBFIGURE_WIDTH_PT
PAPER_FIGSIZE = (SUBFIGURE_WIDTH_PT * PT_TO_INCH, SUBFIGURE_HEIGHT_PT * PT_TO_INCH)
PRESENTATION_FIGSIZE = (2.4, 2.4)

PAPER_OUTPUT_DIR = Path("experiments/exactbo/experiment2/figures")
PRESENTATION_OUTPUT_DIR = PAPER_OUTPUT_DIR / "presentation"

EXACTBO_COLOR_INDICES = [0, 1, 4, 5, 6]
EXACTBO_MARKERS = ["o", "s", "v", "P", "X"]
EXACTBO_LINESTYLE = "-"
FIXED_METHOD_ORDER = {"gridBO": 1, "gradBO": 2}
FIXED_METHOD_STYLES = {
    "gridBO": {"color_index": 2, "marker": "^", "linestyle": "--"},
    "gradBO": {"color_index": 3, "marker": "D", "linestyle": ":"},
}
FILL_STD_SCALE = 0.1


@dataclass(frozen=True)
class FrameworkMetadata:
    order: list[str]
    labels: dict[str, str]
    colors: dict[str, str]
    markers: dict[str, str]
    linestyles: dict[str, str]


@dataclass(frozen=True)
class PlotStyle:
    mode: str
    figsize: tuple[float, float]
    transparent: bool
    tight_bbox: bool
    line_width: float
    marker_edge_width: float
    legend_fontsize: float
    legend_frame: bool
    max_x_bins: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate publication-ready BO comparison plots from experiment_results.csv.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("experiments/exactbo/experiment2/results/experiment_results.csv"),
        help="Path to the experiment results CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory where the PDF figures will be written. Defaults to "
            "figures/ for paper mode and figures/presentation/ for presentation mode."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("paper", "presentation"),
        default="paper",
        help="Plot styling mode. Paper mode preserves the original publication styling.",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        nargs="*",
        default=None,
        help="Dimensions to plot. Defaults to all dimensions present in the CSV.",
    )
    return parser.parse_args()


def get_plot_style(mode: str) -> PlotStyle:
    if mode == "presentation":
        return PlotStyle(
            mode=mode,
            figsize=PRESENTATION_FIGSIZE,
            transparent=True,
            tight_bbox=False,
            line_width=1.4,
            marker_edge_width=0.75,
            legend_fontsize=7.0,
            legend_frame=False,
            max_x_bins=4,
        )
    return PlotStyle(
        mode=mode,
        figsize=PAPER_FIGSIZE,
        transparent=False,
        tight_bbox=True,
        line_width=1.2,
        marker_edge_width=0.9,
        legend_fontsize=8.0,
        legend_frame=False,
        max_x_bins=5,
    )


def configure_matplotlib(mode: str) -> None:
    mpl.rcdefaults()
    if mode == "paper":
        mpl.rcParams.update(
            {
                # Match the asmeconf class more closely by letting LaTeX typeset figure text.
                "text.usetex": True,
                "text.latex.preamble": r"\usepackage[helvratio=.91]{newtxtext}\usepackage{newtxmath}",
                "font.family": "serif",
                "font.size": 8,
                "axes.titlesize": 8,
                "axes.labelsize": 8,
                "axes.labelweight": "regular",
                "xtick.labelsize": 7,
                "ytick.labelsize": 7,
                "legend.fontsize": 8,
                "axes.linewidth": 0.8,
                "lines.linewidth": 1.2,
                "lines.markersize": 4.0,
                "xtick.major.width": 0.8,
                "ytick.major.width": 0.8,
                "xtick.major.size": 3.0,
                "ytick.major.size": 3.0,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
        )
        return

    mpl.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Computer Modern Sans Serif",
                "CMU Sans Serif",
                "CMU Sans",
                "DejaVu Sans",
                "cmss10",
                "cmss",
            ],
            "font.size": 9,
            "mathtext.fontset": "cm",
            "mathtext.rm": "cm",
            "mathtext.sf": "cmss",
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "axes.labelweight": "regular",
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.4,
            "lines.markersize": 3.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_results(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    numeric_columns = ["d", "epsilonEI", "i", "R", "y", "y*", "t (s)"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def format_epsilon(value: float) -> str:
    return f"{value:g}"


def add_framework_column(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    is_exact = frame["Method"] == "exactBO"
    frame.loc[is_exact, "Framework"] = frame.loc[is_exact, "epsilonEI"].map(
        lambda epsilon: f"exactBO (epsilonEI={format_epsilon(epsilon)})"
    )
    frame.loc[~is_exact, "Framework"] = frame.loc[~is_exact, "Method"]
    return frame


def get_framework_metadata(frame: pd.DataFrame, mode: str) -> FrameworkMetadata:
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    available_methods = frame["Method"].dropna().unique().tolist()
    exact_epsilons = sorted(
        frame.loc[frame["Method"] == "exactBO", "epsilonEI"].dropna().unique(),
        reverse=True,
    )

    order = [f"exactBO (epsilonEI={format_epsilon(epsilon)})" for epsilon in exact_epsilons]

    remaining_methods = sorted(
        [method for method in available_methods if method != "exactBO"],
        key=lambda method: (FIXED_METHOD_ORDER.get(method, 99), method),
    )
    order.extend(remaining_methods)

    labels: dict[str, str] = {}
    markers: dict[str, str] = {}
    framework_colors: dict[str, str] = {}
    linestyles: dict[str, str] = {}

    for index, epsilon in enumerate(exact_epsilons):
        key = f"exactBO (epsilonEI={format_epsilon(epsilon)})"
        labels[key] = rf"exactBO ($\varepsilon_{{\mathrm{{EI}}}}={format_epsilon(epsilon)}$)"
        color_index = EXACTBO_COLOR_INDICES[index] if index < len(EXACTBO_COLOR_INDICES) else index
        framework_colors[key] = colors[color_index % len(colors)]
        markers[key] = EXACTBO_MARKERS[index % len(EXACTBO_MARKERS)]
        linestyles[key] = EXACTBO_LINESTYLE

    fallback_color_index = max(EXACTBO_COLOR_INDICES, default=-1) + 1
    fallback_marker_cycle = ["<", ">", "*", "h"]
    fallback_index = 0

    for method in remaining_methods:
        labels[method] = method
        if method in FIXED_METHOD_STYLES:
            framework_colors[method] = colors[
                FIXED_METHOD_STYLES[method]["color_index"] % len(colors)
            ]
            markers[method] = FIXED_METHOD_STYLES[method]["marker"]
            linestyles[method] = FIXED_METHOD_STYLES[method]["linestyle"]
        else:
            framework_colors[method] = colors[(fallback_color_index + fallback_index) % len(colors)]
            markers[method] = fallback_marker_cycle[fallback_index % len(fallback_marker_cycle)]
            linestyles[method] = "-."
            fallback_index += 1

    return FrameworkMetadata(
        order=order,
        labels=labels,
        colors=framework_colors,
        markers=markers,
        linestyles=linestyles,
    )


def summarize_metric(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    return (
        frame.groupby(["Framework", "i"])[column]
        .agg(["mean", "std"])
        .reset_index()
        .fillna({"std": 0.0})
    )


def summarize_best_objective(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values(["Framework", "random_seed", "i"]).copy()
    frame["best_y_so_far"] = frame.groupby(["Framework", "random_seed"])["y"].cummin()
    return (
        frame.groupby(["Framework", "i"])["best_y_so_far"]
        .agg(["mean", "std"])
        .reset_index()
        .fillna({"std": 0.0})
    )


def choose_scale(values: Sequence[float]) -> tuple[float, str]:
    max_abs = max(abs(float(value)) for value in values if pd.notna(value))
    if max_abs >= 1e6:
        return 1e6, r" ($\times 10^{6}$)"
    if max_abs >= 1e3:
        return 1e3, r" ($\times 10^{3}$)"
    return 1.0, ""


def style_axes(ax: plt.Axes, ylabel: str, style: PlotStyle) -> None:
    if style.mode == "presentation":
        ax.set_xlabel("iteration", labelpad=0.0)
    else:
        ax.set_xlabel(r"$n$")
    ax.set_ylabel(ylabel, labelpad=0.0 if style.mode == "presentation" else 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="0.88", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=style.max_x_bins))
    ax.margins(x=0.02)


def marker_positions(num_points: int, series_index: int) -> list[int] | None:
    if num_points <= 4:
        return None

    stride = 3
    start = series_index % stride
    positions = list(range(start, num_points, stride))
    if (num_points - 1) not in positions:
        positions.append(num_points - 1)
    return sorted(set(positions))


def set_vertical_limits(
    ax: plt.Axes,
    lower_values: Sequence[float],
    upper_values: Sequence[float],
    lower_pad: float,
    upper_pad: float,
) -> None:
    ymin = min(lower_values)
    ymax = max(upper_values)
    span = ymax - ymin
    if span <= 0:
        span = max(abs(ymax), 1.0)
    ax.set_ylim(ymin - lower_pad * span, ymax + upper_pad * span)


def plot_summary(
    summary_df: pd.DataFrame,
    metadata: FrameworkMetadata,
    ylabel: str,
    output_path: Path,
    *,
    y_ref: float | None = None,
    y_ref_label: str | None = None,
    lower_pad: float = 0.06,
    upper_pad: float = 0.04,
    style: PlotStyle,
) -> None:
    lower_values = (summary_df["mean"] - FILL_STD_SCALE * summary_df["std"]).tolist()
    upper_values = (summary_df["mean"] + FILL_STD_SCALE * summary_df["std"]).tolist()
    if y_ref is not None:
        lower_values.append(y_ref)
        upper_values.append(y_ref)

    scale, scale_suffix = choose_scale(lower_values + upper_values)
    fig, ax = plt.subplots(figsize=style.figsize)

    for framework in metadata.order:
        framework_summary = summary_df[summary_df["Framework"] == framework].sort_values("i")
        if framework_summary.empty:
            continue

        color = metadata.colors[framework]
        lower_band = (framework_summary["mean"] - FILL_STD_SCALE * framework_summary["std"]) / scale
        upper_band = (framework_summary["mean"] + FILL_STD_SCALE * framework_summary["std"]) / scale
        series_index = metadata.order.index(framework)
        line, = ax.plot(
            framework_summary["i"],
            framework_summary["mean"] / scale,
            color=color,
            linestyle=metadata.linestyles[framework],
            marker=metadata.markers[framework],
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=style.marker_edge_width,
            linewidth=style.line_width,
            markevery=marker_positions(len(framework_summary), series_index),
            label=metadata.labels[framework],
            zorder=3 + series_index,
        )
        ax.fill_between(
            framework_summary["i"],
            lower_band,
            upper_band,
            color=color,
            alpha=0.16,
            linewidth=0.0,
            zorder=1,
        )

    if y_ref is not None:
        ax.axhline(
            y_ref / scale,
            color="black",
            linestyle=(0, (1.2, 1.2)),
            linewidth=0.9,
            #label=y_ref_label or r"$y^\star$",
        )

    style_axes(ax, ylabel + scale_suffix, style)
    set_vertical_limits(
        ax,
        [value / scale for value in lower_values],
        [value / scale for value in upper_values],
        lower_pad=lower_pad,
        upper_pad=upper_pad,
    )
    legend_kwargs = {
        "frameon": style.legend_frame,
        "fontsize": style.legend_fontsize,
        "handlelength": 1.5,
        "handletextpad": 0.5,
        "borderpad": 0.2,
        "labelspacing": 0.25,
    }
    if y_ref is not None:
        legend = ax.legend(loc="upper right", **legend_kwargs)
    else:
        legend = ax.legend(loc="best", **legend_kwargs)
    if style.legend_frame:
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("none")
        legend.get_frame().set_alpha(0.72)

    if style.mode == "presentation":
        # Keep every y-label safely inside the fixed presentation canvas. This
        # also leaves room for wider negative/decimal tick labels when present.
        layout_pad = 0.55
    else:
        layout_pad = 0.15
    fig.tight_layout(pad=layout_pad)
    save_kwargs: dict[str, object] = {"transparent": style.transparent}
    if style.tight_bbox:
        save_kwargs.update({"bbox_inches": "tight", "pad_inches": 0.02})
    fig.savefig(output_path, **save_kwargs)
    plt.close(fig)


def generate_dimension_plots(
    frame: pd.DataFrame,
    dimension: int,
    output_dir: Path,
    style: PlotStyle,
) -> None:
    frame = add_framework_column(frame[frame["d"] == dimension])
    if frame.empty:
        raise ValueError(f"No rows found for d={dimension}.")

    metadata = get_framework_metadata(frame, style.mode)

    regret_summary = summarize_metric(frame, "R")
    plot_summary(
        regret_summary,
        metadata,
        r"$R_n$",
        output_dir / f"regret_d{dimension}.pdf",
        style=style,
    )

    time_summary = summarize_metric(frame, "t (s)")
    plot_summary(
        time_summary,
        metadata,
        r"$t_n$ (s)",
        output_dir / f"time_d{dimension}.pdf",
        style=style,
    )

    best_y_summary = summarize_best_objective(frame)
    y_star = frame["y*"].dropna().iloc[0]
    plot_summary(
        best_y_summary,
        metadata,
        r"$y^\star_n$",
        output_dir / f"best_objective_d{dimension}.pdf",
        y_ref=y_star,
        y_ref_label=r"$y^\star$",
        lower_pad=0.14,
        upper_pad=0.04,
        style=style,
    )


def main() -> None:
    args = parse_args()
    configure_matplotlib(args.mode)
    style = get_plot_style(args.mode)

    frame = read_results(args.input)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = PRESENTATION_OUTPUT_DIR if args.mode == "presentation" else PAPER_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dimensions is None:
        dimensions = sorted(int(value) for value in frame["d"].dropna().unique())
    else:
        dimensions = args.dimensions

    for dimension in dimensions:
        generate_dimension_plots(frame, dimension, output_dir, style)
        print(f"Wrote plots for d={dimension} to {output_dir}")


if __name__ == "__main__":
    main()
