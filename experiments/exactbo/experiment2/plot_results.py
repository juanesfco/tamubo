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
FIGSIZE = (SUBFIGURE_WIDTH_PT * PT_TO_INCH, SUBFIGURE_HEIGHT_PT * PT_TO_INCH)

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
        default=Path("experiments/exactbo/experiment2/figures"),
        help="Directory where the PDF figures will be written.",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        nargs="*",
        default=None,
        help="Dimensions to plot. Defaults to all dimensions present in the CSV.",
    )
    return parser.parse_args()


def configure_matplotlib() -> None:
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


def get_framework_metadata(frame: pd.DataFrame) -> FrameworkMetadata:
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
            framework_colors[method] = colors[FIXED_METHOD_STYLES[method]["color_index"] % len(colors)]
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
        frame.groupby(["Framework", "i"], as_index=False)[column]
        .agg(mean="mean", std="std")
        .fillna({"std": 0.0})
    )


def summarize_best_objective(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values(["Framework", "random_seed", "i"]).copy()
    frame["best_y_so_far"] = frame.groupby(["Framework", "random_seed"])["y"].cummin()
    return (
        frame.groupby(["Framework", "i"], as_index=False)["best_y_so_far"]
        .agg(mean="mean", std="std")
        .fillna({"std": 0.0})
    )


def choose_scale(values: Sequence[float]) -> tuple[float, str]:
    max_abs = max(abs(float(value)) for value in values if pd.notna(value))
    if max_abs >= 1e6:
        return 1e6, r" ($\times 10^{6}$)"
    if max_abs >= 1e3:
        return 1e3, r" ($\times 10^{3}$)"
    return 1.0, ""


def style_axes(ax: plt.Axes, ylabel: str) -> None:
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(ylabel, labelpad=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="0.88", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
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
) -> None:
    lower_values = (summary_df["mean"] - FILL_STD_SCALE * summary_df["std"]).tolist()
    upper_values = (summary_df["mean"] + FILL_STD_SCALE * summary_df["std"]).tolist()
    if y_ref is not None:
        lower_values.append(y_ref)
        upper_values.append(y_ref)

    scale, scale_suffix = choose_scale(lower_values + upper_values)
    fig, ax = plt.subplots(figsize=FIGSIZE)

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
            markeredgewidth=0.9,
            linewidth=1.2,
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
            label=y_ref_label or r"$y^\star$",
        )

    style_axes(ax, ylabel + scale_suffix)
    set_vertical_limits(
        ax,
        [value / scale for value in lower_values],
        [value / scale for value in upper_values],
        lower_pad=lower_pad,
        upper_pad=upper_pad,
    )
    ax.legend(
        loc="best",
        frameon=False,
        handlelength=1.5,
        handletextpad=0.5,
        borderpad=0.2,
        labelspacing=0.35,
    )
    fig.tight_layout(pad=0.15)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def generate_dimension_plots(frame: pd.DataFrame, dimension: int, output_dir: Path) -> None:
    frame = add_framework_column(frame[frame["d"] == dimension])
    if frame.empty:
        raise ValueError(f"No rows found for d={dimension}.")

    metadata = get_framework_metadata(frame)

    regret_summary = summarize_metric(frame, "R")
    plot_summary(
        regret_summary,
        metadata,
        r"$R_n$",
        output_dir / f"regret_d{dimension}.pdf",
    )

    time_summary = summarize_metric(frame, "t (s)")
    plot_summary(
        time_summary,
        metadata,
        r"$t_n$ (s)",
        output_dir / f"time_d{dimension}.pdf",
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
    )


def main() -> None:
    args = parse_args()
    configure_matplotlib()

    frame = read_results(args.input)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dimensions is None:
        dimensions = sorted(int(value) for value in frame["d"].dropna().unique())
    else:
        dimensions = args.dimensions

    for dimension in dimensions:
        generate_dimension_plots(frame, dimension, output_dir)
        print(f"Wrote plots for d={dimension} to {output_dir}")


if __name__ == "__main__":
    main()
