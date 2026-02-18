from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

_LAST_ANIMATION: FuncAnimation | None = None


def _display_animation_if_notebook(anim: FuncAnimation, fig) -> bool:
    """
    Return True when animation was displayed in notebook output.
    """
    try:
        from IPython import get_ipython
        from IPython.display import HTML, display
    except Exception:
        return False

    shell = get_ipython()
    if shell is None:
        return False
    shell_name = shell.__class__.__name__
    if shell_name not in {"ZMQInteractiveShell", "TerminalInteractiveShell"}:
        return False

    # In notebooks, jshtml is the most reliable way to render animation output.
    display(HTML(anim.to_jshtml()))
    plt.close(fig)
    return True


def _sorted_prefixed_keys(data: Mapping[str, Any], prefix: str) -> list[str]:
    keys = [k for k in data.keys() if k.startswith(prefix) and k[len(prefix) :].isdigit()]
    return sorted(keys, key=lambda k: int(k[len(prefix) :]))


def _load_log(log: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(log, Mapping):
        return dict(log)
    with Path(log).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("log must be a JSON object or dict")
    return data


def _as_bounds(bounds: Any) -> np.ndarray:
    arr = np.asarray(bounds, dtype=float)
    if arr.shape != (2, 2):
        raise ValueError(f"bounds must have shape (2,2), got {arr.shape}")
    return arr


def _as_points(points: Any, name: str) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N,2), got {arr.shape}")
    return arr


def _as_point(point: Any, name: str) -> np.ndarray:
    arr = np.asarray(point, dtype=float).ravel()
    if arr.shape != (2,):
        raise ValueError(f"{name} must have shape (2,), got {arr.shape}")
    return arr


def _resolve_iteration_key(log_data: Mapping[str, Any], iteration: int | str) -> str:
    i_keys = _sorted_prefixed_keys(log_data, "i")
    if not i_keys:
        raise ValueError("No iterations found in log.")
    if isinstance(iteration, str):
        key = iteration if iteration.startswith("i") else f"i{iteration}"
        if key not in log_data:
            raise KeyError(f"Iteration '{key}' not found in log.")
        return key

    direct_key = f"i{iteration}"
    if direct_key in log_data:
        return direct_key
    if 0 <= iteration < len(i_keys):
        return i_keys[iteration]
    raise KeyError(f"Iteration '{direct_key}' not found in log.")


def _resolve_partition_key(iter_data: Mapping[str, Any], partition: int | str) -> str:
    p_keys = _sorted_prefixed_keys(iter_data, "p")
    if not p_keys:
        raise ValueError("No partitions found in iteration log.")
    if isinstance(partition, str):
        key = partition if partition.startswith("p") else f"p{partition}"
        if key not in iter_data:
            raise KeyError(f"Partition '{key}' not found in iteration.")
        return key

    direct_key = f"p{partition}"
    if direct_key in iter_data:
        return direct_key
    if 0 <= partition < len(p_keys):
        return p_keys[partition]
    raise KeyError(f"Partition '{direct_key}' not found in iteration.")


def _compute_surface(
    f: Callable[[np.ndarray], np.ndarray],
    bounds: np.ndarray,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(bounds[0, 0], bounds[0, 1], resolution)
    y = np.linspace(bounds[1, 0], bounds[1, 1], resolution)
    xx, yy = np.meshgrid(x, y)
    xy = np.column_stack((xx.ravel(), yy.ravel()))

    zz = np.asarray(f(xy), dtype=float).reshape(-1)
    if zz.size != xy.shape[0]:
        zz = np.asarray([np.asarray(f(p), dtype=float).reshape(-1)[0] for p in xy], dtype=float)
    return xx, yy, zz.reshape(xx.shape)


def _draw_partition(
    ax,
    bounds_L: Any,
    bounds_U: Any,
    active_boxes_mask: Any,
    *,
    sampled: Any | None = None,
    next_point: Any | None = None,
    bounds: np.ndarray,
) -> None:
    box_l = _as_points(bounds_L, "bounds_L")
    box_u = _as_points(bounds_U, "bounds_U")
    active = np.asarray(active_boxes_mask, dtype=bool).ravel()

    if box_l.shape != box_u.shape:
        raise ValueError("bounds_L and bounds_U must have the same shape.")
    if active.size != box_l.shape[0]:
        raise ValueError("active_boxes_mask length must match number of boxes.")

    shown_active = False
    shown_inactive = False
    for lower, upper, is_active in zip(box_l, box_u, active):
        color = "tab:green" if is_active else "tab:red"
        label = None
        if is_active and not shown_active:
            label = "Active Box"
            shown_active = True
        elif (not is_active) and not shown_inactive:
            label = "Inactive Box"
            shown_inactive = True

        rect = Rectangle(
            (lower[0], lower[1]),
            upper[0] - lower[0],
            upper[1] - lower[1],
            facecolor=color,
            edgecolor=color,
            alpha=0.14,
            linewidth=1.0,
            label=label,
        )
        ax.add_patch(rect)

    if sampled is not None:
        X = _as_points(sampled, "X")
        ax.scatter(X[:, 0], X[:, 1], color="black", s=28, marker="o", label="Sampled", zorder=4)

    if next_point is not None:
        Xn = _as_point(next_point, "Xn")
        ax.scatter(
            Xn[0],
            Xn[1],
            color="#ffcc00",
            edgecolors="black",
            linewidth=0.8,
            s=190,
            marker="*",
            label="Next Point",
            zorder=5,
        )

    ax.set_xlim(bounds[0, 0], bounds[0, 1])
    ax.set_ylim(bounds[1, 0], bounds[1, 1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc="upper right")


def _draw_function(
    ax,
    f: Callable[[np.ndarray], np.ndarray],
    bounds: np.ndarray,
    *,
    X: Any | None = None,
    Xn: Any | None = None,
    resolution: int = 220,
    contour_levels: int = 60,
    cmap: str = "viridis",
    surface: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    norm: Normalize | None = None,
):
    if surface is None:
        surface = _compute_surface(f, bounds, resolution)
    xx, yy, zz = surface

    contour = ax.contourf(xx, yy, zz, levels=contour_levels, cmap=cmap, norm=norm)
    ax.contour(xx, yy, zz, levels=max(8, contour_levels // 4), colors="k", linewidths=0.35, alpha=0.5)

    if X is not None:
        X_arr = _as_points(X, "X")
        ax.scatter(
            X_arr[:, 0],
            X_arr[:, 1],
            color="white",
            edgecolors="black",
            s=28,
            marker="o",
            label="Sampled",
            zorder=4,
        )

    if Xn is not None:
        Xn_arr = _as_point(Xn, "Xn")
        ax.scatter(
            Xn_arr[0],
            Xn_arr[1],
            color="#ffcc00",
            edgecolors="black",
            linewidth=0.8,
            s=190,
            marker="*",
            label="Next Point",
            zorder=5,
        )

    ax.set_xlim(bounds[0, 0], bounds[0, 1])
    ax.set_ylim(bounds[1, 0], bounds[1, 1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc="upper right")
    return contour, surface


def plot_log(
    log: str | Path | Mapping[str, Any],
    iteration: int | str,
    partition: int | str,
    *,
    ax=None,
    bounds: Any | None = None,
    show: bool = True,
):
    """
    Static partition plot from log for one (iteration, partition).
    """
    log_data = _load_log(log)
    i_key = _resolve_iteration_key(log_data, iteration)
    iter_data = log_data[i_key]
    p_key = _resolve_partition_key(iter_data, partition)

    p_keys = _sorted_prefixed_keys(iter_data, "p")
    is_last_partition = p_key == p_keys[-1]

    X = _as_points(iter_data["X"], f"{i_key}.X")
    Xn = _as_point(iter_data["Xn"], f"{i_key}.Xn") if is_last_partition and "Xn" in iter_data else None

    if bounds is None:
        box_l = _as_points(iter_data[p_key]["bounds_L"], f"{i_key}.{p_key}.bounds_L")
        box_u = _as_points(iter_data[p_key]["bounds_U"], f"{i_key}.{p_key}.bounds_U")
        lower = np.min(box_l, axis=0)
        upper = np.max(box_u, axis=0)
        bounds_arr = np.column_stack([lower, upper])
    else:
        bounds_arr = _as_bounds(bounds)

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)

    part_data = iter_data[p_key]
    _draw_partition(
        ax,
        part_data["bounds_L"],
        part_data["bounds_U"],
        part_data["active_boxes_mask"],
        sampled=X,
        next_point=Xn,
        bounds=bounds_arr,
    )
    ax.set_title(f"Iteration {int(i_key[1:]) + 1} - {p_key}")

    if show and created_fig:
        plt.show()
    return ax


def plot_f(
    f: Callable[[np.ndarray], np.ndarray],
    bounds: Any,
    X: Any | None = None,
    Xn: Any | None = None,
    *,
    ax=None,
    resolution: int = 220,
    contour_levels: int = 60,
    cmap: str = "viridis",
    surface: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    norm: Normalize | None = None,
    show: bool = True,
):
    """
    Static function colormap plot.
    """
    bounds_arr = _as_bounds(bounds)

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)
    contour, _ = _draw_function(
        ax,
        f,
        bounds_arr,
        X=X,
        Xn=Xn,
        resolution=resolution,
        contour_levels=contour_levels,
        cmap=cmap,
        surface=surface,
        norm=norm,
    )
    ax.set_title("Objective")
    if created_fig:
        plt.colorbar(contour, ax=ax, label="f(x, y)")
    if show and created_fig:
        plt.show()
    return ax


def _build_animation_frames(log_data: Mapping[str, Any], hold_frames: int) -> list[tuple[str, str, bool]]:
    frames: list[tuple[str, str, bool]] = []
    i_keys = _sorted_prefixed_keys(log_data, "i")
    for i_idx, i_key in enumerate(i_keys):
        iter_data = log_data[i_key]
        p_keys = _sorted_prefixed_keys(iter_data, "p")
        if not p_keys:
            continue
        for p_key in p_keys:
            frames.append((i_key, p_key, False))
        if i_idx < len(i_keys) - 1 and hold_frames > 0:
            for _ in range(hold_frames):
                frames.append((i_key, p_keys[-1], True))
    return frames


def plot_opt(
    log: str | Path | Mapping[str, Any],
    f: Callable[[np.ndarray], np.ndarray],
    bounds: Any,
    *,
    interval_ms: int = 300,
    hold_ms: int = 1200,
    resolution: int = 220,
    contour_levels: int = 60,
    cmap: str = "viridis",
    repeat: bool = True,
    figsize: tuple[float, float] = (12, 5),
    show: bool = True,
    save_path: str | Path | None = None,
    dpi: int = 120,
) -> FuncAnimation:
    """
    Animate optimization log with:
    - left subplot: plot_log information
    - right subplot: plot_f information
    """
    if interval_ms <= 0:
        raise ValueError("interval_ms must be > 0")
    if hold_ms < 0:
        raise ValueError("hold_ms must be >= 0")

    log_data = _load_log(log)
    bounds_arr = _as_bounds(bounds)

    hold_frames = int(np.ceil(hold_ms / interval_ms)) if hold_ms > 0 else 0
    frames = _build_animation_frames(log_data, hold_frames)
    if not frames:
        raise ValueError("No iteration/partition frames found in log.")

    surface = _compute_surface(f, bounds_arr, resolution)
    z_min = float(np.nanmin(surface[2]))
    z_max = float(np.nanmax(surface[2]))
    norm = Normalize(vmin=z_min, vmax=z_max)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax_right, label="f(x, y)")

    def _draw(frame_idx: int):
        i_key, p_key, is_hold = frames[frame_idx]
        iter_data = log_data[i_key]
        p_keys = _sorted_prefixed_keys(iter_data, "p")
        is_last_partition = p_key == p_keys[-1]

        ax_left.clear()
        ax_right.clear()

        plot_log(log_data, i_key, p_key, ax=ax_left, bounds=bounds_arr, show=False)
        ax_left.set_title("Partitioning")
        X = _as_points(iter_data["X"], f"{i_key}.X")
        Xn = _as_point(iter_data["Xn"], f"{i_key}.Xn") if is_last_partition and "Xn" in iter_data else None
        plot_f(
            f,
            bounds_arr,
            X,
            Xn,
            ax=ax_right,
            resolution=resolution,
            contour_levels=contour_levels,
            cmap=cmap,
            surface=surface,
            norm=norm,
            show=False,
        )
        ax_right.set_title("Objective")

        title = f"Iteration {int(i_key[1:]) + 1} - {p_key}"
        if is_hold:
            title = f"{title} (hold)"
        fig.suptitle(title)
        return ()

    # Draw first frame so notebook inline output is never blank.
    _draw(0)

    anim = FuncAnimation(
        fig,
        _draw,
        frames=len(frames),
        interval=interval_ms,
        blit=False,
        repeat=repeat,
    )

    # Keep a strong reference so notebook users can call plot_opt(...) without assigning.
    global _LAST_ANIMATION
    _LAST_ANIMATION = anim
    setattr(fig, "_tamubo_anim", anim)

    if save_path is not None:
        fps = max(1, int(round(1000 / interval_ms)))
        anim.save(str(save_path), dpi=dpi, fps=fps)

    if show:
        shown_in_notebook = _display_animation_if_notebook(anim, fig)
        if not shown_in_notebook:
            plt.show()
    return anim


__all__ = ["plot_log", "plot_f", "plot_opt"]
