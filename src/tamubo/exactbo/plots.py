import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

from .ei import expected_improvement

Array = np.ndarray

# Plot parameters
colormap = 'viridis'
colorToSample = "#3366CC"
colorSampled = "#FF7F0E"
colorExcluded = "#D62728"
colorBOBestX = "#008000"
contourfLevels = 100

def plot_iterations(log: dict, model, path: str | None = None):
    return None

def plot_iterations_2d(log: dict, path: str | None = None):
    domain = log['start']['domain']
    f = log['start']['oracle']
    plot_function_2d(log, domain, f, path)
    log_keys = list(log.keys())
    num_iterations = len(log_keys) - 3
    for it in range(num_iterations):
        key = 'ebo_it' + str(it)
        plot_partitions_2d(it, log[key], domain, f, path)

    return None

def plot_function_2d(log: dict, domain: Array, f, path: str | None = None):
    N = 400
    xa, xb = domain[0,0], domain[0,1]
    ya, yb = domain[1,0], domain[1,1]

    x = np.linspace(xa, xb, N)
    y = np.linspace(ya, yb, N)
    XX, YY = np.meshgrid(x, y)
    xx = np.vstack([XX.ravel(),YY.ravel()]).T
    ZZ = f(xx)
    iZZmin = ZZ.argmin()
    xxmin = xx[iZZmin]
    ZZ = ZZ.reshape((N, N))

    fig = plt.figure(figsize=(8, 6))
    plt.contourf(XX, YY, ZZ, levels=contourfLevels, cmap=colormap)
    plt.colorbar(label='f(x, y)')
    plt.contour(XX, YY, ZZ, levels=20, colors='k', linewidths=0.3, alpha=0.5)
    plt.scatter(xxmin[0],xxmin[1], color=colorExcluded, marker="*")
    Lx, Ly = domain[:,0]
    Rx, Ry = domain[:,1]
    plt.vlines(Lx, Ly, Ry, color=colorToSample, linewidth=1)
    plt.vlines(Rx, Ly, Ry, color=colorToSample, linewidth=1)
    plt.hlines(Ly, Lx, Rx, color=colorToSample, linewidth=1)
    plt.hlines(Ry, Lx, Rx, color=colorToSample, linewidth=1)
    plt.xlim(xa-1, xb+1)
    plt.ylim(ya-1, yb+1)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Iteration 0")
    plt.show()
    if path:
        fig.savefig(f"{path}/eBO_2d_it0.png", dpi=150, bbox_inches='tight')

def plot_partitions_2d(it: int, log: dict, domain: Array, f, path: str | None = None):
    # Things I need
    N = 400
    xa, xb = domain[0,0], domain[0,1]
    ya, yb = domain[1,0], domain[1,1]

    x = np.linspace(xa, xb, N)
    y = np.linspace(ya, yb, N)
    XX, YY = np.meshgrid(x, y)
    xx = np.vstack([XX.ravel(),YY.ravel()]).T

    model = log["model"]
    ei_xx = expected_improvement(xx, model)

    X = log['start'].X

    log_keys = list(log.keys())
    num_ploops = len(log_keys) - 3

    for p in range(num_ploops):
        key = 'ploop_' + str(p)
        ploop = log[key]
        plot_ploop_2d(it, p, ploop, domain, f, X, ei_xx, path)

    ploop = log['ploop_final']
    plot_ploop_2d(it, 'final', ploop, domain, f, X, ei_xx, path)

def plot_ploop_2d(it, p, ploop, domain, f, X, ei_xx, path = None):

    boxes = ploop["boxes"]

    # Things I need
    N = 400
    xa, xb = domain[0,0], domain[0,1]
    ya, yb = domain[1,0], domain[1,1]

    x = np.linspace(xa, xb, N)
    y = np.linspace(ya, yb, N)
    XX, YY = np.meshgrid(x, y)
    xx = np.vstack([XX.ravel(),YY.ravel()]).T
    ZZ = f(xx)
    ZZ = ZZ.reshape((N, N))

    best_x = ploop['best_x']
    max_ei = ploop['max_ei']
    
    # Plotting
    fig = plt.figure(figsize=(24,6))
    plt.suptitle(f"Iteration {it+1} - Partition {p}")

    # First entry: dot on top of a square background
    square_bg = Line2D([], [], marker='s', markersize=10,
                    markerfacecolor='white', markeredgecolor='black',
                    linestyle='None')
    dot_fg = Line2D([], [], marker='o', markersize=5,
                    markerfacecolor='black', markeredgecolor='black',
                    linestyle='None')
    dot_on_square = (square_bg, dot_fg)  # tuple for HandlerTuple

    # Blue line with empty square marker
    blue_line = Line2D([], [], color='blue', linestyle='-',
                    marker='s', markerfacecolor='none',
                    markeredgecolor='blue')

    # Red line with empty square marker
    red_line = Line2D([], [], color='red', linestyle='-',
                    marker='s', markerfacecolor='none',
                    markeredgecolor='red')

    handles = [dot_on_square, blue_line, red_line]
    labels  = ['Points', 'Blue curve', 'Red curve']

    ax1 = plt.subplot(1,3,1)

    norm = plt.Normalize(vmin=np.nanmin(boxes.ei_hi), vmax=np.nanmax(boxes.ei_hi))
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    def ei_to_color(ei_value):
        """Maps a single EI value to its color"""
        return sm.to_rgba(ei_value)
    
    #plt.contourf(XX, YY, ei_xx.reshape((N,N)), levels=contourfLevels, cmap=colormap)
    #plt.colorbar(sm, label="EI.hi(Box)")
    #plt.contour(XX, YY, ei_xx.reshape((N,N)), levels=20, colors='k', linewidths=0.3, alpha=0.5)

    plt.scatter(X[:,0], X[:,1], color=colorSampled, marker='o', label="Sampled")
    
    # Plot partitions
    legExcMark = True
    legSamMark = True
    for i in range(len(boxes)):
        box = boxes[i]
        bounds = box.bounds
        wx, wy = box.width
        padx, pady = wx/20, wy/20
        
        Lx, Ly = bounds[:,0]
        Rx, Ry = bounds[:,1]

        Lx, Rx = Lx + padx, Rx - padx
        Ly, Ry = Ly + pady, Ry - pady

        if not box.active:
            plt.vlines(Lx, Ly, Ry, color=colorExcluded, linewidth=1, label="Excluded" if legExcMark else None)
            plt.vlines(Rx, Ly, Ry, color=colorExcluded, linewidth=1)
            plt.hlines(Ly, Lx, Rx, color=colorExcluded, linewidth=1)
            plt.hlines(Ry, Lx, Rx, color=colorExcluded, linewidth=1)
            legExcMark = False
        else:
            if box.sampled:
                plt.fill_betweenx([Ly, Ry], Lx, Rx, color=colorSampled, alpha=0.25)
                plt.fill_between([Lx, Rx], Ly, Ry, color=colorSampled, alpha=0.25)
            else:
                try:
                    color = ei_to_color(box.ei.hi)
                except:
                    color = 'w'
                plt.vlines(Lx, Ly, Ry, color=colorToSample, linewidth=1, label="To sample" if legSamMark else None)
                plt.vlines(Rx, Ly, Ry, color=colorToSample, linewidth=1)
                plt.hlines(Ly, Lx, Rx, color=colorToSample, linewidth=1)
                plt.hlines(Ry, Lx, Rx, color=colorToSample, linewidth=1)
                plt.fill_betweenx([Ly, Ry], Lx, Rx, color=color, alpha=1)
                plt.fill_between([Lx, Rx], Ly, Ry, color=color, alpha=1)
                legSamMark = False
    
    if best_x is not None:
        plt.scatter(best_x[0],best_x[1], color=colorExcluded, marker="*", label=f"exact BO Best X: EI = {max_ei:.2f}")
    plt.colorbar(sm, ax=ax1, label="Boxes EI Top Bound")
    plt.xlim(xa-1, xb+1)
    plt.ylim(ya-1, yb+1)
    plt.xlabel("x")
    plt.ylabel("y")
    #plt.legend(ncol=2)

    plt.subplot(1,3,2)
    plt.contourf(XX, YY, ZZ, levels=contourfLevels, cmap=colormap)
    plt.colorbar(label='f(x, y)')
    plt.contour(XX, YY, ZZ, levels=20, colors='k', linewidths=0.3, alpha=0.5)
    plt.scatter(X[:,0], X[:,1], color=colorSampled, marker='o', label="Sampled")

    if best_x is not None:
        plt.scatter(best_x[0],best_x[1], color=colorExcluded, marker="*", label=f"exact BO Best X: EI = {max_ei:.2f}")

    # Plot partitions
    legExcMark = True
    legSamMark = True
    for i in range(len(boxes)):
        box = boxes[i]
        bounds = box.bounds
        wx, wy = box.width
        padx, pady = wx/20, wy/20
        
        Lx, Ly = bounds[:,0]
        Rx, Ry = bounds[:,1]

        Lx, Rx = Lx + padx, Rx - padx
        Ly, Ry = Ly + pady, Ry - pady
        if not box.active:
            plt.vlines(Lx, Ly, Ry, color=colorExcluded, linewidth=1, label="Excluded" if legExcMark else None)
            plt.vlines(Rx, Ly, Ry, color=colorExcluded, linewidth=1)
            plt.hlines(Ly, Lx, Rx, color=colorExcluded, linewidth=1)
            plt.hlines(Ry, Lx, Rx, color=colorExcluded, linewidth=1)
            legExcMark = False
        else:
            if box.sampled:
                plt.fill_betweenx([Ly, Ry], Lx, Rx, color=colorSampled, alpha=0.25)
                plt.fill_between([Lx, Rx], Ly, Ry, color=colorSampled, alpha=0.25)
            else:
                plt.vlines(Lx, Ly, Ry, color=colorToSample, linewidth=1, label="To sample" if legSamMark else None)
                plt.vlines(Rx, Ly, Ry, color=colorToSample, linewidth=1)
                plt.hlines(Ly, Lx, Rx, color=colorToSample, linewidth=1)
                plt.hlines(Ry, Lx, Rx, color=colorToSample, linewidth=1)
                legSamMark = False
    plt.xlim(xa-1, xb+1)
    plt.ylim(ya-1, yb+1)
    plt.xlabel("x")
    plt.ylabel("y")
    #plt.legend(ncol=2 if legExcMark else 3)

    plt.subplot(1,3,3)
    plt.contourf(XX, YY, ei_xx.reshape((N,N)), levels=contourfLevels, cmap=colormap)
    plt.colorbar(label="EI(x, y)")
    plt.contour(XX, YY, ei_xx.reshape((N,N)), levels=20, colors='k', linewidths=0.3, alpha=0.5)
    plt.scatter(X[:,0], X[:,1], color=colorSampled, marker='o', label="Sampled")

    # Plot partitions
    legExcMark = True
    legSamMark = True
    for i in range(len(boxes)):
        box = boxes[i]
        bounds = box.bounds
        wx, wy = box.width
        padx, pady = wx/20, wy/20
        
        Lx, Ly = bounds[:,0]
        Rx, Ry = bounds[:,1]

        Lx, Rx = Lx + padx, Rx - padx
        Ly, Ry = Ly + pady, Ry - pady
        if not box.active:
            plt.vlines(Lx, Ly, Ry, color=colorExcluded, linewidth=1, label="Excluded" if legExcMark else None)
            plt.vlines(Rx, Ly, Ry, color=colorExcluded, linewidth=1)
            plt.hlines(Ly, Lx, Rx, color=colorExcluded, linewidth=1)
            plt.hlines(Ry, Lx, Rx, color=colorExcluded, linewidth=1)
            legExcMark = False
        else:
            if box.sampled:
                plt.fill_betweenx([Ly, Ry], Lx, Rx, color=colorSampled, alpha=0.25)
                plt.fill_between([Lx, Rx], Ly, Ry, color=colorSampled, alpha=0.25)
            else:
                plt.vlines(Lx, Ly, Ry, color=colorToSample, linewidth=1, label="To sample" if legSamMark else None)
                plt.vlines(Rx, Ly, Ry, color=colorToSample, linewidth=1)
                plt.hlines(Ly, Lx, Rx, color=colorToSample, linewidth=1)
                plt.hlines(Ry, Lx, Rx, color=colorToSample, linewidth=1)
                legSamMark = False

    # If exactBO found Best X, plot it
    if best_x is not None:
        plt.scatter(best_x[0],best_x[1], color=colorExcluded, marker="*", label=f"exactBO Best X: EI = {max_ei:.2f}")
    # Plot BO Best X
    best_i_ei_xx = np.argmax(ei_xx)
    plt.scatter(xx[best_i_ei_xx][0],xx[best_i_ei_xx][1], color=colorBOBestX, marker="*", label=f"BO Best X: EI = {np.max(ei_xx):.2f}")

    plt.xlim(xa-1, xb+1)
    plt.ylim(ya-1, yb+1)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(ncol=2 if legExcMark else 3)
    plt.show()
    if path:
        fig.savefig(f"{path}/eBO_2d_it{it+1}_p{p}.png", dpi=150, bbox_inches='tight')