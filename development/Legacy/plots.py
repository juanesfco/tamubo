import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

from tamubo.acquisition_functions import expected_improvement

Array = np.ndarray

# Plot parameters
colormap = 'viridis'
colorToSample = "#3366CC"
colorSampled = "#FF7F0E"
colorExcluded = "#D62728"
colorEBOBestX = "#FF00FF"
colorBOBestX = "#FC5A50"
contourfLevels = 100

def plot_iterations(log: dict, model, path: str | None = None):
    return None

def plot_iterations_2d(log: dict, path: str | None = None):
    domain = log['start']['domain']
    f = log['start']['oracle']
    plot_function_2d(domain, f, path)
    log_keys = list(log.keys())
    num_iterations = len(log_keys) - 3
    for it in range(num_iterations):
        key = 'ebo_it' + str(it)
        plot_partitions_2d(it, log[key], domain, f, path)

    return None

def plot_function_2d(domain: Array, f, path: str | None = None):
    N = 1000
    xa, xb = domain[0,0], domain[0,1]
    ya, yb = domain[1,0], domain[1,1]
    plotPadx = (xb - xa)/40
    plotPady = (yb - ya)/40

    x = np.linspace(xa, xb, N)
    y = np.linspace(ya, yb, N)
    XX, YY = np.meshgrid(x, y)
    xx = np.vstack([XX.ravel(),YY.ravel()]).T
    ZZ = f(xx)
    iZZmin = ZZ.argmin()
    xxmin = xx[iZZmin]
    zzmin = ZZ[iZZmin]
    ZZ = ZZ.reshape((N, N))

    fig = plt.figure(figsize=(8, 6))
    plt.contourf(XX, YY, ZZ, levels=contourfLevels, cmap=colormap)
    plt.colorbar(label='f(x, y)')
    plt.contour(XX, YY, ZZ, levels=20, colors='k', linewidths=0.3, alpha=0.5)
    plt.scatter(xxmin[0],xxmin[1], color=colorBOBestX, marker="*", label=f"Min: f({xxmin[0]:.3f}, {xxmin[1]:.3f}) = {zzmin:.3f}")
    Lx, Ly = domain[:,0]
    Rx, Ry = domain[:,1]
    plt.vlines(Lx, Ly, Ry, color=colorToSample, linewidth=1)
    plt.vlines(Rx, Ly, Ry, color=colorToSample, linewidth=1)
    plt.hlines(Ly, Lx, Rx, color=colorToSample, linewidth=1)
    plt.hlines(Ry, Lx, Rx, color=colorToSample, linewidth=1)
    plt.xlim(xa-plotPadx, xb+plotPadx)
    plt.ylim(ya-plotPady, yb+plotPady)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
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
    num_ploops = len(log_keys) - 4

    for p in range(num_ploops):
        key = 'ploop_' + str(p)
        ploop = log[key]
        plot_ploop_2d(it, p, ploop, domain, f, X, ei_xx, path)

    ploop = log['ploop_final']
    plot_ploop_2d(it, 'final', ploop, domain, f, X, ei_xx, path)

def plot_ploop_2d(it, p, ploop, domain, f, X, ei_xx, path = None):

    boxes = ploop["boxes"]
    n_boxes = len(boxes)

    # Things I need
    N = 400
    xa, xb = domain[0,0], domain[0,1]
    ya, yb = domain[1,0], domain[1,1]
    plotPadx = (xb - xa)/40
    plotPady = (yb - ya)/40

    x = np.linspace(xa, xb, N)
    y = np.linspace(ya, yb, N)
    XX, YY = np.meshgrid(x, y)
    xx = np.vstack([XX.ravel(),YY.ravel()]).T
    ZZ = f(xx)
    ZZ = ZZ.reshape((N, N))

    best_x = ploop['best_x']
    max_ei = ploop['max_ei']

    wx = xb - xa
    wy = yb - ya
    padx = (0.03*wx)/(int(n_boxes**(0.5)))
    pady = (0.03*wy)/(int(n_boxes**(0.5)))
    
    # Plotting
    fig = plt.figure(figsize=(24,6))
    plt.suptitle(f"Iteration {it+1} - Partition {p}")
    
    # Plot exactBO EI Top Bound
    ax1 = plt.subplot(1,3,1)

    norm = plt.Normalize(vmin=np.nanmin(boxes.ei_hi), vmax=np.nanmax(boxes.ei_hi))
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    def ei_to_color(ei_value):
        """Maps a single EI value to its color"""
        return sm.to_rgba(ei_value)
    
    legExcMark = True
    legSamMark = True
    for i in range(n_boxes):
        box = boxes[i]
        bounds = box.bounds
        #wx, wy = box.width
        #padx, pady = wx/20, wy/20
        
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
    
    plt.scatter(X[:,0], X[:,1], color=colorSampled, marker='o', label="Sampled")
    if best_x is not None:
        plt.scatter(best_x[0],best_x[1], color=colorEBOBestX, marker="X", label=f"exact BO Best X: EI = {max_ei:.2f}")
    
    plt.colorbar(sm, ax=ax1, label="Boxes EI Top Bound")
    plt.xlim(xa-plotPadx, xb+plotPadx)
    plt.ylim(ya-plotPady, yb+plotPady)
    plt.xlabel("x")
    plt.ylabel("y")

    # Plot function to minimize
    plt.subplot(1,3,2)
    plt.contourf(XX, YY, ZZ, levels=contourfLevels, cmap=colormap)
    plt.colorbar(label='f(x, y)')
    plt.contour(XX, YY, ZZ, levels=20, colors='k', linewidths=0.3, alpha=0.5)

    legExcMark = True
    legSamMark = True
    for i in range(n_boxes):
        box = boxes[i]
        bounds = box.bounds
        #wx, wy = box.width
        #padx, pady = wx/20, wy/20
        
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
    
    plt.scatter(X[:,0], X[:,1], color=colorSampled, marker='o', label="Sampled")
    if best_x is not None:
        plt.scatter(best_x[0],best_x[1], color=colorEBOBestX, marker="X", label=f"exact BO Best X: EI = {max_ei:.2f}")
    
    plt.xlim(xa-plotPadx, xb+plotPadx)
    plt.ylim(ya-plotPady, yb+plotPady)
    plt.xlabel("x")
    plt.ylabel("y")

    # Plot actual EI
    plt.subplot(1,3,3)
    plt.contourf(XX, YY, ei_xx.reshape((N,N)), levels=contourfLevels, cmap=colormap)
    plt.colorbar(label="EI(x, y)")
    plt.contour(XX, YY, ei_xx.reshape((N,N)), levels=20, colors='k', linewidths=0.3, alpha=0.5)
    
    legExcMark = True
    legSamMark = True
    for i in range(n_boxes):
        box = boxes[i]
        bounds = box.bounds
        #wx, wy = box.width
        #padx, pady = wx/20, wy/20
        
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

    plt.scatter(X[:,0], X[:,1], color=colorSampled, marker='o', label="Sampled")
    if best_x is not None:
        plt.scatter(best_x[0],best_x[1], color=colorEBOBestX, marker="X", label=f"exactBO Best X: EI = {max_ei:.2f}")
    best_i_ei_xx = np.argmax(ei_xx)
    plt.scatter(xx[best_i_ei_xx][0],xx[best_i_ei_xx][1], color=colorBOBestX, marker="*", label=f"BO Best X: EI = {np.max(ei_xx):.2f}")

    plt.xlim(xa-plotPadx, xb+plotPadx)
    plt.ylim(ya-plotPady, yb+plotPady)
    plt.xlabel("x")
    plt.ylabel("y")
    
    # Legend
    sampled_square_marker = Line2D([], [], marker='s', markersize=10,
                    markerfacecolor=colorSampled, markeredgecolor=colorSampled, alpha=0.25,
                    linestyle='None')
    sampled_dot_marker = Line2D([], [], marker='o', markersize=5,
                    markerfacecolor=colorSampled, markeredgecolor=colorSampled,
                    linestyle='None')
    sampled_marker = (sampled_square_marker, sampled_dot_marker)  # tuple for HandlerTuple

    toSample_square_marker = Line2D([], [], color=colorToSample, linestyle='None',
                    marker='s', markerfacecolor='none',
                    markeredgecolor=colorToSample)

    excluded_square_marker = Line2D([], [], color=colorExcluded, linestyle='None',
                    marker='s', markerfacecolor='none',
                    markeredgecolor=colorExcluded)
    
    eboBestX_marker = Line2D([], [], marker='X', markersize=8,
                    markerfacecolor=colorEBOBestX, markeredgecolor=colorEBOBestX,
                    linestyle='None')
    
    boBestX_marker = Line2D([], [], marker='*', markersize=8,
                    markerfacecolor=colorBOBestX, markeredgecolor=colorBOBestX,
                    linestyle='None')

    handles = [toSample_square_marker, excluded_square_marker, eboBestX_marker, boBestX_marker, sampled_marker]
    labels  = ['To Sample', 'Excluded', f'exactBO Best X: EI = {max_ei:.2f}', f'BO Best X: EI = {np.max(ei_xx):.2f}', 'Sampled']

    fig.legend(
        handles, labels,
        handler_map={tuple: HandlerTuple(ndivide=None)},
        loc='upper right',        # corner of the figure
        bbox_to_anchor=(0.9, 1),
        ncol=3
    )
    
    plt.show()
    if path:
        fig.savefig(f"{path}/eBO_2d_it{it+1}_p{p}.png", dpi=150, bbox_inches='tight')