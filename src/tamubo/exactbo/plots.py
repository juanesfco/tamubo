import numpy as np
import matplotlib.pyplot as plt
from .ei import expected_improvement

Array = np.ndarray

# Plot parameters
colormap = 'viridis'
colorToSample = "#3366CC"
colorSampled = "#FF7F0E"
colorExcluded = "#D62728"
contourfLevels = 100

def plot_iterations(log: dict, model, path: str | None = None):
    return None

def plot_iterations_2d(log: dict, model, path: str | None = None):
    domain = log['start']['domain']
    f = log['start']['oracle']
    plot_function_2d(log, domain, f, path)
    log_keys = list(log.keys())
    num_iterations = len(log_keys) - 3
    for it in range(num_iterations):
        key = 'ebo_it' + str(it)
        plot_partitions_2d(it, log[key], model, domain, f, path)

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
    plt.plot(xxmin[0],xxmin[1], color=colorExcluded, marker="*")
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

def plot_partitions_2d(it: int, log: dict, model, domain: Array, f, path: str | None = None):
    # Things I need
    N = 400
    xa, xb = domain[0,0], domain[0,1]
    ya, yb = domain[1,0], domain[1,1]

    x = np.linspace(xa, xb, N)
    y = np.linspace(ya, yb, N)
    XX, YY = np.meshgrid(x, y)
    xx = np.vstack([XX.ravel(),YY.ravel()]).T

    ei_xx = expected_improvement(xx, model)

    X = log['start'].X

    log_keys = list(log.keys())
    num_ploops = len(log_keys) - 3
    ploops_keys = ['ploop_start'] + ['ploop_'+str(i) for i in range(num_ploops)] + ['ploop_final']

    for p in range(len(ploops_keys)):
        key = ploops_keys[p]
        ploop = log[key]
        plot_ploop_2d(it, p, ploop, domain, f, X, ei_xx, path)

    

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

    
    # Plotting
    fig = plt.figure(figsize=(24,6))
    plt.suptitle(f"Iteration {it+1} - Partition {p}")
    ax1 = plt.subplot(1,3,1)

    norm = plt.Normalize(vmin=ei_xx.min(), vmax=ei_xx.max())
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
        Lx, Ly = bounds[:,0]
        Rx, Ry = bounds[:,1]
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
    plt.colorbar(sm, ax=ax1, label="EI.hi(Box)")
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
    # Plot partitions
    legExcMark = True
    legSamMark = True
    for i in range(len(boxes)):
        box = boxes[i]
        bounds = box.bounds
        Lx, Ly = bounds[:,0]
        Rx, Ry = bounds[:,1]
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
    plt.legend(ncol=2 if legExcMark else 3)

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
        Lx, Ly = bounds[:,0]
        Rx, Ry = bounds[:,1]
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
    #plt.legend(ncol=2)
    plt.show()
    if path:
        fig.savefig(f"{path}/eBO_2d_it{it+1}_p{p}.png", dpi=150, bbox_inches='tight')