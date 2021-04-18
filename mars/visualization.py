import matplotlib.pyplot as plt
import numpy as np
from .configuration import Configuration
config = Configuration()
del Configuration
from mars.utils import binary_cmap, get_number_of_rows_and_columns
import os
import torch
from matplotlib.lines import Line2D



def generate_trajectories(states_init, closed_loop_dynamics, dt, horizon):
    if isinstance(states_init, np.ndarray):
        states_init = torch.tensor(np.copy(states_init), dtype=config.ptdtype)
    nindex = states_init.shape[0]
    ndim = states_init.shape[1]
    
    trajectories = torch.zeros((nindex, ndim, horizon+1), dtype=config.ptdtype)
    grad_field = torch.zeros((nindex, ndim, horizon), dtype=config.ptdtype)
    trajectories[:, :, 0] = states_init
    
    with torch.no_grad():
        for t in range(1, horizon+1):
            trajectories[:, :, t] = closed_loop_dynamics(trajectories[:, :, t - 1])
            grad_field[:, :, t-1] = (trajectories[:, :, t] - trajectories[:, :, t-1]) / dt
    return trajectories[:,:, 0:-1], grad_field


def plot_roa(roa, plot_limits, plot_labels, full_path):
    """ take roa data as a 2D matrix and save
    the plot in full_path

    Parameter
    -------------
    roa: N x N grid
    plot_limits: [(xmin, xmax), (ymin, ymax)]
    full_path: path to save the image file
    plot_labels: [label of x axis, label of y axis]
    """

    dir_path = os.path.dirname(full_path)
    filename = os.path.basename(full_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    fig = plt.figure(figsize=(10, 10), dpi=config.dpi, frameon=False)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    ax = plt.subplot(111)
    color=[None]
    color[0] = (0, 158/255, 115/255)       # ROA - bluish-green
    # True ROA
    z = roa
    alpha = 1
    ax.set_xlabel(plot_labels[0], fontsize=50)
    ax.set_ylabel(plot_labels[1], fontsize=50)
    ax.contour(z.T, origin='lower', extent=plot_limits.ravel(), colors=(color[0],), linewidths=1)
    ax.imshow(z.T, origin='lower', extent=plot_limits.ravel(), cmap=binary_cmap(color[0]), alpha=alpha)
    ax.tick_params(axis='both', which='major', labelsize=30, grid_linewidth=20)
    ax.set_aspect(0.4)
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.savefig(full_path, dpi=config.dpi)
    # plt.close(fig)


def plot_nested_roas(grid_size, ind_small, ind_big, plot_limits, plot_labels, ind_points=None, ind_gap_stable=None, ind_true_roa=None, full_path=None, alpha=0.5):
    """ take a grid and the indices for small and big ROAs and plot them overlaid
    the plot in full_path

    Parameter
    -------------
    grid_size: size of the encompassing grid for both ROAs
    ind_small: binary ndarray vector, indices for the points of the inner ROA
    ind_big: binary ndarray vector, indices for points of the outer ROA
    ind_points, ind_gap_stable, ind_true_roa: optional, a binary vector with the same size of prod(grid_size).
            Each group of points are plotted with different colour to differentiate betweent the nature of points.
    plot_limits: [(xmin, xmax), (ymin, ymax)]
    full_path: path to save the image file
    """
    if full_path is not None: 
        dir_path = os.path.dirname(full_path)
        filename = os.path.basename(full_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    fig = plt.figure(figsize=(10, 10), dpi=config.dpi, frameon=False)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    ax = plt.subplot(111)
    color=[None, None, None, None, None]
    color[0] = (80/255, 100/255, 250/255)       # ROA - bluish-green
    color[1] = (158/255, 0, 115/255)
    color[2] = (158/255, 115/255, 0)
    color[3] = (0, 158/255, 115/255)
    color[4] = (255/255, 0, 0)
    nindex = grid_size[0] * grid_size[1]
    roa_small = np.zeros(nindex) != 0
    roa_big = np.zeros(nindex) != 0
    roa_small[ind_small] = True
    roa_big[ind_big] = True
    roa_small = roa_small.reshape(grid_size)
    roa_big = roa_big.reshape(grid_size)
    ax.imshow(roa_big.T, origin='lower', extent=plot_limits.ravel(), cmap=binary_cmap(color[1]), alpha=alpha)
    ax.contour(roa_small.T, origin='lower', extent=plot_limits.ravel(), colors=(color[0],), linewidths=0.05, alpha=1.0)
    ax.imshow(roa_small.T, origin='lower', extent=plot_limits.ravel(), cmap=binary_cmap(color[0]), alpha=1.0)
    ax.contour(roa_big.T, origin='lower', extent=plot_limits.ravel(), colors=(color[1],), linewidths=0.05, alpha=1.0)
    if ind_gap_stable is not None:
        roa_gap_stable = np.zeros(nindex) != 0
        roa_gap_stable[ind_gap_stable] = True
        roa_gap_stable = roa_gap_stable.reshape(grid_size)
        ax.imshow(roa_gap_stable.T, origin='lower', extent=plot_limits.ravel(), cmap=binary_cmap(color[2]))
    if ind_true_roa is not None:
        roa_true = np.zeros(nindex) != 0
        roa_true[ind_true_roa] = True
        roa_true = roa_true.reshape(grid_size)
        ax.contour(roa_true.T, origin='lower', extent=plot_limits.ravel(), colors=(color[3],), linewidths=0.05, alpha=1.0)
        ax.imshow(roa_true.T, origin='lower', extent=plot_limits.ravel(), cmap=binary_cmap(color[3]), alpha=0.3)
    if ind_points is not None:
        grid_points = np.zeros(nindex) != 0
        grid_points[ind_points] = True
        grid_points = grid_points.reshape(grid_size)
        print(grid_points.sum())
        ax.imshow(grid_points.T, origin='lower', extent=plot_limits.ravel(), cmap=binary_cmap(color[4]), alpha=alpha)
    ax.set_xlabel(plot_labels[0], fontsize=50)
    ax.set_ylabel(plot_labels[1], fontsize=50)
    ax.tick_params(axis='both', which='major', labelsize=30, grid_linewidth=20)
    ax.set_aspect(0.4)
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    if full_path is not None:
        plt.savefig(full_path, dpi=config.dpi)
    # plt.close(fig)


def plot_levelsets(interval, margin, func, plot_labels, res=32, figsize=(7,7), nlevels=10, full_path=None):
    """ take coordinate intervals and the height function and save
    the plot in full_path

    Parameter
    -------------
    interval: [[xmin, xmax], [ymin, ymax]], the limits of the axis to plot
    margin: A small value added to the plot limit
    full_path: path to save the image file
    res: resolution of the heatmap
    func: A torch (scalar-valued) function whose levelsets are to be pllotted
    nlevels = number of level sets (None for no level set)
    """
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    xmin, xmax = interval[0]
    ymin, ymax = interval[1]
    xmin -= margin
    ymin -= margin
    xmax += margin
    ymax += margin
    x = np.linspace(xmin, xmax, res)
    y = np.linspace(ymin, ymax, res)
    xv, yv = np.meshgrid(x, y)
    xyv = np.vstack([xv.flatten(), yv.flatten()]).transpose()
    zv = []
    with torch.no_grad():
        for p in xyv:
            V = torch.tensor(p, dtype=torch.float64).view(1, 2)
            zv.append(func(V))
    zv = np.array(zv).transpose()
    fig, ax = plt.subplots(figsize=figsize, dpi=config.dpi, frameon=False)
    ax.pcolormesh(xv, yv, zv.reshape(xv.shape), cmap='viridis')
    if nlevels is not None:
        CS = ax.contour(xv, yv, zv.reshape(xv.shape), cmap='YlOrBr', levels=nlevels)
        ax.clabel(CS, inline=1, fontsize=10)
    ax.set_xlabel(plot_labels[0], fontsize=50)
    ax.set_ylabel(plot_labels[1], fontsize=50)
    ax.tick_params(axis='both', which='major', labelsize=30, grid_linewidth=20)
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    if full_path is not None:
        plt.savefig(full_path, dpi=config.dpi)
    # plt.close(fig)


def plot_scalar(x_axis_vals, y_axis_vals, plot_labels, full_path=None):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig, ax = plt.subplots(figsize=(10, 10), dpi=config.dpi, frameon=False)
    ax.plot(x_axis_vals, y_axis_vals, linewidth=5, color=(0, 0, 0))
    ax.set_xlabel(plot_labels[0], fontsize=50)
    ax.set_ylabel(plot_labels[1], fontsize=50)
    ax.tick_params(axis='both', which='major', labelsize=30, grid_linewidth=20)
    plt.xticks(np.arange(min(x_axis_vals), max(x_axis_vals)+1, 20))
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    if full_path is not None:
        plt.savefig(full_path, dpi=config.dpi)


def plot_phase_portrait(initial_states, list_of_closed_loop_dynamics, dt, horizon, skip, 
                        axis_denormalized_factors=[1.0, 1.0], plot_ticks=[None, None], 
                        plot_labels=[None, None], alpha=1.0, plot_legend=True, roa_args=None, full_path=None):
    
    # Plot Trajectories
    if not isinstance(initial_states, torch.Tensor):
        initial_states = torch.tensor(initial_states, dtype=config.ptdtype)
    if not isinstance(list_of_closed_loop_dynamics, list):
        list_of_closed_loop_dynamics = [list_of_closed_loop_dynamics]
    n_trajs, sdim = initial_states.shape
    colors = ['r', 'b']
    fig, ax = plt.subplots(figsize=(10, 10), dpi=config.dpi, frameon=False)
    for i, closed_loop_dynamics in enumerate(list_of_closed_loop_dynamics):    
        pt_trajectories, dx_dt = generate_trajectories(initial_states, closed_loop_dynamics, dt, horizon)
        trajectories = pt_trajectories.detach().numpy()
        dx_dt = dx_dt.detach().numpy()
        dx_dt_normalized = dx_dt / np.linalg.norm(dx_dt, ord=2, axis=1, keepdims=True)
        for n in range(n_trajs):
            x = trajectories[n, 0, :] * np.rad2deg(axis_denormalized_factors[0])
            y = trajectories[n, 1, :] * np.rad2deg(axis_denormalized_factors[1])
            ax.plot(x, y, '--', color=colors[i], linewidth=0.25, alpha=alpha)
            ax.quiver(x[::skip], y[::skip], dx_dt[n, 0, ::skip], dx_dt[n, 1, ::skip], 
                  scale=100, pivot='mid', headwidth=5, headlength=6, color=colors[i], alpha=alpha)
    # Plot ROA
    if roa_args is not None:
        grid_size = roa_args["grid_size"]
        ind_small = roa_args["ind_small"]
        ind_big = roa_args["ind_big"]
        ind_gap_stable = roa_args["ind_gap_stable"]
        ind_true_roa = roa_args["ind_true_roa"]
        alpha = roa_args["alpha"]

        color=[None, None, None, None]
        color[0] = (0, 158/255, 115/255)       # ROA - bluish-green
        color[1] = (158/255, 0, 115/255)
        color[2] = (158/255, 115/255, 0)
        color[3] = (0, 158/255, 115/255) 
        nindex = grid_size[0] * grid_size[1]
        roa_small = np.zeros(nindex) != 0
        roa_big = np.zeros(nindex) != 0
        roa_small[ind_small] = True
        roa_big[ind_big] = True
        roa_small = roa_small.reshape(grid_size)
        roa_big = roa_big.reshape(grid_size)
        roa_small = roa_small.reshape(grid_size)
        roa_big = roa_big.reshape(grid_size)
        plot_limits = np.column_stack((- np.rad2deg([axis_denormalized_factors[0], axis_denormalized_factors[1]]), 
                                        np.rad2deg([axis_denormalized_factors[0], axis_denormalized_factors[1]])))


        ax.contour(roa_small.T, origin='lower', extent=plot_limits.ravel(), colors=(color[0],), linewidths=0.05, alpha=1.0)
        ax.imshow(roa_small.T, origin='lower', extent=plot_limits.ravel(), cmap=binary_cmap(color[0]), alpha=1.0)
        ax.contour(roa_big.T, origin='lower', extent=plot_limits.ravel(), colors=(color[1],), linewidths=0.05, alpha=1.0)
        ax.imshow(roa_big.T, origin='lower', extent=plot_limits.ravel(), cmap=binary_cmap(color[1]), alpha=alpha)
        if ind_gap_stable is not None:
            roa_gap_stable = np.zeros(nindex) != 0
            roa_gap_stable[ind_gap_stable] = True
            roa_gap_stable = roa_gap_stable.reshape(grid_size)
            ax.contour(roa_gap_stable.T, origin='lower', extent=plot_limits.ravel(), colors=(color[2],), linewidths=0.05, alpha=1.0)
            ax.imshow(roa_gap_stable.T, origin='lower', extent=plot_limits.ravel(), cmap=binary_cmap(color[2]), alpha=1.0)
        if ind_true_roa is not None:
            roa_true = np.zeros(nindex) != 0
            roa_true[ind_true_roa] = True
            roa_true = roa_true.reshape(grid_size)
            ax.contour(roa_true.T, origin='lower', extent=plot_limits.ravel(), colors=(color[3],), linewidths=0.05, alpha=1.0)
            ax.imshow(roa_true.T, origin='lower', extent=plot_limits.ravel(), cmap=binary_cmap(color[3]), alpha=0.3)
        
        if plot_legend == 'roa':
            proxy = [plt.Rectangle((0,0), 1, 1, fc=c) for c in color]    
            legend = ax.legend(proxy, [r'$\hat{\mathcal{S}}$', r'$\hat{\mathcal{S}} \times 5$', r'$\mathcal{S}_\pi \cap \mathcal{S}$', r'$\mathcal{S}_\pi$'], loc='upper right', fontsize=20)
            legend.get_frame().set_alpha(1.)
        elif plot_legend == 'traj':
            lines = [Line2D([0], [0], color=c, linewidth=3.0, linestyle='-', alpha=alpha) for c in colors]    
            labels = ['Untrained Controller', 'Trained Controller']
            legend = ax.legend(lines, labels, fontsize=20)
            legend.get_frame().set_alpha(1.)     
        elif plot_legend == 'all':
            patches = [plt.Rectangle((0,0), 1, 1, fc=c) for c in color]    
            labels_patches = [r'$\hat{\mathcal{S}}$', r'$\hat{\mathcal{S}} \times 5$', r'$\mathcal{S}_\pi \cap \mathcal{S}$', r'$\mathcal{S}_\pi$']
            lines = [Line2D([0], [0], color=c, linewidth=3.0, linestyle='-', alpha=alpha) for c in colors]    
            labels_lines = ['Untrained Controller', 'Trained Controller']
            proxy = patches + lines
            labels = labels_patches + labels_lines
            legend = ax.legend(proxy, labels, fontsize=20)
            legend.get_frame().set_alpha(1.)    

    ax.set_xlabel(plot_labels[0], fontsize=20)
    ax.set_ylabel(plot_labels[1], fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=15, grid_linewidth=15)
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)

    ax.set_aspect(axis_denormalized_factors[0] / axis_denormalized_factors[1] / 1.2)
    ax.set_xlim(axis_denormalized_factors[0])
    ax.set_ylim(axis_denormalized_factors[1])
    ax.xaxis.set_ticks(plot_ticks[0])
    ax.yaxis.set_ticks(plot_ticks[1])

    if full_path is not None:
        plt.savefig(full_path, dpi=config.dpi)


def plot_trajs_on_levelsets(interval, initial_states, list_of_closed_loop_dynamics, func, dt, horizon, skip, 
                        margin=10, nlevels=10, res=30, axis_denormalized_factors=[1.0, 1.0], plot_ticks=[None, None], 
                        plot_labels=[None, None], alpha=1.0, log_values=True, legend_type="controller", roa_args=None, full_path=None):
    """
    interval: The interval in the normalized coordinates for which the system is simulated
    """

    # plot levelsets
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    colors = [(200/255, 200/255, 0), (158/255, 0, 115/255)]
    colors_levelsets = [(0, 255/255, 200/255), (255/255, 0, 0)]
    xmin, xmax = interval[0]
    ymin, ymax = interval[1]
    xmin -= margin
    ymin -= margin
    xmax += margin
    ymax += margin
    x = np.linspace(xmin, xmax, res)
    y = np.linspace(ymin, ymax, res)
    xv, yv = np.meshgrid(x, y)
    xyv = np.vstack([xv.flatten(), yv.flatten()]).transpose()
    factor = np.array([[axis_denormalized_factors[0], axis_denormalized_factors[1]]])
    xyv = xyv / factor
    zv = []
    with torch.no_grad():
        for p in xyv:
            V = torch.tensor(p, dtype=torch.float64).view(1, 2)
            zv.append(func(V))
    zv = np.array(zv).transpose()
    fig, ax = plt.subplots(figsize=(10,10), dpi=config.dpi, frameon=False)
    ax.pcolormesh(xv, yv, zv.reshape(xv.shape))
    if nlevels is not None:
        if log_values:
            CS = ax.contour(xv, yv, np.log(zv.reshape(xv.shape) + 0.1), cmap='Wistia', levels=nlevels, alpha=0.7)
        else:
            CS = ax.contour(xv, yv, zv.reshape(xv.shape), levels=nlevels, colors=colors_levelsets, alpha=0.7)
        ax.clabel(CS, inline=1, fontsize=15)
            
    ax.set_xlabel(plot_labels[0], fontsize=30)
    ax.set_ylabel(plot_labels[1], fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=20, grid_linewidth=20)
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)

    # Plot Trajectories
    if not isinstance(initial_states, torch.Tensor):
        initial_states = torch.tensor(initial_states, dtype=config.ptdtype)
    if not isinstance(list_of_closed_loop_dynamics, list):
        list_of_closed_loop_dynamics = [list_of_closed_loop_dynamics]
    n_trajs, sdim = initial_states.shape
    
    for i, closed_loop_dynamics in enumerate(list_of_closed_loop_dynamics):    
        pt_trajectories, dx_dt = generate_trajectories(initial_states, closed_loop_dynamics, dt, horizon)
        trajectories = pt_trajectories.detach().numpy()
        dx_dt = dx_dt.detach().numpy()
        dx_dt_normalized = dx_dt / np.linalg.norm(dx_dt, ord=2, axis=1, keepdims=True)
        for n in range(n_trajs):
            x = trajectories[n, 0, :] * axis_denormalized_factors[0]
            y = trajectories[n, 1, :] * axis_denormalized_factors[1]
            ax.plot(x, y, '--', color=colors[i], linewidth=1.0, alpha=0.9)
            ax.quiver(x[::skip], y[::skip], dx_dt[n, 0, ::skip], dx_dt[n, 1, ::skip], 
                  scale=100, pivot='mid', headwidth=6, headlength=7, color=colors[i], alpha=0.7)
            
    if legend_type=="controller":
        lines = [Line2D([0], [0], color=c, linewidth=3.0, linestyle='--', alpha=0.9) for c in colors]
        labels = ['Untrained Controller', 'Trained Controller']
        legend = ax.legend(lines, labels, fontsize=20)
        legend.get_frame().set_alpha(0.5)
    elif legend_type=="sampler":
        lines = [Line2D([0], [0], color=c, linewidth=3.0, linestyle='-', alpha=0.9) for c in colors_levelsets] + [Line2D([0], [0], color=c, linewidth=3.0, linestyle='--', alpha=0.9) for c in colors[0:len(list_of_closed_loop_dynamics)]]
        labels = ['Safe set', 'Expanded safe set', 'Trajectories']
        legend = ax.legend(lines, labels, fontsize=20)
        legend.get_frame().set_alpha(0.5)
        
    
    

    ax.xaxis.set_ticks(plot_ticks[0])
    ax.yaxis.set_ticks(plot_ticks[1])

    if full_path is not None:
        plt.savefig(full_path, dpi=config.dpi)
    # plt.close(fig)


def plot_scalar_function_over_trajectories(initial_states, list_of_closed_loop_dynamics, scalar_function, dt, 
                                            horizon, plot_ticks=[None, None], plot_labels=[None, None], alpha=1.0, full_path=None):
    if not isinstance(initial_states, torch.Tensor):
        initial_states = torch.tensor(initial_states, dtype=config.ptdtype)
    if not isinstance(list_of_closed_loop_dynamics, list):
        list_of_closed_loop_dynamics = [list_of_closed_loop_dynamics]
    n_trajs, sdim = initial_states.shape
    colors = [(158/255, 0, 115/255), (200/255, 200/255, 0)]
    p = [] * len(list_of_closed_loop_dynamics)
    fig, ax = plt.subplots(figsize=(10, 10), dpi=config.dpi, frameon=False)
    scalar_values = np.zeros((int(n_trajs), int(horizon)))
    for i, closed_loop_dynamics in enumerate(list_of_closed_loop_dynamics):    
        pt_trajectories, _ = generate_trajectories(initial_states, closed_loop_dynamics, dt, horizon)
        vals = np.zeros((n_trajs, horizon), dtype=config.dtype)
        for n in range(n_trajs):
            vals[n] = scalar_function(pt_trajectories[n].permute(1, 0)).detach().numpy().squeeze()
        x = np.arange(0, horizon, 1)
        for n in range(n_trajs):
            y = vals[n]
            ax.plot(x, y, '--', color=colors[i], linewidth=2.0, alpha=alpha)
    
    lines = [Line2D([0], [0], color=c, linewidth=3.0, linestyle='--', alpha=alpha) for c in colors]    
    labels = ['Untrained Controller', 'Trained Controller']
    legend = ax.legend(lines, labels, fontsize=20)
    legend.get_frame().set_alpha(0.5)

    ax.set_facecolor((0.0, 76/255, 153/255))
    ax.set_xlabel(plot_labels[0])
    ax.set_ylabel(plot_labels[1])
    ax.set_xlabel(plot_labels[0], fontsize=20)
    ax.set_ylabel(plot_labels[1], fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=15, grid_linewidth=15)
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    if full_path is not None:
        plt.savefig(full_path, dpi=config.dpi)

    
def plot_levelsets_as_binary_maps(gridsize, plot_limits, lyapunov, c_values, 
                    plot_labels=[None, None], full_path=None):

    """ Take a function and plot its levelsets as binary maps

    Parameter
    -------------
    gridsize : Tuple, Size of the rectangular grid on which the levelsets are plotted
    lyapunov: A Lyapunov class instance, It contains its values as a property
    c_values: Iterable, The values of the function corresponding to the requested levelsets
    plot_labels: [label of x axis, label of y axis]
    
    """

    dir_path = os.path.dirname(full_path)
    filename = os.path.basename(full_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    fig = plt.figure(figsize=(10, 10), dpi=config.dpi, frameon=False)
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    color=[None]
    color[0] = (0, 158/255, 115/255)       # ROA - bluish-green
    nrows, ncolumns = get_number_of_rows_and_columns(len(c_values))
    for i, c in enumerate(c_values):
        z = (lyapunov.values.detach().numpy() < c).reshape(gridsize)
        alpha = 1
        ax = plt.subplot(nrows, ncolumns, i+1)
        ax.set_title("c={:10.5}".format(c))
        ax.contour(z.T, origin='lower', extent=plot_limits.ravel(), colors=(color[0],), linewidths=1)
        ax.imshow(z.T, origin='lower', extent=plot_limits.ravel(), cmap=binary_cmap(color[0]), alpha=alpha)
        ax.tick_params(axis='both', which='major', labelsize=10, grid_linewidth=20)
        plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
        if full_path is not None:
            plt.savefig(full_path, dpi=config.dpi)


def plot_2d_streamplot(interval, df, density=1, margin=10, res=30, plot_labels=[None, None], plot_ticks=[None, None], full_path=None):
    """
    Takes the 2D df (rhs of ODE) and plots its stream plot in the specified interval.
    interval : [[xmin, xmax], [ymin, ymax]]
    df: takes (x, y) and outputs (xdot, ydot)
    density:density of the streamplot
    margin: margin of the plot in addition to the xmin/max and ymin/ymax
    res: determines the density of the flow
    alpha: opacity of the flow plot
    full_path: the path to solve the plot
    """
    
    # plot levelsets
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    colors = [(200/255, 200/255, 0), (158/255, 0, 115/255)]
    xmin, xmax = interval[0]
    ymin, ymax = interval[1]
    xmin -= margin
    ymin -= margin
    xmax += margin
    ymax += margin
    X, Y = np.meshgrid(np.linspace(xmin, xmax, res), np.linspace(ymin, ymax, res))
    u, v = np.zeros_like(X), np.zeros_like(Y)
    NI, NJ = X.shape
    for i in range(NI):
        for j in range(NJ):
            x, y = X[i, j], Y[i, j]
            dx, dy = df(x, y)
            u[i,j] = dx
            v[i,j] = dy
            
    fig, ax = plt.subplots(figsize=(10,10), dpi=config.dpi, frameon=False)
    ax.streamplot(X, Y, u, v, density=density)
    ax.set_xlabel(plot_labels[0], fontsize=30)
    ax.set_ylabel(plot_labels[1], fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=20, grid_linewidth=20)
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    plt.axis('square')
    plt.axis([xmin, xmax, ymin, ymax])
    ax.xaxis.set_ticks(plot_ticks[0])
    ax.yaxis.set_ticks(plot_ticks[1])
    if full_path is not None:
        plt.savefig(full_path, dpi=config.dpi)