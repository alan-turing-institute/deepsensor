import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import cartopy

from deepsensor.model.nps import compute_encoding_tensor


def plot_context_encoding(
    model,
    task,
    task_loader,
    context_set_idxs=None,
    land_idx=None,
    cbar=True,
    clim=None,
    cmap="viridis",
    titles=None,
    size=3,
):
    """Plot the encoding of a context set in a task

    Parameters
    ----------
    model : DeepSensor model
    task : dict
        Task containing context set to plot encoding of
    task_loader : deepsensor.data.loader.TaskLoader
        DataLoader used to load the data, containing context set metadata used for plotting
    context_set_idxs : list or int, optional
        Indices of context sets to plot, by default None (plots all context sets)
    land_idx : int, optional
        Index of the land mask in the encoding (used to overlay land contour on plots), by default None
    titles : list, optional
        List of titles for each subplot, by default None
    size : int, optional
        Size of the figure in inches, by default 20
    """
    encoding_tensor = compute_encoding_tensor(model, task)

    if isinstance(context_set_idxs, int):
        context_set_idxs = [context_set_idxs]
    if context_set_idxs is None:
        context_set_idxs = np.array(range(len(task_loader.context)))

    context_var_ID_set_sizes = [
        ndim + 1 for ndim in np.array(task_loader.context_dims)[context_set_idxs]
    ]  # Add density channel to each set size
    max_context_set_size = max(context_var_ID_set_sizes)
    ncols = max_context_set_size
    nrows = len(context_set_idxs)

    figsize = (ncols * size, nrows * size)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    if nrows == 1:
        axes = axes[np.newaxis]

    channel_i = 0
    for ctx_i in context_set_idxs:
        var_IDs = task_loader.context_var_IDs[ctx_i]
        size = task_loader.context_dims[ctx_i] + 1  # Add density channel
        for var_i in range(size):
            ax = axes[ctx_i, var_i]
            # Need `origin="lower"` because encoding has `x1` increasing from top to bottom,
            # whereas in visualisations we want `x1` increasing from bottom to top.
            im = ax.imshow(
                encoding_tensor[channel_i], origin="lower", clim=clim, cmap=cmap
            )
            if titles is not None:
                ax.set_title(titles[channel_i])
            elif var_i == 0:
                ax.set_title(f"Density {ctx_i + 1}")
            elif var_i > 0:
                ax.set_title(f"{var_IDs[var_i - 1]}")
            if var_i == 0:
                ax.set_ylabel(f"Context set {ctx_i + 1}")
            if cbar:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax)
            ax.patch.set_edgecolor("black")
            ax.patch.set_linewidth(1)
            if land_idx is not None:
                ax.contour(
                    encoding_tensor[land_idx], colors="k", levels=[0.5], origin="lower"
                )
            ax.tick_params(
                which="both",
                bottom=False,
                left=False,
                labelbottom=False,
                labelleft=False,
            )
            channel_i += 1
        for var_i in range(size, ncols):
            # Hide unused axes
            ax = axes[ctx_i, var_i]
            ax.axis("off")

    plt.tight_layout()
    return fig


def plot_offgrid_context(axes, task, data_processor=None, **scatter_kwargs):
    """Plot the off-grid context points on `axes`

    Uses `data_processor` to unnormalise the context coordinates if provided.
    """
    markers = "ovs^D"
    colors = "kbrgy"

    if type(axes) is np.ndarray:
        axes = axes.ravel()
    elif isinstance(axes, (mpl.axes.Axes, cartopy.mpl.geoaxes.GeoAxesSubplot)):
        axes = [axes]

    for context_i, X_c in enumerate(task["X_c"]):
        if isinstance(X_c, tuple):
            continue  # Don't plot gridded context data locations
        if X_c.ndim == 3:
            X_c = X_c[0]  # select first batch

        if data_processor is not None:
            X_c = data_processor.map_coord_array(X_c, unnorm=True)

        X_c = X_c[::-1]  # flip 2D coords for Cartesian fmt

        for ax in axes:
            ax.scatter(
                *X_c,
                marker=markers[context_i],
                color=colors[context_i],
                **scatter_kwargs,
                facecolors=None if markers[context_i] == "x" else "none",
            )
