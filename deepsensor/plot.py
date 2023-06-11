import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def context_encoding(
    model,
    task,
    task_loader,
    batch_idx=0,
    context_set_idxs=None,
    land_idx=None,
    cbar=True,
    clim=None,
    cmap="viridis",
    verbose_titles=True,
    titles=None,
    size=3,
    return_axes=False,
):
    """Plot the encoding of a context set in a task

    Parameters
    ----------
    model : DeepSensor model
    task : Task
        Task containing context set to plot encoding of
    task_loader : deepsensor.data.loader.TaskLoader
        DataLoader used to load the data, containing context set metadata used for plotting
    batch_idx : int, optional
        Batch index in encoding to plot, by default 0
    context_set_idxs : list or int, optional
        Indices of context sets to plot, by default None (plots all context sets)
    land_idx : int, optional
        Index of the land mask in the encoding (used to overlay land contour on plots), by default None
    verbose_titles : bool, optional
        Whether to include verbose titles for the variable IDs in the context set (including
        the time index), by default True
    titles : list, optional
        List of titles to override for each subplot, by default None.
        If None, titles are generated from context set metadata
    size : int, optional
        Size of the figure in inches, by default 20
    return_axes : bool, optional
        Whether to return the axes of the figure, by default False
    """
    from .model.nps import compute_encoding_tensor

    encoding_tensor = compute_encoding_tensor(model, task)
    encoding_tensor = encoding_tensor[batch_idx]

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
        if verbose_titles:
            var_IDs = task_loader.context_var_IDs_and_delta_t[ctx_i]
        else:
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
                ax.set_title(f"Density {ctx_i}")
            elif var_i > 0:
                ax.set_title(f"{var_IDs[var_i - 1]}")
            if var_i == 0:
                ax.set_ylabel(f"Context set {ctx_i}")
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
    if not return_axes:
        return fig
    elif return_axes:
        return fig, axes


def offgrid_context(
    axes,
    task,
    data_processor=None,
    task_loader=None,
    plot_target=False,
    add_legend=True,
    **scatter_kwargs,
):
    """Plot the off-grid context points on `axes`

    Uses `data_processor` to unnormalise the context coordinates if provided.
    """
    markers = "ovs^D"
    colors = "kbrgy"

    if type(axes) is np.ndarray:
        axes = axes.ravel()
    elif not isinstance(axes, (list, tuple)):
        axes = [axes]

    if plot_target:
        X = [*task["X_c"], *task["X_t"]]
    else:
        X = task["X_c"]

    for set_i, X in enumerate(X):
        if isinstance(X, tuple):
            continue  # Don't plot gridded context data locations
        if X.ndim == 3:
            X = X[0]  # select first batch

        if data_processor is not None:
            x1, x2 = data_processor.map_x1_and_x2(X[0], X[1], unnorm=True)
            X = np.stack([x1, x2], axis=0)

        X = X[::-1]  # flip 2D coords for Cartesian fmt

        label = ""
        if plot_target and set_i < len(task["X_c"]):
            label += f"Context set {set_i} "
            if task_loader is not None:
                label += f"({task_loader.context_var_IDs[set_i]})"
        elif plot_target and set_i >= len(task["X_c"]):
            label += f"Target set {set_i - len(task['X_c'])} "
            if task_loader is not None:
                label += f"({task_loader.target_var_IDs[set_i - len(task['X_c'])]})"

        for ax in axes:
            ax.scatter(
                *X,
                marker=markers[set_i],
                color=colors[set_i],
                **scatter_kwargs,
                facecolors=None if markers[set_i] == "x" else "none",
                label=label,
            )

    if add_legend:
        axes[0].legend(loc="best")
