import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as mpatches

import lab as B

from typing import Optional, Union, List, Tuple

from deepsensor.data.task import Task, flatten_X
from deepsensor.data.loader import TaskLoader
from deepsensor.data.processor import DataProcessor
from deepsensor.model.pred import Prediction
from pandas import DataFrame
from matplotlib.colors import Colormap
from matplotlib.axes import Axes


def task(
    task: Task,
    task_loader: TaskLoader,
    figsize=3,
    markersize=None,
    equal_aspect=False,
    plot_ticks=False,
    extent=None,
) -> plt.Figure:
    """Plot the context and target sets of a task.

    Args:
        task (:class:`~.data.task.Task`):
            Task to plot.
        task_loader (:class:`~.data.loader.TaskLoader`):
            Task loader used to load ``task``, containing variable IDs used for
            plotting.
        figsize (int, optional):
            Figure size in inches, by default 3.
        markersize (int, optional):
            Marker size (in units of points squared), by default None. If None,
            the marker size is set to ``(2**2) * figsize / 3``.
        equal_aspect (bool, optional):
            Whether to set the aspect ratio of the plots to be equal, by
            default False.
        plot_ticks (bool, optional):
            Whether to plot the coordinate ticks on the axes, by default False.
        extent (Tuple[int, int, int, int], optional):
            Extent of the plot in format (x2_min, x2_max, x1_min, x1_max).
            Defaults to None (uses the smallest extent that contains all data points
            across all context and target sets).

    Returns:
        :class:`matplotlib:matplotlib.figure.Figure`:
    """
    if markersize is None:
        markersize = (2**2) * figsize / 3

    # Scale font size with figure size
    fontsize = 10 * figsize / 3
    params = {
        "axes.labelsize": fontsize,
        "axes.titlesize": fontsize,
        "font.size": fontsize,
        "figure.titlesize": fontsize,
        "legend.fontsize": fontsize,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
    }

    var_IDs = task_loader.context_var_IDs + task_loader.target_var_IDs
    Y_c = task["Y_c"]
    X_c = task["X_c"]
    if task["Y_t"] is not None:
        Y_t = task["Y_t"]
        X_t = task["X_t"]
    else:
        Y_t = []
        X_t = []
    n_context = len(Y_c)
    n_target = len(Y_t)
    if "Y_t_aux" in task and task["Y_t_aux"] is not None:
        # Assumes only 1 target set
        X_t = X_t + [task["X_t"][-1]]
        Y_t = Y_t + [task["Y_t_aux"]]
        var_IDs = var_IDs + (task_loader.aux_at_target_var_IDs,)
        ncols = n_context + n_target + 1
    else:
        ncols = n_context + n_target
    nrows = max([Y.shape[0] for Y in Y_c + Y_t])

    if extent is None:
        x1_min = np.min([np.min(X[0]) for X in X_c + X_t])
        x1_max = np.max([np.max(X[0]) for X in X_c + X_t])
        x2_min = np.min([np.min(X[1]) for X in X_c + X_t])
        x2_max = np.max([np.max(X[1]) for X in X_c + X_t])
        extent = (x2_min, x2_max, x1_min, x1_max)

    with plt.rc_context(params):
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(ncols * figsize, nrows * figsize),
        )
        if nrows == 1:
            axes = axes[np.newaxis]
        if ncols == 1:
            axes = axes[:, np.newaxis]
        # j = loop index over columns/context sets
        # i = loop index over rows/variables within context sets
        for j, (X, Y) in enumerate(zip(X_c + X_t, Y_c + Y_t)):
            for i in range(Y.shape[0]):
                if i == 0:
                    if j < n_context:
                        axes[0, j].set_title(f"Context set {j}")
                    elif j < n_context + n_target:
                        axes[0, j].set_title(f"Target set {j - n_context}")
                    else:
                        axes[0, j].set_title(f"Auxiliary at targets")
                if isinstance(X, tuple):
                    X = flatten_X(X)
                    Y = Y.reshape(Y.shape[0], -1)
                axes[i, j].scatter(X[1, :], X[0, :], c=Y[i], s=markersize, marker=".")
                if equal_aspect:
                    # Don't warp aspect ratio
                    axes[i, j].set_aspect("equal")
                if not plot_ticks:
                    axes[i, j].set_xticks([])
                    axes[i, j].set_yticks([])
                axes[i, j].set_ylabel(var_IDs[j][i])

                axes[i, j].set_xlim(extent[0], extent[1])
                axes[i, j].set_ylim(extent[2], extent[3])

                # Add colorbar with same height as axis
                divider = make_axes_locatable(axes[i, j])
                box = axes[i, j].get_position()
                ratio = 0.3
                pad = 0.1
                width = box.width * ratio
                cax = divider.append_axes("right", size=width, pad=pad)
                fig.colorbar(axes[i, j].collections[0], cax=cax)

            for i in range(Y.shape[0], nrows):
                axes[i, j].axis("off")

        plt.tight_layout()

    return fig


def context_encoding(
    model,
    task: Task,
    task_loader: TaskLoader,
    batch_idx: int = 0,
    context_set_idxs: Optional[Union[List[int], int]] = None,
    land_idx: Optional[int] = None,
    cbar: bool = True,
    clim: Optional[Tuple] = None,
    cmap: Union[str, Colormap] = "viridis",
    verbose_titles: bool = True,
    titles: Optional[dict] = None,
    size: int = 3,
    return_axes: bool = False,
):
    """Plot the ``ConvNP`` SetConv encoding of a context set in a task.

    Args:
        model (:class:`~.model.convnp.ConvNP`):
            ConvNP model.
        task (:class:`~.data.task.Task`):
            Task containing context set to plot encoding of ...
        task_loader (:class:`~.data.loader.TaskLoader`):
            DataLoader used to load the data, containing context set metadata
            used for plotting.
        batch_idx (int, optional):
            Batch index in encoding to plot, by default 0.
        context_set_idxs (List[int] | int, optional):
            Indices of context sets to plot, by default None (plots all context
            sets).
        land_idx (int, optional):
            Index of the land mask in the encoding (used to overlay land
            contour on plots), by default None.
        cbar (bool, optional):
            Whether to add a colorbar to the plots, by default True.
        clim (tuple, optional):
            Colorbar limits, by default None.
        cmap (str | matplotlib.colors.Colormap, optional):
            Color map to use for the plots, by default "viridis".
        verbose_titles (bool, optional):
            Whether to include verbose titles for the variable IDs in the
            context set (including the time index), by default True.
        titles (dict, optional):
            Dict of titles to override for each subplot, by default None. If
            None, titles are generated from context set metadata.
        size (int, optional):
            Size of the figure in inches, by default 3.
        return_axes (bool, optional):
            Whether to return the axes of the figure, by default False.

    Returns:
        :obj:`matplotlib.figure.Figure` | Tuple[:obj:`matplotlib.figure.Figure`, :obj:`matplotlib.pyplot.Axes`]:
            Either a figure containing the context set encoding plots, or a
            tuple containing the :obj:`figure <matplotlib.figure.Figure>` and
            the :obj:`axes <matplotlib.axes.Axes>` of the figure (if
            ``return_axes`` was set to ``True``).
    """
    from .model.nps import compute_encoding_tensor

    encoding_tensor = compute_encoding_tensor(model, task)
    encoding_tensor = encoding_tensor[batch_idx]

    if isinstance(context_set_idxs, int):
        context_set_idxs = [context_set_idxs]
    if context_set_idxs is None:
        context_set_idxs = np.array(range(len(task_loader.context_dims)))

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

    ctx_channel_idxs = np.cumsum(np.array(task_loader.context_dims) + 1)

    for row_i, ctx_i in enumerate(context_set_idxs):
        channel_i = (
            ctx_channel_idxs[ctx_i - 1] if ctx_i > 0 else 0
        )  # Starting channel index
        if verbose_titles:
            var_IDs = task_loader.context_var_IDs_and_delta_t[ctx_i]
        else:
            var_IDs = task_loader.context_var_IDs[ctx_i]

        ncols_row_i = task_loader.context_dims[ctx_i] + 1  # Add density channel
        for col_i in range(ncols_row_i):
            ax = axes[row_i, col_i]
            # Need `origin="lower"` because encoding has `x1` increasing from top to bottom,
            # whereas in visualisations we want `x1` increasing from bottom to top.

            im = ax.imshow(
                encoding_tensor[channel_i],
                origin="lower",
                clim=clim,
                cmap=cmap,
            )
            if titles is not None:
                ax.set_title(titles[channel_i])
            elif col_i == 0:
                ax.set_title(f"Density {ctx_i}")
            elif col_i > 0:
                ax.set_title(f"{var_IDs[col_i - 1]}")
            if col_i == 0:
                ax.set_ylabel(f"Context set {ctx_i}")
            if cbar:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax)
            ax.patch.set_edgecolor("black")
            ax.patch.set_linewidth(1)
            if land_idx is not None:
                ax.contour(
                    encoding_tensor[land_idx],
                    colors="k",
                    levels=[0.5],
                    origin="lower",
                )
            ax.tick_params(
                which="both",
                bottom=False,
                left=False,
                labelbottom=False,
                labelleft=False,
            )
            channel_i += 1
        for col_i in range(ncols_row_i, ncols):
            # Hide unused axes
            ax = axes[ctx_i, col_i]
            ax.axis("off")

    plt.tight_layout()
    if not return_axes:
        return fig
    elif return_axes:
        return fig, axes


def offgrid_context(
    axes: Union[np.ndarray, List[plt.Axes], Tuple[plt.Axes]],
    task: Task,
    data_processor: Optional[DataProcessor] = None,
    task_loader: Optional[TaskLoader] = None,
    plot_target: bool = False,
    add_legend: bool = True,
    context_set_idxs: Optional[Union[List[int], int]] = None,
    markers: Optional[str] = None,
    colors: Optional[str] = None,
    **scatter_kwargs,
) -> None:
    """Plot the off-grid context points on ``axes``.

    Uses a provided :class:`~.data.processor.DataProcessor` to unnormalise the
    context coordinates if provided.

    Args:
        axes (:class:`numpy:numpy.ndarray` | List[:class:`matplotlib:matplotlib.axes.Axes`] | Tuple[:class:`matplotlib:matplotlib.axes.Axes`]):
            Axes to plot on.
        task (:class:`~.data.task.Task`):
            Task containing the context set to plot.
        data_processor (:class:`~.data.processor.DataProcessor`, optional):
            Data processor used to unnormalise the context set, by default
            None.
        task_loader (:class:`~.data.loader.TaskLoader`, optional):
            Task loader used to load the data, containing context set metadata
            used for plotting, by default None.
        plot_target (bool, optional):
            Whether to plot the target set, by default False.
        add_legend (bool, optional):
            Whether to add a legend to the plot, by default True.
        context_set_idxs (List[int] | int, optional):
            Indices of context sets to plot, by default None (plots all context
            sets).
        markers (str, optional):
            Marker styles to use for each context set, by default None.
        colors (str, optional):
            Colors to use for each context set, by default None.
        scatter_kwargs:
            Additional keyword arguments to pass to the scatter plot.

    Returns:
        None
    """
    if markers is None:
        # all matplotlib markers
        markers = "ovs^Dxv<>1234spP*hHd|_"
    if colors is None:
        # all one-letter matplotlib colors
        colors = "kbrgy" * 10

    if isinstance(context_set_idxs, int):
        context_set_idxs = [context_set_idxs]

    if type(axes) is np.ndarray:
        axes = axes.ravel()
    elif not isinstance(axes, (list, tuple)):
        axes = [axes]

    if plot_target:
        X = [*task["X_c"], *task["X_t"]]
    else:
        X = task["X_c"]

    for set_i, X in enumerate(X):
        if context_set_idxs is not None and set_i not in context_set_idxs:
            continue

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


def offgrid_context_observations(
    axes: Union[np.ndarray, List[plt.Axes], Tuple[plt.Axes]],
    task: Task,
    data_processor: DataProcessor,
    task_loader: TaskLoader,
    context_set_idx: int,
    format_str: Optional[str] = None,
    extent: Optional[Tuple[int, int, int, int]] = None,
    color: str = "black",
) -> None:
    """Plot unnormalised context observation values.

    Args:
        axes (:class:`numpy:numpy.ndarray` | List[:class:`matplotlib:matplotlib.axes.Axes`] | Tuple[:class:`matplotlib:matplotlib.axes.Axes`]):
            Axes to plot on.
        task (:class:`~.data.task.Task`):
            Task containing the context set to plot.
        data_processor (:class:`~.data.processor.DataProcessor`):
            Data processor used to unnormalise the context set.
        task_loader (:class:`~.data.loader.TaskLoader`):
            Task loader used to load the data, containing context set metadata
            used for plotting.
        context_set_idx (int):
            Index of the context set to plot.
        format_str (str, optional):
            Format string for the context observation values. By default
            ``"{:.2f}"``.
        extent (Tuple[int, int, int, int], optional):
            Extent of the plot, by default None.
        color (str, optional):
            Color of the text, by default "black".

    Returns:
        None.

    Raises:
        AssertionError:
            If the context set is gridded.
        AssertionError:
            If the context set is not 1D.
        AssertionError:
            If the task's "Y_c" value for the context set ID is not 2D.
        AssertionError:
            If the task's "Y_c" value for the context set ID does not have
            exactly one variable.
    """
    if type(axes) is np.ndarray:
        axes = axes.ravel()
    elif not isinstance(axes, (list, tuple)):
        axes = [axes]

    if format_str is None:
        format_str = "{:.2f}"

    var_ID = task_loader.context_var_IDs[
        context_set_idx
    ]  # Tuple of variable IDs for the context set
    assert (
        len(var_ID) == 1
    ), "Plotting context observations only supported for single-variable (1D) context sets"
    var_ID = var_ID[0]

    X_c = task["X_c"][context_set_idx]
    assert not isinstance(
        X_c, tuple
    ), f"The context set must not be gridded but is of type {type(X_c)} for context set at index {context_set_idx}"
    X_c = data_processor.map_coord_array(X_c, unnorm=True)

    Y_c = task["Y_c"][context_set_idx]
    assert Y_c.ndim == 2
    assert Y_c.shape[0] == 1
    Y_c = data_processor.map_array(Y_c, var_ID, unnorm=True).ravel()

    for x_c, y_c in zip(X_c.T, Y_c):
        if extent is not None:
            if not (
                extent[0] <= x_c[0] <= extent[1] and extent[2] <= x_c[1] <= extent[3]
            ):
                continue
        for ax in axes:
            ax.text(*x_c[::-1], format_str.format(float(y_c)), color=color)


def receptive_field(
    receptive_field,
    data_processor: DataProcessor,
    crs,
    extent: Union[str, Tuple[float, float, float, float]] = "global",
) -> plt.Figure:  # pragma: no cover
    """...

    Args:
        receptive_field (...):
            Receptive field to plot.
        data_processor (:class:`~.data.processor.DataProcessor`):
            Data processor used to unnormalise the context set.
        crs (cartopy CRS):
            Coordinate reference system for the plots.
        extent (str | Tuple[float, float, float, float], optional):
            Extent of the plot, in format (x2_min, x2_max, x1_min, x1_max), e.g. in
            lat-lon format (lon_min, lon_max, lat_min, lat_max). By default "global".

    Returns:
        None.
    """
    fig, ax = plt.subplots(subplot_kw=dict(projection=crs))

    if isinstance(extent, str):
        extent = extent_str_to_tuple(extent)
    else:
        extent = tuple([float(x) for x in extent])
    x2_min, x2_max, x1_min, x1_max = extent
    ax.set_extent(extent, crs=crs)

    x11, x12 = data_processor.config["coords"]["x1"]["map"]
    x21, x22 = data_processor.config["coords"]["x2"]["map"]

    x1_rf_raw = receptive_field * (x12 - x11)
    x2_rf_raw = receptive_field * (x22 - x21)

    x1_midpoint_raw = (x1_max + x1_min) / 2
    x2_midpoint_raw = (x2_max + x2_min) / 2

    # Compute bottom left corner of receptive field
    x1_corner = x1_midpoint_raw - x1_rf_raw / 2
    x2_corner = x2_midpoint_raw - x2_rf_raw / 2

    ax.add_patch(
        mpatches.Rectangle(
            xy=[x2_corner, x1_corner],  # Cartesian fmt: x2, x1
            width=x2_rf_raw,
            height=x1_rf_raw,
            facecolor="black",
            alpha=0.3,
            transform=crs,
        )
    )
    ax.coastlines()
    ax.gridlines(draw_labels=True, alpha=0.2)

    x1_name = data_processor.config["coords"]["x1"]["name"]
    x2_name = data_processor.config["coords"]["x2"]["name"]
    ax.set_title(
        f"Receptive field in raw coords: {x1_name}={x1_rf_raw:.2f}, "
        f"{x2_name}={x2_rf_raw:.2f}"
    )

    return fig


def feature_maps(
    model,
    task: Task,
    n_features_per_layer: int = 1,
    seed: Optional[int] = None,
    figsize: int = 3,
    add_colorbar: bool = False,
    cmap: Union[str, Colormap] = "Greys",
) -> plt.Figure:
    """Plot the feature maps of a ``ConvNP`` model's decoder layers after a
    forward pass with a ``Task``.

    Args:
        model (:class:`~.model.model.convnp.ConvNP`):
            ...
        task (:class:`~.data.task.Task`):
            ...
        n_features_per_layer (int, optional):
            ..., by default 1.
        seed (int, optional):
            ..., by default None.
        figsize (int, optional):
            ..., by default 3.
        add_colorbar (bool, optional):
            ..., by default False.
        cmap (str | matplotlib.colors.Colormap, optional):
            ..., by default "Greys".

    Returns:
        matplotlib.figure.Figure:
            A figure containing the feature maps.

    Raises:
        ValueError:
            If the backend is not recognised.
    """
    from .model.nps import compute_encoding_tensor

    import deepsensor

    # Hacky way to load the correct __init__.py to get `convert_to_tensor` method
    if deepsensor.backend.str == "tf":
        import deepsensor.tensorflow as deepsensor
    elif deepsensor.backend.str == "torch":
        import deepsensor.torch as deepsensor
    else:
        raise ValueError(f"Unknown backend: {deepsensor.backend.str}")

    unet = model.model.decoder[0]

    # Produce encoding
    x = deepsensor.convert_to_tensor(compute_encoding_tensor(model, task))

    # Manually construct the U-Net forward pass from
    # `neuralprocesses.construct_convgnp` to get the feature maps
    def unet_forward(unet, x):
        feature_maps = []

        h = unet.activations[0](unet.before_turn_layers[0](x))
        hs = [h]
        feature_map = B.to_numpy(h)
        feature_maps.append(feature_map)
        for layer, activation in zip(
            unet.before_turn_layers[1:],
            unet.activations[1:],
        ):
            h = activation(layer(hs[-1]))
            hs.append(h)
            feature_map = B.to_numpy(h)
            feature_maps.append(feature_map)

        # Now make the turn!

        h = unet.activations[-1](unet.after_turn_layers[-1](hs[-1]))
        feature_map = B.to_numpy(h)
        feature_maps.append(feature_map)
        for h_prev, layer, activation in zip(
            reversed(hs[:-1]),
            reversed(unet.after_turn_layers[:-1]),
            reversed(unet.activations[:-1]),
        ):
            h = activation(layer(B.concat(h_prev, h, axis=1)))
            feature_map = B.to_numpy(h)
            feature_maps.append(feature_map)

        h = unet.final_linear(h)
        feature_map = B.to_numpy(h)
        feature_maps.append(feature_map)

        return feature_maps

    feature_maps = unet_forward(unet, x)

    figs = []
    rng = np.random.default_rng(seed)
    for layer_i, feature_map in enumerate(feature_maps):
        n_features = feature_map.shape[1]
        n_features_to_plot = min(n_features_per_layer, n_features)
        feature_idxs = rng.choice(n_features, n_features_to_plot, replace=False)

        fig, axes = plt.subplots(
            nrows=1,
            ncols=n_features_to_plot,
            figsize=(figsize * n_features_to_plot, figsize),
        )
        if n_features_to_plot == 1:
            axes = [axes]
        for f_i, ax in zip(feature_idxs, axes):
            fm = feature_map[0, f_i]
            im = ax.imshow(fm, origin="lower", cmap=cmap)
            ax.set_title(f"Feature {f_i}", fontsize=figsize * 15 / 4)
            ax.tick_params(
                which="both",
                bottom=False,
                left=False,
                labelbottom=False,
                labelleft=False,
            )
            if add_colorbar:
                cbar = ax.figure.colorbar(im, ax=ax, format="%.2f")

        fig.suptitle(
            f"Layer {layer_i} feature map. Shape: {feature_map.shape}. Min={np.min(feature_map):.2f}, Max={np.max(feature_map):.2f}.",
            fontsize=figsize * 15 / 4,
        )
        plt.tight_layout()
        plt.subplots_adjust(top=0.75)
        figs.append(fig)

    return figs


def placements(
    task: Task,
    X_new_df: DataFrame,
    data_processor: DataProcessor,
    crs,
    extent: Optional[Union[Tuple[int, int, int, int], str]] = None,
    figsize: int = 3,
    **scatter_kwargs,
) -> plt.Figure:  # pragma: no cover
    """...

    Args:
        task (:class:`~.data.task.Task`):
            Task containing the context set used to compute the acquisition
            function.
        X_new_df (:class:`pandas.DataFrame`):
            Dataframe containing the placement locations.
        data_processor (:class:`~.data.processor.DataProcessor`):
            Data processor used to unnormalise the context set and placement
            locations.
        crs (cartopy CRS):
            Coordinate reference system for the plots.
        extent (Tuple[int, int, int, int] | str, optional):
            Extent of the plots, by default None.
        figsize (int, optional):
            Figure size in inches, by default 3.

    Returns:
        :class:`matplotlib:matplotlib.figure.Figure`
            A figure containing the placement plots.
    """
    fig, ax = plt.subplots(subplot_kw={"projection": crs}, figsize=(figsize, figsize))
    ax.scatter(*X_new_df.values.T[::-1], c="r", linewidths=0.5, **scatter_kwargs)
    offgrid_context(ax, task, data_processor, linewidths=0.5, **scatter_kwargs)

    ax.coastlines()
    if extent is None:
        pass
    elif extent == "global":
        ax.set_global()
    else:
        ax.set_extent(extent, crs=crs)

    return fig


def acquisition_fn(
    task: Task,
    acquisition_fn_ds: np.ndarray,
    X_new_df: DataFrame,
    data_processor: DataProcessor,
    crs,
    col_dim: str = "iteration",
    cmap: Union[str, Colormap] = "Greys_r",
    figsize: int = 3,
    add_colorbar: bool = True,
    max_ncol: int = 5,
) -> plt.Figure:  # pragma: no cover
    """Args:
        task (:class:`~.data.task.Task`):
            Task containing the context set used to compute the acquisition
            function.
        acquisition_fn_ds (:class:`numpy:numpy.ndarray`):
            Acquisition function dataset.
        X_new_df (:class:`pandas.DataFrame`):
            Dataframe containing the placement locations.
        data_processor (:class:`~.data.processor.DataProcessor`):
            Data processor used to unnormalise the context set and placement
            locations.
        crs (cartopy CRS):
            Coordinate reference system for the plots.
        col_dim (str, optional):
            Column dimension to plot over, by default "iteration".
        cmap (str | matplotlib.colors.Colormap, optional):
            Color map to use for the plots, by default "Greys_r".
        figsize (int, optional):
            Figure size in inches, by default 3.
        add_colorbar (bool, optional):
            Whether to add a colorbar to the plots, by default True.
        max_ncol (int, optional):
            Maximum number of columns to use for the plots, by default 5.

    Returns:
        matplotlib.pyplot.Figure
            A figure containing the acquisition function plots.

    Raises:
        ValueError:
            If a column dimension is encountered that is not one of
            ``["time", "sample"]``.
        AssertionError:
            If the number of columns in the acquisition function dataset is
            greater than ``max_ncol``.
    """
    # Remove spatial dims using data_processor.raw_spatial_coords_names
    plot_dims = [col_dim, *data_processor.raw_spatial_coord_names]
    non_plot_dims = [dim for dim in acquisition_fn_ds.dims if dim not in plot_dims]
    valid_avg_dims = ["time", "sample"]
    for dim in non_plot_dims:
        if dim not in valid_avg_dims:
            raise ValueError(
                f"Cannot average over dim {dim} for plotting. Must be one of "
                f"{valid_avg_dims}. Select a single value for {dim} using "
                f"`acquisition_fn_ds.sel({dim}=...)`."
            )
    if len(non_plot_dims) > 0:
        # Average over non-plot dims
        print(
            "Averaging acquisition function over dims for plotting: " f"{non_plot_dims}"
        )
        acquisition_fn_ds = acquisition_fn_ds.mean(dim=non_plot_dims)

    col_vals = acquisition_fn_ds[col_dim].values
    if col_vals.size == 1:
        n_col_vals = 1
    else:
        n_col_vals = len(col_vals)
    ncols = np.min([max_ncol, n_col_vals])

    if n_col_vals > ncols:
        nrows = int(np.ceil(n_col_vals / ncols))
    else:
        nrows = 1

    fig, axes = plt.subplots(
        subplot_kw={"projection": crs},
        ncols=ncols,
        nrows=nrows,
        figsize=(figsize * ncols, figsize * nrows),
    )
    if nrows == 1 and ncols == 1:
        axes = [axes]
    else:
        axes = axes.ravel()
    if add_colorbar:
        min, max = acquisition_fn_ds.min(), acquisition_fn_ds.max()
    else:
        # Use different colour scales for each plot
        min, max = None, None
    for i, col_val in enumerate(col_vals):
        ax = axes[i]
        if i == len(col_vals) - 1:
            final_axis = True
        else:
            final_axis = False
        acquisition_fn_ds.sel(**{col_dim: col_val}).plot(
            ax=ax, cmap=cmap, vmin=min, vmax=max, add_colorbar=False
        )
        if add_colorbar and final_axis:
            im = ax.get_children()[0]
            label = acquisition_fn_ds.name
            cax = plt.axes([0.93, 0.035, 0.02, 0.91])  # add a small custom axis
            cbar = plt.colorbar(
                im, cax=cax, label=label
            )  # specify axis for colorbar to occupy with cax
        ax.set_title(f"{col_dim}={col_val}")
        ax.coastlines()
        if col_dim == "iteration":
            X_new_df_plot = X_new_df.loc[slice(0, col_val)].values.T[::-1]
        else:
            # Assumed plotting single iteration
            iter = acquisition_fn_ds.iteration.values
            assert iter.size == 1, "Expected single iteration"
            X_new_df_plot = X_new_df.loc[slice(0, iter.item())].values.T[::-1]
        ax.scatter(
            *X_new_df_plot,
            c="r",
            linewidths=0.5,
        )

    offgrid_context(axes, task, data_processor, linewidths=0.5)

    # Remove any unused axes
    for ax in axes[len(col_vals) :]:
        ax.remove()

    return fig


def prediction(
    pred: Prediction,
    date: Optional[Union[str, pd.Timestamp]] = None,
    data_processor: Optional[DataProcessor] = None,
    task_loader: Optional[TaskLoader] = None,
    task: Optional[Task] = None,
    prediction_parameters: Union[List[str], str] = "all",
    crs=None,
    colorbar: bool = True,
    cmap: str = "viridis",
    size: int = 5,
    extent: Optional[Union[Tuple[float, float, float, float], str]] = None,
) -> plt.Figure:  # pragma: no cover
    """Plot the mean and standard deviation of a prediction.

    Args:
        pred (:class:`~.model.prediction.Prediction`):
            Prediction to plot.
        date (str | :class:`pandas:pandas.Timestamp`):
            Date of the prediction.
        data_processor (:class:`~.data.processor.DataProcessor`):
            Data processor used to unnormalise the context set.
        task_loader (:class:`~.data.loader.TaskLoader`):
            Task loader used to load the data, containing context set metadata
            used for plotting.
        task (:class:`~.data.task.Task`, optional):
            Task containing the context data to overlay.
        prediction_parameters (List[str] | str, optional):
            Prediction parameters to plot, by default "all".
        crs (cartopy CRS, optional):
            Coordinate reference system for the plots, by default None.
        colorbar (bool, optional):
            Whether to add a colorbar to the plots, by default True.
        cmap (str):
            Colormap to use for the plots. By default "viridis".
        size (int, optional):
            Size of the figure in inches per axis, by default 5.
        extent: (tuple | str, optional):
            Tuple of (lon_min, lon_max, lat_min, lat_max) or string of region name.
            Options are: "global", "usa", "uk", "europe". Defaults to None (no
            setting of extent).
        c
    """
    if pred.mode == "off-grid":
        assert date is None, "Cannot pass a `date` for off-grid predictions"
        assert (
            data_processor is None
        ), "Cannot pass a `data_processor` for off-grid predictions"
        assert (
            task_loader is None
        ), "Cannot pass a `task_loader` for off-grid predictions"
        assert task is None, "Cannot pass a `task` for off-grid predictions"
        assert crs is None, "Cannot pass a `crs` for off-grid predictions"

    x1_name = pred.x1_name
    x2_name = pred.x2_name

    if prediction_parameters == "all":
        prediction_parameters = {
            var_ID: [param for param in pred[var_ID]] for var_ID in pred
        }
    else:
        prediction_parameters = {var_ID: prediction_parameters for var_ID in pred}

    n_vars = len(pred.target_var_IDs)
    n_params = max(len(params) for params in prediction_parameters.values())

    if isinstance(extent, str):
        extent = extent_str_to_tuple(extent)
    elif isinstance(extent, tuple):
        extent = tuple([float(x) for x in extent])

    fig, axes = plt.subplots(
        n_vars,
        n_params,
        figsize=(size * n_params, size * n_vars),
        subplot_kw=dict(projection=crs),
    )
    axes = np.array(axes)
    if n_vars == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_params == 1:
        axes = np.expand_dims(axes, axis=1)
    for row_i, var_ID in enumerate(pred.target_var_IDs):
        for col_i, param in enumerate(prediction_parameters[var_ID]):
            ax = axes[row_i, col_i]

            if pred.mode == "on-grid":
                if param == "std":
                    vmin = 0
                else:
                    vmin = None
                pred[var_ID][param].sel(time=date).plot(
                    ax=ax,
                    cmap=cmap,
                    vmin=vmin,
                    add_colorbar=False,
                    center=False,
                )
                # ax.set_aspect("auto")
                if colorbar:
                    im = ax.get_children()[0]
                    # add axis to right
                    cax = fig.add_axes(
                        [
                            ax.get_position().x1 + 0.01,
                            ax.get_position().y0,
                            0.02,
                            ax.get_position().height,
                        ]
                    )
                    cbar = plt.colorbar(
                        im, cax=cax
                    )  # specify axis for colorbar to occupy with cax
                if task is not None:
                    offgrid_context(
                        ax,
                        task,
                        data_processor,
                        task_loader,
                        linewidths=0.5,
                        add_legend=False,
                    )
                if crs is not None:
                    da = pred[var_ID][param]
                    ax.coastlines()
                    import cartopy.feature as cfeature

                    ax.add_feature(cfeature.BORDERS)
                    # ax.set_extent(
                    #     [da["lon"].min(), da["lon"].max(), da["lat"].min(), da["lat"].max()]
                    # )

            elif pred.mode == "off-grid":
                import seaborn as sns

                hue = (
                    pred[var_ID]
                    .reset_index()[[x1_name, x2_name]]
                    .apply(lambda row: f"({row[x1_name]}, {row[x2_name]})", axis=1)
                )
                hue.name = f"{x1_name}, {x2_name}"

                sns.lineplot(
                    data=pred[var_ID],
                    x="time",
                    y=param,
                    ax=ax,
                    hue=hue.values,
                )
                # set legend title
                ax.legend(title=hue.name, loc="best")

                # rotate date times
                ax.set_xticklabels(
                    ax.get_xticklabels(),
                    rotation=45,
                    horizontalalignment="right",
                )

            ax.set_title(f"{var_ID} {param}")

            if extent is not None:
                ax.set_extent(extent, crs=crs)

    plt.subplots_adjust(wspace=0.3)
    return fig


def extent_str_to_tuple(extent: str) -> Tuple[float, float, float, float]:
    """Convert extent string to (lon_min, lon_max, lat_min, lat_max) tuple.

    Args:
        extent: str
            String of region name. Options are: "global", "usa", "uk", "europe".

    Returns:
        tuple
            Tuple of (lon_min, lon_max, lat_min, lat_max).
    """
    if extent == "global":
        return (-180, 180, -90, 90)
    elif extent == "north_america":
        return (-160, -60, 15, 75)
    elif extent == "uk":
        return (-12, 3, 50, 60)
    elif extent == "europe":
        return (-15, 40, 35, 70)
    elif extent == "germany":
        return (5, 15, 47, 55)
    else:
        raise ValueError(
            f"Region {extent} not in supported list of regions with default bounds. "
            f"Options are: 'global', 'north_america', 'uk', 'europe'."
        )
