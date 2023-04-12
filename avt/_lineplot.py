"""
This module includes functions for generating continuous graphs.

Examples can be found here: 
https://github.com/alexcapstick/alexs-visualisation-tools/blob/main/examples/lineplot.ipynb

"""
import numpy as np
import pandas as pd
import typing
import matplotlib.colors as mcs
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from ._utils import interpolate_nans
import typing


def stackplot(
    data: pd.DataFrame,
    x: typing.Union[str, None] = None,
    y: typing.Union[str, None] = None,
    hue: typing.Union[str, None] = None,
    ax: typing.Union[plt.axes, None] = None,
    hue_order: typing.Union[typing.List[str], None] = None,
    cmap: typing.Union[mcs.Colormap, str, None] = None,
    legend: bool = True,
    cumulative: bool = False,
    **kwargs,
):
    """
    This function plots a stacked continuous graph. The
    missing values are interpolated for all x values.



    Examples
    ---------

    .. code-block::

        >>> import avt
        >>> import seaborn as sns
        >>> import pandas as pd
        >>> flights = sns.load_dataset('flights')
        >>> ax = avt.stackplot(flights, x='year', y='passengers', hue='month', cmap='Blues')

    This will return the plot:

    .. image:: figures/stackplot.png
        :width: 600
        :align: center
        :alt: Stack Plot Example


    Arguments
    ---------

    - data: pd.DataFrame:
        The data.

    - x: typing.Union[str, None], optional:
        The column name containing the x values.
        Defaults to :code:`None`.

    - y: typing.Union[str, None], optional:
        The column containing the heights.
        Defaults to :code:`None`.

    - hue: typing.Union[str, None], optional:
        Semantic variable that is mapped to determine
        the color of plot elements. This will determine
        the stacked bars.
        Defaults to :code:`None`.

    - ax: typing.Union[plt.axes, None], optional:
        A matplotlib axes that the plot can be drawn on.
        Defaults to :code:`None`.

    - hue_order: typing.Union[typing.List[str], None], optional:
        The order of the hue and stacked bars.
        Defaults to :code:`None`.

    - cmap: typing.Union[mcs.Colormap, str, None], optional:
        The colours of the plot. If a string is passed,
        this will be used to colour all of the stacked bars.
        If a cmap is passed, then this is used.
        If :code:`None`, then matplotlib handles
        the colours.
        Defaults to :code:`None`.

    - legend: bool, optional:
        Whether to plot a legend.
        Defaults to :code:`True`.

    - cumulative: bool, optional:
        If :code:`True`, then the cumulative
        values will be plotted, rather than
        the raw values.
        Defaults to :code:`False`.

    - kwargs:
        Any other keyword arguments are
        passed to :code:`plt.fill_between`. From here,
        you can change a variety of the bar
        attributes.


    Returns
    --------

    - out: plt.axes:
        The axes containing the plot.


    """

    data = data.sort_values(x)

    if not hue is None:
        if hue_order is None:
            hue_order = data[hue].unique()

        if type(cmap) == str:
            cmap = plt.get_cmap(cmap)
            cmap = [cmap(i) for i in np.linspace(0, 1, len(hue_order))]
        elif isinstance(cmap, mcs.Colormap):
            cmap = [cmap(i) for i in np.linspace(0, 1, len(hue_order))]

        data = (
            data[[x, y, hue]]
            .pivot_table(index=x, columns=hue, values=y)
            .apply(interpolate_nans, axis=0)
            .reset_index()
            .melt(id_vars=x, var_name=hue, value_name=y)
            .reset_index()
        )

        if cumulative:
            data = (
                data.groupby(by=[hue, x]).sum().groupby(level=0).cumsum().reset_index()
            )

        y_plot = np.vstack(
            [
                data[y].values[data[hue].values == hue_val].reshape(1, -1)
                for hue_val in hue_order
            ]
        )
        y_plot = np.cumsum(y_plot, axis=0)
        x_plot = np.vstack(
            [
                data[x].values[data[hue].values == hue_val].reshape(1, -1)
                for hue_val in hue_order
            ]
        )

    else:
        if type(cmap) == str:
            cmap = plt.get_cmap(cmap)
            cmap = [cmap(1)]
        elif isinstance(cmap, mcs.Colormap):
            cmap = [cmap(1)]

        hue_order = [x]
        y_plot = data[y].values.reshape(1, -1)
        x_plot = data[x].values.reshape(1, -1)
        if cumulative:
            y_plot = np.cumsum(y_plot, axis=1)

    if ax is None:
        ax = plt.gca()

    n_plots = 1 if hue_order is None else len(hue_order)

    for ng in range(n_plots):

        if not cmap is None:
            kwargs["color"] = [cmap[::-1][ng]]
        if not "edgecolor" in kwargs:
            kwargs["edgecolor"] = "white"

        ax.fill_between(
            x=x_plot[n_plots - 1 - ng, :],
            y1=y_plot[n_plots - 1 - ng, :],
            y2=0,
            label=hue_order[::-1][ng],
            **kwargs,
        )

    if legend:
        ax.legend()

    return ax


import matplotlib.ticker


def parallelplot(
    data: pd.DataFrame = None,
    x: str = None,
    y: str = None,
    hue: str = None,
    units: str = None,
    estimator: typing.Callable = np.mean,
    hue_order: typing.List[str] = None,
    order: typing.List[str] = None,
    bezier: bool = True,
    cmap: typing.Union[str, mcs.Colormap] = None,
    cbar: bool = False,
    cbar_x: float = 1.1,
    legend: bool = True,
    cbar_kwargs: typing.Dict[str, typing.Any] = {},
    legend_kwargs: typing.Dict[str, typing.Any] = {},
    ax=None,
    **kwargs,
) -> plt.Axes:
    """
    Plot a parallel plot.



    Examples
    ---------

    You can plot:

    .. image:: figures/parallelplot.png
        :width: 600
        :align: center
        :alt: Alternative text



    Arguments
    ---------

    - data: pd.DataFrame:
        The data to plot. Must be in long format.
        Defaults to :code:`None`.

    - x: str:
        The name of the column in :code:`data`
        that contains the categories to be
        plot along the x axis.
        Defaults to :code:`None`.

    - y: str:
        The name of the column in :code:`data`
        that contains the values to be plotted
        for each category.
        Defaults to :code:`None`.

    - hue: str:
        The name of the column in :code:`data`
        that contains the categories to be
        used to color the lines.
        Defaults to :code:`None`.

    - units: str:
        The name of the column in :code:`data`
        that contains the categories to be
        used to distinguish between lines
        that have the same :code:`hue` value.
        If not :code:`None`, then no
        estimation will be performed.
        Defaults to :code:`None`.

    - estimator: typing.Callable:
        The function to use to estimate the
        value of :code:`y` for each category.
        This is used if :code:`units` is not
        :code:`None`.
        Defaults to :code:`np.mean`.

    - hue_order: typing.List:
        The order in which to plot the
        :code:`hue` categories.
        Defaults to :code:`None`.

    - bezier: bool:
        Whether to use bezier curves to
        connect the points.
        Defaults to :code:`True`.

    - cmap: typing.Union[str, mcs.Colormap]:
        The colormap to use to color the
        lines.
        Defaults to :code:`None`.

    - cbar: bool:
        Whether to add a colorbar.
        Defaults to :code:`False`.

    - cbar_x: float:
        The x position of the colorbar.
        Defaults to :code:`1.1`.

    - cbar_kwargs: typing.Dict[str, typing.Any]:
        Additional keyword arguments to pass
        to :code:`plt.colorbar`.
        Defaults to :code:`{}`.

    - legend_kwargs: typing.Dict[str, typing.Any]:
        Additional keyword arguments to pass
        to :code:`plt.legend`.
        Defaults to :code:`{}`.

    - legend: bool:
        Whether to add a legend.
        Defaults to :code:`True`.

    - ax: plt.Axes:
        The axes on which to plot.
        Defaults to :code:`None`.

    - **kwargs:
        Additional keyword arguments to pass
        to :code:`plt.plot` or patches.PathPatch.


    Returns
    ---------

    - axes: plt.Axes:
        The axes on which the plot was made.


    """

    # edited from https://stackoverflow.com/a/60401570/19451559

    if ax is None:
        fig, host = plt.subplots()
    else:
        host = ax

    variable_names = list(data[x].unique()) if order is None else order

    index_cols = []
    if hue is not None:
        hues = data[hue].unique() if hue_order is None else hue_order
        index_cols.append(hue)
        if type(cmap) == str:
            cmap = plt.get_cmap(cmap)
            cmap = [cmap(i) for i in np.linspace(0, 1, len(hues))]
        elif isinstance(cmap, mcs.Colormap):
            cmap = [cmap(i) for i in np.linspace(0, 1, len(hues))]
        else:
            cmap = [
                next(host._get_lines.prop_cycler)["color"] for _ in range(len(hues))
            ]
    else:
        if type(cmap) == str:
            cmap = plt.get_cmap(cmap)
            cmap = [cmap(1)]
        elif isinstance(cmap, mcs.Colormap):
            cmap = [cmap(1)]
        else:
            cmap = [next(host._get_lines.prop_cycler)["color"]]

    if units is not None:
        index_cols.append(units)

    data = data.pivot_table(
        index=index_cols,
        columns=x,
        values=y,
        aggfunc=estimator,
        fill_value=np.nan,
    )
    data = data.reindex(variable_names, axis=1)

    # organize the data
    ys = data.values

    if ys.shape[0] == 1:
        raise ValueError(
            "Cannot plot parallel plot with only one line. " "Please pass hue or units"
        )

    if hue is not None:
        hue_values = data.index.get_level_values(hue).values
    else:
        hue_values = np.zeros(ys.shape[0]).astype(int)

    hue_values_to_cmap = dict(zip(np.unique(hue_values), cmap))

    ymins = ys.min(axis=0)
    ymaxs = ys.max(axis=0)
    dys = ymaxs - ymins
    ymins = ymins - dys * 0.05  # add 5% padding below and above
    ymaxs = ymaxs + dys * 0.05
    dys = ymaxs - ymins

    # transform all data to be compatible with the main axis
    zs = np.zeros_like(ys)
    zs[:, 0] = ys[:, 0]
    zs[:, 1:] = (ys[:, 1:] - ymins[1:]) / dys[1:] * dys[0] + ymins[0]

    nticks = len(host.get_yticklabels())

    axes = [host] + [host.twinx() for i in range(ys.shape[1] - 1)]
    for i, ax in enumerate(axes):
        ax.grid(False)
        ax.set_ylim(ymins[i], ymaxs[i])
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(True)
        # ax.yaxis.set_major_locator(matplotlib.ticker.LinearLocator(nticks))
        # ax.set_yticks(np.linspace(ymins[i], ymaxs[i], 5, endpoint=True))
        if ax != host:
            ax.spines["left"].set_visible(False)
            ax.yaxis.set_ticks_position("right")
            ax.spines["right"].set_position(("axes", i / (ys.shape[1] - 1)))

    host.set_xlim(0, ys.shape[1] - 1)
    host.set_xticks(range(ys.shape[1]))
    host.set_xticklabels(variable_names)
    host.tick_params(axis="x", which="major", pad=7)
    host.spines["right"].set_visible(False)
    host.xaxis.tick_top()

    if hue is not None:
        legend_handles = {h: None for h in hues}

    for j in range(zs.shape[0]):
        if bezier:
            # create bezier curves
            # for each axis, there will a control vertex at the point itself,
            # one at 1/3rd towards the previous and one
            # at one third towards the next axis; the first and last axis have one less control vertex
            # x-coordinate of the control vertices: at each integer (for the axes) and two inbetween
            # y-coordinate: repeat every point three times, except the first and last only twice
            verts = list(
                zip(
                    [
                        x
                        for x in np.linspace(
                            0, ys.shape[1] - 1, ys.shape[1] * 3 - 2, endpoint=True
                        )
                    ],
                    np.repeat(zs[j, :], 3)[1:-1],
                )
            )
            # for x, y in verts:
            #    host.plot(x, y, "go")  # to show the control points of the beziers
            codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
            path = Path(verts, codes)
            patch = patches.PathPatch(
                path,
                facecolor="none",
                edgecolor=hue_values_to_cmap[hue_values[j]],
                **kwargs,
            )
            host.add_patch(patch)
        else:
            # to just draw straight lines between the axes:
            host.plot(
                range(ys.shape[1]),
                zs[j, :],
                c=hue_values_to_cmap[hue_values[j]],
                **kwargs,
            )

        if hue is not None:
            legend_handles[hue_values[j]] = Line2D(
                [0],
                [0],
                label=hue_values[j],
                color=hue_values_to_cmap[hue_values[j]],
            )

    if cbar:
        cax = host.inset_axes(
            bounds=[cbar_x, 0, 0.05, 1.0],
        )

        norm = plt.Normalize(hue_values.min(), hue_values.max())
        sm = plt.cm.ScalarMappable(
            cmap=mcs.LinearSegmentedColormap.from_list("new_map", cmap), norm=norm
        )
        sm.set_array([])

        # Remove the legend and add a colorbar
        # ax.get_legend().remove()
        cbar = ax.figure.colorbar(sm, cax=cax, **cbar_kwargs)
        cbar.outline.set_linewidth(1)

    if legend:
        if hue is not None:
            host.legend(
                [legend_handles[h] for h in hues], hues, title=hue, **legend_kwargs
            )

    host.grid(False, axis="y")

    return host
