"""
This module includes functions for generating categorical graphs.

Examples can be found here: 
https://github.com/alexcapstick/alexs-visualisation-tools/blob/main/examples/catplot.ipynb

"""

import pandas as pd
import numpy as np
import matplotlib.colors as mcs
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import typing
from ._utils import update_with_defaults


def clockplot(
    data: pd.DataFrame,
    x: typing.Union[str, None] = None,
    y: typing.Union[str, None] = None,
    hue: typing.Union[str, None] = None,
    ax: typing.Union[plt.axes, None] = None,
    hue_order: typing.Union[typing.List[str], None] = None,
    freq: str = "30T",
    label_format: typing.Union[bool, str] = True,
    label_freq: typing.Union[str, None] = None,
    cmap: typing.Union[mcs.Colormap, str, None] = None,
    legend: bool = True,
    label_kwargs: typing.Dict[str, typing.Any] = {},
    **kwargs,
):
    """
    This function plots a circular graph representing
    a day, with bars representing frequencies.



    Examples
    ---------

    .. code-block::

        >>> ax = avt.clockplot(
                data,
                x='datetime',
                hue='group',
                label_format='%H:%M',
                label_freq='3H',
                )

    This will return the plot:

    .. image:: figures/clockplot.png
        :width: 600
        :align: center
        :alt: Alternative text


    Arguments
    ---------

    - data: pd.DataFrame:
        The data.

    - x: typing.Union[str, None], optional:
        The column name containing the datetimes to use
        for calculating the time bins.
        Defaults to :code:`None`.

    - y: typing.Union[str, None], optional:
        Ignored.
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

    - freq: str, optional:
        The frequency to bin the bars at.
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases.
        Defaults to :code:`'30T'`.

    - label_format: typing.Union[bool, str], optional:
        The format of the time labels.
        Any argument to :code:`dt.strftime` is acceptable.
        https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior.
        Defaults to :code:`True`.

    - label_freq: typing.Union[str, None], optional:
        How often to show the time labels should be shown.
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases.
        :code:`True` leaves the labels as default and
        :code:`False` removes them.
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

    - label_kwargs: typing.Dict[str, typing.Any], optional:
        Keyword arguments to pass to the time labels.
        These are passed to :code:`plt.text`.
        Defaults to :code:`{}`.

    - kwargs:
        Any other keyword arguments are
        passed to :code:`plt.bar`. From here,
        you can change a variety of the bar
        attributes.


    Returns
    --------

    - out: plt.axes:
        The axes containing the plot.


    """

    data = data.copy()

    if not hue is None:
        if hue_order is None:
            hue_order = data[hue].unique()

        if type(cmap) == str:
            cmap = plt.get_cmap(cmap)
            cmap = [cmap(i) for i in np.linspace(0, 1, len(hue_order))]
        elif isinstance(cmap, mcs.Colormap):
            cmap = [cmap(i) for i in np.linspace(0, 1, len(hue_order))]

        agg_col = hue

    if hue is None:
        data = data.assign(__value__="any")
        agg_col = "__value__"

        if type(cmap) == str:
            cmap = plt.get_cmap(cmap)
            cmap = [cmap(1)]
        elif isinstance(cmap, mcs.Colormap):
            cmap = [cmap(1)]

    # making sure that 00:00 and 23:59 are in groups
    time_values = pd.timedelta_range(
        start="0 day", end="1 day", freq=freq, closed="left"
    )
    if type(label_format) != bool:
        time_values = (pd.to_datetime(0) + time_values).strftime(label_format)
    else:
        time_values = pd.to_datetime(0) + time_values
    new_row = pd.DataFrame(
        {
            x: time_values,
            agg_col: ["__nan__"] * len(time_values),
        }
    )
    data = pd.concat([new_row, data])

    # grouping data
    data = (
        data.assign(__value2__=1)
        .assign(**{x: lambda t: pd.to_datetime(t[x])})
        .assign(time=lambda t: t[x] - pd.to_datetime(t[x].dt.date))
        .assign(time=lambda t: pd.to_datetime(0) + t["time"])[
            [
                "time",
                agg_col,
            ]
        ]
        .groupby(by=[pd.Grouper(key="time", axis=0, freq=freq), agg_col])
        .agg(
            {
                agg_col: [
                    "count",
                ]
            }
        )
        .reset_index()
        .assign(time=lambda t: t["time"] - pd.to_datetime(t["time"].dt.date))
    )
    data["time_agg"] = pd.factorize(data["time"])[0]
    data.columns = [
        "time",
        agg_col,
        "count",
        "time_agg",
    ]

    # removing extra rows placed in data frame
    # when ensuring all time groups would be used
    hue_cats = [cat for cat in data[agg_col].unique() if not cat == "__nan__"]
    real_data_df = data[data[agg_col] != "__nan__"].copy()
    placeholder_data_df = data[data[agg_col] == "__nan__"].copy()
    for cat in hue_cats:
        real_data_df = (
            (real_data_df.set_index(["time", "time_agg", agg_col]))
            .add(
                placeholder_data_df.assign(**{agg_col: cat})
                .assign(count=0)
                .set_index(["time", "time_agg", agg_col]),
                fill_value=0,
            )
            .reset_index()
        )
    data = real_data_df.sort_values("time").copy()

    # plotting
    x = "time_agg"
    y = "count"

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw={"projection": "polar"})

    # plot polar axis
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    ax.axis("on")
    ax.grid(True)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.spines.get("polar").set_visible(False)

    # scaling bar heights to max of 1

    # sorting the bar heights to stack
    if hue is None:
        bar_heights = data[y].values.reshape(1, -1)
    else:
        bar_heights = np.vstack(
            [data[y].values[data[hue] == cat].reshape(1, -1) for cat in hue_order]
        )

    bar_heights = np.cumsum(bar_heights, axis=0)

    max_height = np.max(bar_heights[-1, :])
    bar_heights = bar_heights / max_height
    max_height = 1

    # Set the coordinate limits
    lower_limit = 5 * max_height / 8

    # Compute the bar_width of each bar. In total we have 2*Pi = 360°
    bar_width = 2 * np.pi / bar_heights.shape[1]

    if x is None:
        # Compute the angle each bar is centered on:
        indexes = list(range(0, bar_heights.shape[1]))
        angles = [element * bar_width for element in indexes]
    else:
        angles = data[x].unique()
        max_angle = np.max(angles)
        min_angle = np.min(angles)
        range_angle = max_angle - min_angle
        angles = (
            (angles - min_angle) / (range_angle + 1)
        ) * 2 * np.pi + bar_width * 0.5

    if not "bottom" in kwargs:
        kwargs["bottom"] = lower_limit

    bars_list = []
    n_plots = 1 if hue_order is None else len(hue_order)

    for nb in range(n_plots):
        # Draw bars

        if not cmap is None:
            kwargs["color"] = [cmap[::-1][nb]]

        bars = ax.bar(
            x=angles,
            height=bar_heights[::-1][nb],
            width=bar_width,
            linewidth=2,
            label="" if hue_order is None else hue_order[::-1][nb],
            **kwargs,
        )
        bars_list.append(bars)

    # little space between the bar and the label
    label_padding = 0.05

    if not hue is None:
        if legend:
            ax.legend()

    if type(label_format) == bool:
        if not label_format:
            return ax
        else:
            labels_allowed = data["time"]
            labels = (pd.to_datetime(0) + data["time"]).unique()
    else:
        if not label_freq is None:
            labels_allowed = pd.timedelta_range(
                start="0 day", end="1 day", freq=label_freq, closed="left"
            )
            labels_allowed = (pd.to_datetime(0) + labels_allowed).strftime(label_format)
        labels = (pd.to_datetime(0) + data["time"]).dt.strftime(label_format).unique()

    bars_ravel = [bars_list[0][-1]]
    bars_ravel.extend(bars_list[0])
    bars_ravel.append(bars_list[0][0])

    ha_alignment = label_kwargs.pop("ha", None)
    va_alignment = label_kwargs.pop("va", None)
    label_rotation = label_kwargs.pop("rotation", None)

    label_kwargs = update_with_defaults(
        label_kwargs,
        default_dict={
            "fontsize": 12,
            "rotation_mode": "anchor",
        },
    )

    for nb, (angle, height, label) in enumerate(zip(angles, bar_heights[-1], labels)):

        if height == 0:
            continue

        if not label_freq is None:
            if not label in labels_allowed:
                continue

        bar_height = np.max(
            [bars_ravel[nb].get_height(), bars_ravel[nb + 1].get_height()]
        )

        rotation = np.rad2deg(np.pi - (angle - bar_width * 0.5))

        # Flip some labels upside down
        ha_alignment_ = ""
        if label_rotation != 0.0:
            if angle >= 0 and angle < np.pi:
                ha_alignment_ = "left"
                va_alignment_ = "center"
                rotation = rotation + 270 if label_rotation is None else label_rotation
            else:
                ha_alignment_ = "right"
                va_alignment_ = "center"
                rotation = rotation + 90 if label_rotation is None else label_rotation
        else:
            if angle >= 0 and angle < np.pi / 4:
                ha_alignment_ = "center"
                va_alignment_ = "bottom"
                rotation = label_rotation
            elif angle >= np.pi / 4 and angle < 3 * np.pi / 4:
                ha_alignment_ = "left"
                va_alignment_ = "center"
                rotation = label_rotation
            elif angle >= 3 * np.pi / 4 and angle < np.pi:
                ha_alignment_ = "center"
                va_alignment_ = "top"
                rotation = label_rotation
            elif angle >= 3 * np.pi / 4 and angle < 5 * np.pi / 4:
                ha_alignment_ = "center"
                va_alignment_ = "top"
                rotation = label_rotation
            elif angle >= 5 * np.pi / 4 and angle < 7 * np.pi / 4:
                ha_alignment_ = "right"
                va_alignment_ = "center"
                rotation = label_rotation
            elif angle >= 7 * np.pi / 4 and angle < 2 * np.pi:
                ha_alignment_ = "center"
                va_alignment_ = "bottom"
                rotation = label_rotation
            else:
                ha_alignment_ = "center"
                va_alignment_ = "center"
                rotation = label_rotation

        # Finally add the labels
        ax.text(
            x=angle - bar_width * 0.5,
            y=max_height
            + lower_limit
            + label_padding
            * max_height,  # lowerLimit + bar_height + label_padding * max_height,  # bar.get_height(),
            s=label,
            ha=ha_alignment_ if ha_alignment is None else ha_alignment,
            va=va_alignment_ if va_alignment is None else va_alignment,
            rotation=rotation,
            **label_kwargs,
        )

    return ax


def timefreqheatmap(
    data: pd.DataFrame,
    x: typing.Union[str, None] = None,
    y: typing.Union[str, None] = None,
    hue: typing.Union[str, None] = None,
    ax: typing.Union[plt.axes, None] = None,
    hue_order: typing.Union[typing.List[str], None] = None,
    freq: str = "30T",
    label_format: typing.Union[bool, str] = True,
    cmap: typing.Union[typing.List[str], str, None] = None,
    binary: bool = False,
    **kwargs,
):
    """
    This function plots a heatmap with the frequencies
    of data points, against the date.



    Examples
    ---------

    .. code-block::

        >>> ax = avt.timefreqheatmap(
                data,
                x='datetime',
                hue='group',
                freq='1H',
                label_format='%H:%M-%d/%b/%Y',
                cmap='Blues',
                ax=ax
                )

    This will return the plot:

    .. image:: figures/timefreqheatmap.png
        :width: 600
        :align: center
        :alt: Alternative text



    Arguments
    ---------

    - data: pd.DataFrame:
        The data.

    - x: typing.Union[str, None], optional:
        The column name containing the datetimes to use
        for calculating the time bins.
        Defaults to :code:`None`.

    - y: typing.Union[str, None], optional:
        Ignored.
        Defaults to :code:`None`.

    - hue: typing.Union[str, None], optional:
        Semantic variable that is mapped to determine
        the color of plot elements. This will determine
        the different rows of the heatmap.
        Defaults to :code:`None`.

    - ax: typing.Union[plt.axes, None], optional:
        A matplotlib axes that the plot can be drawn on.
        Defaults to :code:`None`.

    - hue_order: typing.Union[typing.List[str], None], optional:
        The order of the hue and rows.
        Defaults to :code:`None`.

    - freq: str, optional:
        The frequency to bin the columns of the heatmap
        at.
        https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases.
        Defaults to :code:`'30T'`.

    - label_format: typing.Union[bool, str], optional:
        The format of the time labels.
        Any argument to :code:`dt.strftime` is acceptable.
        https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior.
        :code:`True` leaves the labels as default and
        :code:`False` removes them.
        Defaults to :code:`True`.

    - cmap: typing.Union[str, None], optional:
        This is the cmap for plotting the colours
        on the heatmap. If a :code:`str`, then this should
        be an acceptable argument to :code:`sns.mpl_palette`.
        If :code:`None`, this heatmap defaults to :code:`'inferno'`.
        Defaults to :code:`None`.

    - binary: bool:
        Whether to plot if there
        was a value or not, rather than the number of
        recorded values.
        Defaults to :code:`False`.

    - kwargs:
        Any other keyword arguments are
        passed to :code:`sns.heatmap`. From here,
        you can change a variety of the bar
        attributes.


    Returns
    --------

    - out: plt.axes:
        The axes containing the plot.


    """

    data = data.copy()

    data[x] = pd.to_datetime(data[x])

    if not hue is None:
        if hue_order is None:
            hue_order = data[hue].unique()

    if hue is None:
        data = data.assign(__value__="any")
        agg_col = "__value__"
    else:
        agg_col = hue

    # making sure that all times are in groups
    min_datetime = data[x].min()
    max_datetime = data[x].max()
    time_values = pd.timedelta_range(
        start=0, end=max_datetime - min_datetime, freq=freq, closed="left"
    )

    time_values = min_datetime + time_values
    new_row = pd.DataFrame(
        {
            x: time_values,
            agg_col: ["__nan__"] * len(time_values),
        }
    )
    data = pd.concat([new_row, data])

    # grouping data
    data = (
        data.assign(__value2__=1)
        .assign(**{x: lambda t: pd.to_datetime(t[x])})
        .assign(date_time=lambda t: t[x])[
            [
                "date_time",
                agg_col,
            ]
        ]
        .groupby(by=[pd.Grouper(key="date_time", axis=0, freq=freq), agg_col])
        .agg(
            {
                agg_col: [
                    "count",
                ]
            }
        )
        .reset_index()
    )

    data["date_time_agg"] = pd.factorize(data["date_time"])[0]
    data.columns = [
        "date_time",
        agg_col,
        "count",
        "date_time_agg",
    ]

    # removing extra rows placed in data frame
    # when ensuring all time groups would be used
    hue_cats = [cat for cat in data[agg_col].unique() if not cat == "__nan__"]
    real_data_df = data[data[agg_col] != "__nan__"].copy()
    placeholder_data_df = data[data[agg_col] == "__nan__"].copy()
    for cat in hue_cats:
        real_data_df = (
            (real_data_df.set_index(["date_time", "date_time_agg", agg_col]))
            .add(
                placeholder_data_df.assign(**{agg_col: cat})
                .assign(count=0)
                .set_index(["date_time", "date_time_agg", agg_col]),
                fill_value=0,
            )
            .reset_index()
        )
    data = real_data_df.sort_values("date_time").copy()

    # plotting
    x = "date_time_agg"
    date_col = "date_time"
    y = "count"

    if ax is None:
        ax = plt.gca()

    # getting array of counts
    groupby_list = [x, date_col]
    if not hue is None:
        groupby_list.append(hue)
    data = data.groupby(by=groupby_list).agg({"count": "sum"})
    if hue is None:
        data = data.assign(__level__=1).set_index("__level__", append=True)
    data = data.unstack().fillna(0)

    data = (
        data.reset_index()[[x, "date_time", "count"]]
        .sort_values(x)
        .assign(
            date_time=lambda t: t["date_time"].dt.strftime(label_format)
            if not type(label_format) is bool
            else t["date_time"]
        )
    )

    date_times = data["date_time"]
    data = data.set_index(x).T.reset_index(level=0, drop=True).loc[hue_order]

    # plotting
    cbar = True
    if binary:
        data = data > 0
        cbar = False
    else:
        data = data.astype(int)

    if cmap is None:
        cmap = sns.mpl_palette("inferno", as_cmap=not binary)
    elif type(cmap) == str:
        cmap = sns.mpl_palette(cmap, as_cmap=not binary)

    ax = sns.heatmap(data=data, ax=ax, cmap=cmap, cbar=cbar, **kwargs)

    ax.tick_params("x", rotation=-90)

    if hue is None:
        ax.set_ylabel("")
        ax.set_yticklabels("")

    if binary:
        legend_handles = [
            Patch(color=cmap[-1], label="On"),
            Patch(color=cmap[0], label="Off"),
        ]
        ax.legend(
            handles=legend_handles,
            ncol=2,
            bbox_to_anchor=[0.5, 1.02],
            loc="lower center",
        )

    x_ticks = np.asarray([tick.get_text() for tick in ax.get_xticklabels()], dtype=int)
    date_times = date_times.values[x_ticks]
    ax.set_xticklabels(date_times)

    if type(label_format) is bool:
        if not label_format:
            ax.set_xticks([])

    return ax


def waterfallplot(
    data: pd.DataFrame = None,
    x: str = None,
    y: str = None,
    hue: str = None,
    order: typing.List[str] = None,
    base: float = 0,
    orient: str = "h",
    estimator: typing.Union[str, typing.Callable] = "sum",
    cmap: str = None,
    alpha: bool = 0.75,
    positive_colour: str = "#648fff",
    negative_colour: str = "#fe6100",
    width: float = 0.8,
    bar_label: bool = True,
    ax: plt.Axes = None,
    arrow_kwargs: typing.Dict[str, typing.Any] = {},
    bar_kwargs: typing.Dict[str, typing.Any] = {},
    bar_label_kwargs: typing.Dict[str, typing.Any] = {},
):

    """
    This function allows you to draw a waterfall plot
    with a dataframe.




    Examples
    ---------

    You can build plots like:

    .. image:: figures/waterfallplot.png
        :width: 600
        :align: center
        :alt: Water fall plot

    To see the code producing this plot,
    view the examples :code:`ipynb` file.


    Arguments
    ---------

    - data: pd.DataFrame:
        The dataframe containing the data to plot.
        It must have at least two columns, one for the
        x-axis and one for the y-axis.
        Defaults to :code:`None`.

    - x: str:
        The name of the column in the dataframe that
        will be used for the x-axis.
        Defaults to :code:`None`.

    - y: str:
        The name of the column in the dataframe that
        will be used for the y-axis.
        Defaults to :code:`None`.

    - hue: str:
        The name of the column in the dataframe that
        will be used to colour the bars. This is not
        currently implemented.
        Defaults to :code:`None`.

    - order: list:
        The order in which the bars should be plotted.
        This is used for the categorical axis.
        Defaults to :code:`None`.

    - base: float:
        This is the value of the base of the waterfall
        plot. If not provided, a line will be drawn here and
        the first waterfall bar will be drawn from it.
        Defaults to :code:`None`.

    - orient: bool:
        Whether to plot the waterfall horizontally or vertically.
        Options are :code:`'h'` or :code:`'v'`.
        Defaults to :code:`h`.

    - estimator: typing.Union[str, typing.Callable]:
        The statistical function to use to aggregate the values.
        Defaults to :code:`'sum'`.

    - alpha: str:
        The alpha value to use for the arrows.
        Defaults to :code:`0.75`.

    - cmap: str:
        The name of the colourmap to use for the bars.
        If not provided, the bars will be coloured
        based on whether the value is positive or
        negative.
        Defaults to :code:`None`.

    - positive_colour: str:
        The colour to use for positive values.
        Defaults to :code:`'#648fff'`.

    - negative_colour: str:
        The colour to use for negative values.
        Defaults to :code:`'#fe6100'`.

    - width: float:
        The width of the bars.
        Defaults to :code:`0.8`.

    - bar_label: bool:
        Whether to add labels to the bars.
        Defaults to :code:`True`.

    - ax: matplotlib.axes.Axes:
        The axes to plot on. If not provided, a new
        figure and axes will be created.
        Defaults to :code:`None`.

    - arrow_kwargs: dict:
        A dictionary of keyword arguments to pass to
        the :code:`matplotlib.axes.Axes.arrow` function.
        Defaults to :code:`{}`.

    - bar_kwargs: dict:
        A dictionary of keyword arguments to pass to
        the :code:`matplotlib.axes.Axes.bar` function that
        the arrows are drawn upon.
        Defaults to :code:`{}`.

    - bar_label_kwargs: dict:
        A dictionary of keyword arguments to pass to
        the :code:`matplotlib.axes.Axes.bar_label` function
        that is overlayed on the arrows.
        Defaults to :code:`{}`.

    """

    if data is None:
        raise ValueError("data must be provided.")

    if hue is not None:
        raise NotImplementedError("hue not currently implemented.")

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    horizontal = orient == "h"

    categories = y if horizontal else x
    values = x if horizontal else y
    unique_categories = data[categories].unique()
    order = unique_categories if order is None else order
    bar_labels, bar_values = (
        data.groupby(categories)[values]
        .agg(estimator)
        .loc[order]
        .reset_index()
        .values.T
    )

    # if not all of the categories are in the order
    categories_not_in_order = unique_categories[~np.isin(unique_categories, order)]
    if len(categories_not_in_order) > 0:
        bar_labels_other, bar_values_other = (
            data.groupby(categories)[values]
            .agg(estimator)
            .loc[categories_not_in_order]
            .reset_index()
            .values.T
        )
        bar_labels_other = ["Other"]
        bar_values_other = [bar_values_other.sum()]
        bar_labels = np.concatenate([bar_labels, bar_labels_other])
        bar_values = np.concatenate([bar_values, bar_values_other])

    pos_bar = np.arange(len(bar_labels))
    height = bar_values
    bottom = np.concatenate([np.array([base]), base + np.cumsum(bar_values)])[:-1]
    ends = bottom + bar_values

    if horizontal:
        height = height[::-1]
        bottom = bottom[::-1]
        ends = ends[::-1]
        bar_labels = bar_labels[::-1]

    if horizontal:
        bars = ax.barh(
            y=pos_bar, height=width, left=bottom, alpha=0, width=height, **bar_kwargs
        )
    else:
        bars = ax.bar(
            pos_bar, height=height, bottom=bottom, alpha=0, width=width, **bar_kwargs
        )

    min_value = np.min([base, ends.min()])
    max_value = np.max([base, ends.max()])

    whole_range = max_value - min_value

    if horizontal:

        ax.set_xlim(
            min_value - 0.1 * whole_range,
            max_value + 0.1 * whole_range,
        )
    else:
        ax.set_ylim(
            min_value - 0.1 * whole_range,
            max_value + 0.1 * whole_range,
        )

    bar_boxes = [rect.get_bbox().get_points() for rect in bars.patches]

    if cmap is not None:
        cmap = sns.color_palette(cmap, len(pos_bar))
    else:
        cmap = []
        for bar_box in bar_boxes:
            if horizontal:
                if (bar_box[1, 0] - bar_box[0, 0]) > 0:
                    cmap.append(positive_colour)
                else:
                    cmap.append(negative_colour)
            else:
                if (bar_box[1, 1] - bar_box[0, 1]) > 0:
                    cmap.append(positive_colour)
                else:
                    cmap.append(negative_colour)

    arrow_kwargs = update_with_defaults(
        arrow_kwargs,
        {
            "head_width": width,
            "length_includes_head": True,
            "alpha": alpha,
            "linewidth": 0,
            "width": width,
            "head_length": whole_range * 0.025,
        },
    )

    for nb, bar_box in enumerate(bar_boxes):
        if horizontal:
            y_bar = pos_bar[nb]
            x_bar = bar_box[0, 0]
            dx_bar = bar_box[1, 0] - bar_box[0, 0]
            dy_bar = 0
        else:
            x_bar = pos_bar[nb]
            y_bar = bar_box[0, 1]
            dy_bar = bar_box[1, 1] - bar_box[0, 1]
            dx_bar = 0

        ax.arrow(
            x=x_bar,
            y=y_bar,
            dy=dy_bar,
            dx=dx_bar,
            color=cmap[nb] if cmap is not None else cmap,
            **arrow_kwargs,
        )

    # if horizontal:
    #    ax.axvline(base, color='black', linestyle='--', linewidth=2, zorder=0)
    # else:
    #    ax.axhline(base, color='black', linestyle='--', linewidth=2, zorder=0)

    bar_label_kwargs = update_with_defaults(
        bar_label_kwargs,
        {
            "fmt": "%.2f",
            "label_type": "center",
            "padding": 0,
            "fontsize": 16,
        },
    )

    if bar_label:
        ax.bar_label(
            bars,
            **bar_label_kwargs,
        )

    if horizontal:
        ax.set_yticks(pos_bar)
        ax.set_yticklabels(bar_labels)
    else:
        ax.set_xticks(pos_bar)
        ax.set_xticklabels(bar_labels)

    ax.set_ylabel(y)
    ax.set_xlabel(x)

    ax.grid(False, axis="y" if horizontal else "x")

    return ax


def radarplot(
    data: pd.DataFrame = None,
    x: str = None,
    y: str = None,
    hue: str = None,
    order: typing.List[str] = None,
    hue_order: typing.List[str] = None,
    estimator: typing.Callable = np.mean,
    fill: bool = True,
    cmap: typing.Union[str, typing.List[str]] = None,
    legend: bool = True,
    ax: plt.Axes = None,
    **kwargs,
) -> plt.Axes:
    """
    This function allows you to create a radar
    plot from a dataframe.


    Examples
    ---------

    You can build plots like:

    .. image:: figures/radarplot.png
        :width: 600
        :align: center
        :alt: Radar Plot

    To see the code producing this plot,
    view the examples :code:`ipynb` file.



    Arguments
    ---------

    - data: pd.DataFrame:
        The dataframe containing the data to plot.
        Defaults to :code:`None`.

    - x: str:
        The name of the column containing the
        categories to plot.
        Defaults to :code:`None`.

    - y: str:
        The name of the column containing the
        values to plot.
        Defaults to :code:`None`.

    - hue: str:
        The name of the column containing the
        hues to plot.
        Defaults to :code:`None`.

    - order: list:
        The order of the categories to plot.
        Defaults to :code:`None`.

    - hue_order: list:
        The order of the hues to plot.
        Defaults to :code:`None`.

    - estimator: callable:
        The function to use to aggregate the
        values when plotting.
        Defaults to :code:`np.mean`.

    - fill: bool:
        Whether to fill the area under the curve.
        Defaults to :code:`True`.

    - cmap: str or list:
        The name of the colormap to use or a list
        of colours to use.
        Defaults to :code:`None`.

    - legend: bool:
        Whether to show the legend.
        Defaults to :code:`True`.

    - ax: plt.Axes:
        The axes to plot on.
        Defaults to :code:`None`.

    - **kwargs:
        Additional keyword arguments to pass to
        :code:`plt.plot`.

    Returns
    ---------
    - ax: plt.Axes:
        The axes containing the plot.


    """

    groupby_cols = [hue, x] if hue is not None else [x]
    all_cols = groupby_cols + [y]
    data = data[all_cols].groupby(groupby_cols).agg(estimator).reset_index()

    # number of categories to plot
    variables = list(data[x].unique()) if order is None else order
    variables_n = len(variables)

    # setting the order of the variables around the circle
    order = list(data[x].unique()) if order is None else order

    # getting the hues, order and colours
    if hue is not None:
        hues = list(data[hue].unique())
        hues = hues if hue_order is None else hue_order
        variables_and_values = [
            (
                data.set_index(hue).loc[h].query(f"{x} in {order}")[x].values.tolist(),
                data.set_index(hue).loc[h].query(f"{x} in {order}")[y].values.tolist(),
            )
            for h in hues
        ]
        if type(cmap) == str:
            cmap = plt.get_cmap(cmap)
            cmap = [cmap(i) for i in np.linspace(0, 1, len(hues))]
        elif isinstance(cmap, mcs.Colormap):
            cmap = [cmap(i) for i in np.linspace(0, 1, len(hues))]
    else:

        variables_and_values = [
            (
                data.query(f"{x} in {order}")[x].values.tolist(),
                data.query(f"{x} in {order}")[y].values.tolist(),
            )
        ]

        if type(cmap) == str:
            cmap = plt.get_cmap(cmap)
            cmap = [cmap(1)]
        elif isinstance(cmap, mcs.Colormap):
            cmap = [cmap(1)]

    # angles for the plot
    angles = [n / float(variables_n) * 2 * np.pi for n in range(variables_n)]
    variable_angle_dict = {v: a for v, a in zip(variables, angles)}

    # setting plotting area
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6), subplot_kw={"polar": True})

    # place first cat on top of plot and set direction clockwise
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels
    ax.set_xticks(angles)
    ax.set_xticklabels(variables)

    # Draw ylabels
    ax.set_rlabel_position(0)

    # plotting
    for nh, (variables_plot, values_plot) in enumerate(variables_and_values):
        angles_plot = [variable_angle_dict[v] for v in variables_plot]
        values_plot = [
            v
            for _, v in sorted(zip(angles_plot, values_plot), key=lambda pair: pair[0])
        ]
        angles_plot = sorted(angles_plot)

        angles_plot += angles_plot[:1]
        values_plot += values_plot[:1]

        colour = (
            cmap[nh] if cmap is not None else next(ax._get_lines.prop_cycler)["color"]
        )

        ax.plot(
            angles_plot,
            values_plot,
            label=hues[nh] if hue is not None else None,
            color=colour,
            **kwargs,
        )
        # Fill area
        if fill:
            ax.fill(
                angles_plot,
                values_plot,
                color=colour,
                alpha=0.1,
            )

    if legend:
        if hue is not None:
            ax.legend(title=hue)

    return ax


def bar_labels(
    obj: plt.figure,
    labels: typing.Union[None, typing.List[str]] = None,
    label_format: str = "{height}",
    **kwargs,
) -> plt.figure:
    """
    Adds labels to bars in graph.


    Examples
    ---------

    The following will add label heights and format
    them as a percentage:

    .. code-block::

        >>> g = sns.catplot(...)
        >>> avt.bar_labels(g.figure, label_format='{height:.1%}')


    The following will add the labels :code:`'bar1'` and :code:`'bar2'`
    to the bars in the first axes and :code:`'bar3'`
    and :code:`'bar4'` to the bars in the second axes.

    .. code-block::

        >>> g = sns.catplot(...)
        >>> labels = [
                ['bar1','bar2'], # first axis
                ['bar3', 'bar4'] # second axis
                ]
        >>> avt.bar_labels(g.figure, labels=labels)


    Arguments
    ---------

    - obj: plt.figure or plt.Axes:
        The matplotlib figure or axes to add bar labels to.

    - labels: typing.List[str]:
        Labels to add to the bars.
        This should be a list of lists, in which
        the outer list acts over the axes in
        the plot and the inner list is the labels
        for the bars. If :code:`None`, then the bar labels
        will be the heights of the bars.
        Defaults to :code:`None`.

    - label_format: str:
        The format of the bar labels, used only
        if :code:`labels=None`. This needs to contain
        the word :code:`height`, in :code:`{}`, where the bar
        heights will be placed.
        Defaults to :code:`'{height}'`.

    - kwargs: :
        Keyword arguments passed to :code:`plt.bar_label`.
        Examples could be :code:`rotation`, :code:`padding`,
        :code:`label_type`, and :code:`fontsize`.


    Returns
    --------

    - out: plt.figure:
        Matplotlib figure, containing the added
        bar labels.


    """

    if isinstance(obj, plt.Axes):
        axes_list = [obj]

    elif isinstance(obj, plt.figure):
        axes_list = obj.axes

    else:
        raise TypeError("obj must be a matplotlib figure or axes")

    for ax in axes_list:
        for nc, c in enumerate(ax.containers):
            if labels is None:
                # heights = [v.get_height() for v in c]
                # l = [label_format.format(height=height) for height in heights]
                l = None
            else:
                l = labels[nc]
            ax.bar_label(
                c,
                labels=l,
                **kwargs,
            )

    return obj
