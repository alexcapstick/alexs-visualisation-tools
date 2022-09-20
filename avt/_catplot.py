import pandas as pd
import numpy as np
import matplotlib.colors as mcs
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import typing



def clockplot(
    data:pd.DataFrame, 
    x:typing.Union[str, None]=None, 
    y:typing.Union[str, None]=None, 
    hue:typing.Union[str, None]=None, 
    ax:typing.Union[plt.axes, None]=None, 
    hue_order:typing.Union[typing.List[str], None]=None, 
    freq:str='30T', 
    label_format:typing.Union[bool, str]=True, 
    label_freq:typing.Union[str, None]=None,
    cmap:typing.Union[mcs.Colormap, str, None]=None,
    legend:bool=True,
    **kwargs,
    ): 
    '''
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
    
    - kwargs:
        Any other keyword arguments are
        passed to :code:`plt.bar`. From here,
        you can change a variety of the bar
        attributes.
    
    
    Returns
    --------
    
    - out: plt.axes: 
        The axes containing the plot.
    
    
    '''



    data = (data
        .copy()
        )

    if not hue is None:
        if hue_order is None:
            hue_order = data[hue].unique()

        if type(cmap) == str:
            cmap = plt.get_cmap(cmap)
            cmap = [cmap(i) for i in np.linspace(0,1,len(hue_order))]
        elif isinstance(cmap, mcs.Colormap):
            cmap = [cmap(i) for i in np.linspace(0,1,len(hue_order))]
        
        agg_col = hue


    if hue is None:
        data = data.assign(__value__='any')
        agg_col = '__value__'
        
        if type(cmap) == str:
            cmap = plt.get_cmap(cmap)
            cmap = [cmap(1)]
        elif isinstance(cmap, mcs.Colormap):
            cmap = [cmap(1)]


    # making sure that 00:00 and 23:59 are in groups
    time_values = pd.timedelta_range(start='0 day', end='1 day', freq=freq, closed='left')
    if type(label_format) != bool:
        time_values = (pd.to_datetime(0) + time_values).strftime(label_format)
    else:
        time_values = pd.to_datetime(0) + time_values
    new_row = pd.DataFrame({
        x: time_values,
        agg_col: ['__nan__']*len(time_values),
        
        })
    data = pd.concat([new_row, data])

    # grouping data
    data = (data
        .assign(__value2__ = 1)
        .assign(**{x : lambda t: pd.to_datetime(t[x])})
        .assign(time = lambda t: t[x] - pd.to_datetime(t[x].dt.date))
        .assign(time=lambda t: pd.to_datetime(0) + t['time'])
        [['time', agg_col,]]
        .groupby(by=[pd.Grouper(key='time', axis=0, freq=freq), agg_col])
        .agg({agg_col: ['count',]})
        .reset_index()
        .assign(time = lambda t: t['time'] - pd.to_datetime(t['time'].dt.date))
        )
    data['time_agg'] = pd.factorize(data['time'])[0]
    data.columns = ['time', agg_col, 'count', 'time_agg', ]

    # removing extra rows placed in data frame
    # when ensuring all time groups would be used
    hue_cats = [cat for cat in data[agg_col].unique() if not cat == '__nan__']
    real_data_df = data[data[agg_col] != '__nan__'].copy()
    placeholder_data_df = data[data[agg_col] == '__nan__'].copy()
    for cat in hue_cats:
        real_data_df = (real_data_df
            .set_index(['time', 'time_agg', agg_col])
            ).add(
                placeholder_data_df
                    .assign(**{agg_col: cat})
                    .assign(count=0)
                    .set_index(['time', 'time_agg', agg_col]),
                fill_value=0
                ).reset_index()
    data = real_data_df.sort_values('time').copy()

    # plotting
    x = 'time_agg'
    y = 'count'
    
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(8,8), subplot_kw={'projection': 'polar'})

    # plot polar axis
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    # remove grid
    ax.axis('off')

    # scaling bar heights to max of 1

    # sorting the bar heights to stack
    if hue is None:
        bar_heights = data[y].values.reshape(1,-1)
    else:
        bar_heights = np.vstack(
            [
                data[y].values[data[hue] == cat].reshape(1,-1) 
                for cat in hue_order
                ]
            )

    bar_heights = np.cumsum(bar_heights, axis=0)

    max_height = np.max(bar_heights[-1,:])
    bar_heights = bar_heights/max_height
    max_height=1

    # Set the coordinate limits
    lowerLimit = 5*max_height/8

    # Compute the bar_width of each bar. In total we have 2*Pi = 360°
    bar_width = 2*np.pi / bar_heights.shape[1]

    if x is None:
        # Compute the angle each bar is centered on:
        indexes = list(range(0, bar_heights.shape[1]))
        angles = [element * bar_width for element in indexes]
    else:
        angles = data[x].unique()
        max_angle = np.max(angles)
        min_angle = np.min(angles)
        range_angle = max_angle-min_angle
        angles = ((angles-min_angle)/(range_angle+1))*2*np.pi + bar_width*0.5

    if not 'bottom' in kwargs:
        kwargs['bottom'] = lowerLimit
    
    bars_list = []
    n_plots = 1 if hue_order is None else len(hue_order)

    for nb in range(n_plots):
        # Draw bars

        if not cmap is None:
            kwargs['color'] = [cmap[::-1][nb]]

        bars = ax.bar(
            x=angles, 
            height=bar_heights[::-1][nb], 
            width=bar_width, 
            linewidth=2,
            label= '' if hue_order is None else hue_order[::-1][nb],
            **kwargs)
        bars_list.append(bars)

    # little space between the bar and the label
    labelPadding = 0.05

    if not hue is None:
        if legend:
            ax.legend() 

    if type(label_format) == bool:
        if not label_format:
            return ax
        else:
            labels = (pd.to_datetime(0) + data['time']).unique()
    else:
        if not label_freq is None:
            labels_allowed = pd.timedelta_range(start='0 day', end='1 day', freq=label_freq, closed='left')
            labels_allowed = (pd.to_datetime(0) + labels_allowed).strftime(label_format)
        labels = (pd.to_datetime(0) + data['time']).dt.strftime(label_format).unique()
        

    bars_ravel = [bars_list[0][-1]]
    bars_ravel.extend(bars_list[0])
    bars_ravel.append(bars_list[0][0])

    for nb, (angle, height, label) in enumerate(zip(angles, bar_heights[-1], labels)):

        if height == 0:
            continue

        if not label_freq is None:
            if not label in labels_allowed:
                continue


        bar_height = np.max([bars_ravel[nb].get_height(), bars_ravel[nb+1].get_height()])

        rotation = np.rad2deg(np.pi-(angle-bar_width*0.5))

        # Flip some labels upside down
        alignment = ""
        if angle >= 0 and angle < np.pi:
            alignment = "left"
            rotation = rotation + 270
        else: 
            alignment = "right"
            rotation = rotation + 90

        # Finally add the labels
        ax.text(
            x=angle-bar_width*0.5, 
            y=lowerLimit + bar_height + labelPadding * max_height, #bar.get_height(), 
            s=label, 
            ha=alignment, 
            va='center', 
            rotation=rotation, 
            rotation_mode="anchor",
            fontsize=12,
            ) 

    return ax










def timefreqheatmap(
    data:pd.DataFrame, 
    x:typing.Union[str, None]=None, 
    y:typing.Union[str, None]=None, 
    hue:typing.Union[str, None]=None, 
    ax:typing.Union[plt.axes, None]=None, 
    hue_order:typing.Union[typing.List[str], None]=None, 
    freq:str='30T', 
    label_format:typing.Union[bool, str]=True, 
    cmap:typing.Union[typing.List[str], str, None]=None,
    binary:bool=False,
    **kwargs,
    ): 
    '''
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
    
    
    '''


    data = (data
        .copy()
        )
    
    data[x] = pd.to_datetime(data[x])

    if not hue is None:
        if hue_order is None:
            hue_order = data[hue].unique()

    if hue is None:
        data = data.assign(__value__='any')
        agg_col = '__value__' 
    else:
        agg_col = hue

    # making sure that all times are in groups
    min_datetime = data[x].min()
    max_datetime = data[x].max()
    time_values = pd.timedelta_range(start=0, end=max_datetime-min_datetime, freq=freq, closed='left')


    time_values = min_datetime + time_values
    new_row = pd.DataFrame({
        x: time_values,
        agg_col: ['__nan__']*len(time_values),
        
        })
    data = pd.concat([new_row, data])


    # grouping data
    data = (data
        .assign(__value2__ = 1)
        .assign(**{x : lambda t: pd.to_datetime(t[x])})
        .assign(date_time = lambda t: t[x])
        [['date_time', agg_col,]]
        .groupby(by=[pd.Grouper(key='date_time', axis=0, freq=freq), agg_col])
        .agg({agg_col: ['count',]})
        .reset_index()
        )

    data['date_time_agg'] = pd.factorize(data['date_time'])[0]
    data.columns = ['date_time', agg_col, 'count', 'date_time_agg', ]

    # removing extra rows placed in data frame
    # when ensuring all time groups would be used
    hue_cats = [cat for cat in data[agg_col].unique() if not cat == '__nan__']
    real_data_df = data[data[agg_col] != '__nan__'].copy()
    placeholder_data_df = data[data[agg_col] == '__nan__'].copy()
    for cat in hue_cats:
        real_data_df = (real_data_df
            .set_index(['date_time', 'date_time_agg', agg_col])
            ).add(
                placeholder_data_df
                    .assign(**{agg_col: cat})
                    .assign(count=0)
                    .set_index(['date_time', 'date_time_agg', agg_col]),
                fill_value=0
                ).reset_index()
    data = real_data_df.sort_values('date_time').copy()

    # plotting
    x = 'date_time_agg'
    date_col = 'date_time'
    y = 'count'

    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(8,8))

    # getting array of counts
    groupby_list=[x, date_col]
    if not hue is None: groupby_list.append(hue)
    data = (data
        .groupby(by=groupby_list)
        .agg({'count':'sum'})
        )
    if hue is None:
        data = data.assign(__level__=1).set_index('__level__', append=True)
    data = data.unstack().fillna(0)

    data = (data
        .reset_index()
        [[x, 'date_time', 'count']]
        .sort_values(x)
        .assign(date_time=lambda t: 
            t['date_time'].dt.strftime(label_format) if not type(label_format) is bool else t['date_time']
            )
        )

    date_times = data['date_time']
    data= (data
        .set_index(x)
        .T
        .reset_index(level=0, drop=True)
        .loc[hue_order]
        )

    # plotting
    cbar=True
    if binary:
        data = data > 0
        cbar=False
    else:
        data = data.astype(int)

    if cmap is None:
        cmap = sns.mpl_palette('inferno', as_cmap= not binary)
    elif type(cmap) == str:
        cmap = sns.mpl_palette(cmap, as_cmap=not binary)

    ax = sns.heatmap(data=data, ax=ax, cmap=cmap, cbar=cbar, **kwargs)

    ax.tick_params('x', rotation=-90)

    if hue is None:
        ax.set_ylabel('')
        ax.set_yticklabels('')
    
    if binary:
        legend_handles = [Patch(color=cmap[-1], label='On'),
                        Patch(color=cmap[0], label='Off')] 
        ax.legend(handles=legend_handles, ncol=2, bbox_to_anchor=[0.5, 1.02], loc='lower center')
    
    x_ticks = np.asarray([tick.get_text() for tick in ax.get_xticklabels()], dtype=int)
    date_times = date_times.values[x_ticks]
    
    ax.set_xticklabels(date_times)

    return ax


















def bar_labels(
    fig:plt.figure, 
    labels:typing.Union[None, typing.List[str]]=None,
    label_format:str='{height}',
    **kwargs,
    ) -> plt.figure:
    '''
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
    
    - fig: plt.figure: 
        The matplotlib figure to add bar labels to.
    
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
    
    
    '''
    
    for ax in fig.axes:
        for nc, c in enumerate(ax.containers):
            if labels is None:
                heights = [v.get_height() for v in c]
                l = [label_format.format(height=height) for height in heights]
            else:
                l = labels[nc]
            ax.bar_label(
                c, 
                labels=l, 
                **kwargs,
                )
        ax.margins(y=0.2)
        ax.set_ylim(0,1.2)

    return fig
