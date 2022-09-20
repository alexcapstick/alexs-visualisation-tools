import numpy as np
import pandas as pd
import typing

import matplotlib.colors as mcs
import matplotlib.pyplot as plt

def stackplot(
    data:pd.DataFrame, 
    x:typing.Union[str, None]=None, 
    y:typing.Union[str, None]=None, 
    hue:typing.Union[str, None]=None, 
    ax:typing.Union[plt.axes, None]=None, 
    hue_order:typing.Union[typing.List[str], None]=None, 
    cmap:typing.Union[mcs.Colormap, str, None]=None,
    legend:bool=True,
    **kwargs,
    ):
    '''
    This function plots a stacked continuous graph.
    
    
    
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
        :alt: Alternative text



    
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
    
    - kwargs:
        Any other keyword arguments are
        passed to :code:`plt.fill_between`. From here,
        you can change a variety of the bar
        attributes.
    
    
    Returns
    --------
    
    - out: plt.axes: 
        The axes containing the plot.
    
    
    '''

    if not hue is None:
        if hue_order is None:
            hue_order = data[hue].unique()

        if type(cmap) == str:
            cmap = plt.get_cmap(cmap)
            cmap = [cmap(i) for i in np.linspace(0,1,len(hue_order))]
        elif isinstance(cmap, mcs.Colormap):
            cmap = [cmap(i) for i in np.linspace(0,1,len(hue_order))]

        y_plot = np.vstack(
            [
                data[y].values[data[hue].values==hue_val].reshape(1,-1) 
                for hue_val in hue_order]
            )
        y_plot = np.cumsum(y_plot, axis=0)
        x_plot = np.vstack(
            [
                data[x].values[data[hue].values==hue_val].reshape(1,-1) 
                for hue_val in hue_order]
            )

    else:
        if type(cmap) == str:
            cmap = plt.get_cmap(cmap)
            cmap = [cmap(1)]
        elif isinstance(cmap, mcs.Colormap):
            cmap = [cmap(1)]

        hue_order = [x]
        y_plot = data[y].values.reshape(1,-1)
        x_plot = data[x].values.reshape(1,-1)


    if ax is None:
        fig, ax = plt.subplots(1,1)



    n_plots = 1 if hue_order is None else len(hue_order)

    for ng in range(n_plots):

        if not cmap is None:
            kwargs['color'] = [cmap[::-1][ng]]
        if not 'edgecolor' in kwargs:
            kwargs['edgecolor'] = 'white'

        ax.fill_between(
            x=x_plot[n_plots-1-ng, :],
            y1=y_plot[n_plots-1-ng, :],
            y2=0,
            label=hue_order[::-1][ng],
            **kwargs,
        )

    if legend:
        ax.legend()

    return ax