import numpy as np
import pandas as pd
import typing

import matplotlib.pyplot as plt

def stackplot(
    data:pd.DataFrame, 
    x:typing.Union[str, None]=None, 
    y:typing.Union[str, None]=None, 
    hue:typing.Union[str, None]=None, 
    ax:typing.Union[plt.axes, None]=None, 
    hue_order:typing.Union[typing.List[str], None]=None, 
    colors:typing.Union[typing.List[str], str, None]=None,
    legend:bool=True,
    **kwargs,
    ):
    '''
    This function plots a stacked continuous graph.
    
    
    
    Examples
    ---------
    ```
    >>> fig, ax = plt.subplots(1, 1, figsize=(8,8))
    >>> ax = avt.stackplot(
            data=data_plot, 
            x='date', 
            y='Cumulative Frequency',
            hue='Outcome',
            hue_order=['Negative UTI', 'Positive UTI'],
            colors=[colour1, colour2],
            ax=ax,
            legend=False,
            )
    ```
    
    Arguments
    ---------
    
    - `data`: `pd.DataFrame`: 
        The data.
    
    - `x`: `typing.Union[str, None]`, optional:
        The column name containing the datetimes to use
        for calculating the time bins. 
        Defaults to `None`.
    
    - `y`: `typing.Union[str, None]`, optional:
        Ignored. 
        Defaults to `None`.
    
    - `hue`: `typing.Union[str, None]`, optional:
        Semantic variable that is mapped to determine 
        the color of plot elements. This will determine
        the stacked bars.
        Defaults to `None`.
    
    - `ax`: `typing.Union[plt.axes, None]`, optional:
        A matplotlib axes that the plot can be drawn on. 
        Defaults to `None`.
    
    - `hue_order`: `typing.Union[typing.List[str], None]`, optional:
        The order of the hue and stacked bars. 
        Defaults to `None`.
    
    - `colors`: `typing.Union[typing.List[str], str, None]`, optional:
        The colours of the plot. If a string is passed,
        this will be used to colour all of the stacked bars.
        If a list is passed, it will be iterated over when plotting
        the stacked bars (please ensure it is at least as long as
        the number of hue values). If `None`, then matplotlib handles
        the colours. 
        Defaults to `None`.
    
    - `legend`: `bool`, optional:
        Whether to plot a legend. 
        Defaults to `True`.
    
    - `kwargs`:
        Any other keyword arguments are
        passed to `plt.fill_between`. From here,
        you can change a variety of the bar
        attributes.
    
    
    Returns
    --------
    
    - `out`: `plt.axes` : 
        The axes containing the plot.
    
    
    '''

    if not hue is None:
        if hue_order is None:
            hue_order = data[hue].unique()
        

    if hue is None:
        if type(colors) == str:
            colors = [colors]
        elif colors is None:
            colors = [None]

    else:
        if type(colors) == str:
            colors = [colors]*len(hue_order)
        elif colors is None:
            colors = [None]*len(hue_order)
    
    if hue is None:
        y_plot = data[y].values.reshape(1,-1)
        x_plot = data[x].values
    else:
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

    if ax is None:
        fig, ax = plt.subplots(1,1)

    if not 'edgecolor' in kwargs:
        kwargs['edgecolor'] = 'white'

    n_plots = 1 if hue_order is None else len(hue_order)

    for ng in range(n_plots):
        ax.fill_between(
            x=x_plot[n_plots-1-ng, :],
            y1=y_plot[n_plots-1-ng, :],
            y2=0,
            label=hue_order[::-1][ng],
            color=[colors[::-1][ng]],
            **kwargs,
        )

    if legend:
        ax.legend()

    return ax