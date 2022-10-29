'''
This module includes functions for generating scatter graphs.

Examples can be found here: 
https://github.com/alexcapstick/alexs-visualisation-tools/blob/main/examples/scatterplot.ipynb

'''


import pandas as pd
import numpy as np
import matplotlib.markers as mm
import matplotlib.colors as mcs
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import typing


def scatter3dplot(
    data:pd.DataFrame,
    x:typing.Union[str, None]=None, 
    y:typing.Union[str, None]=None, 
    z:typing.Union[str, None]=None, 
    hue:typing.Union[str, None]=None, 
    style:typing.Union[str, None]=None,
    size:typing.Union[str, int]=25, 
    ax:typing.Union[plt.axes, None]=None, 
    hue_order:typing.Union[typing.List[str], None]=None, 
    style_order:typing.Union[typing.List[str], None]=None, 
    cmap:typing.Union[mcs.Colormap, str, None]='RdBu_r',
    legend:bool=True,
    **kwargs,
    ) -> plt.axes:
    '''
    This function plots a 3D scatterplot.

    Note: This currently only supports categorical hue values.
    Please also ensure that ax is a 3D projection axes.
    
    
    Examples
    ---------

    .. code-block:: 

        >>> import avt
        >>> import numpy as np
        >>> from sklearn.datasets import load_iris
        >>> data_dict = load_iris(as_frame=True)
        >>> data, target = data_dict['data'], data_dict['target']
        >>> data['target'] = target
        >>> data['random group'] = np.random.choice(2, size=len(data))
        >>> ax = avt.scatter3dplot(
            data=data,
            x='sepal length (cm)',
            y='sepal width (cm)',
            z='petal length (cm)',
            hue='target',
            size='petal width (cm)',
            style='random group',
            )

    This will return the plot:

    .. image:: figures/scatter3dplot.png
        :width: 600
        :align: center
        :alt: Alternative text

    Alternatively, you can use an animation to 
    better visualise the distribution:

    .. code-block::

        >>> import avt
        >>> import numpy as np
        >>> from sklearn.datasets import load_iris
        >>> from matplotlib import animation
        >>> data_dict = load_iris(as_frame=True)
        >>> data, target = data_dict['data'], data_dict['target']
        >>> data['target'] = target
        >>> data['random group'] = np.random.choice(2, size=len(data))
        >>> ax = avt.scatter3dplot(
                data=data,
                x='sepal length (cm)',
                y='sepal width (cm)',
                z='petal length (cm)',
                hue='target',
                size='petal width (cm)',
                style='random group',
                )
        >>> ani = animation.FuncAnimation(
                ax.figure, 
                lambda x: ax.view_init(30,x), 
                frames=np.linspace(1,360,90), 
                interval=1, 
                blit=False,
                )
        >>> ax.figure.tight_layout()

    This will return the plot:

    .. image:: figures/scatter3dplot.gif
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
        The column containing the  y values. 
        Defaults to :code:`None`.
    
    - z: typing.Union[str, None], optional:
        The column containing the  z values. 
        Defaults to :code:`None`.
    
    - hue: typing.Union[str, None], optional:
        Semantic variable that is mapped to determine 
        the color of plot elements.
        Defaults to :code:`None`.
    
    - style: typing.Union[str, None], optional:
        Semantic variable that is mapped to determine 
        the shape of plot elements.
        Defaults to :code:`None`.
    
    - size: typing.Union[str, int], optional:
        Semantic variable that is mapped to determine 
        the size of the plot elements. If :code:`str` 
        then the sizes are determined by the column
        values. If :code:`int` then this will be
        used as the size.
        Defaults to :code:`None`.
    
    - ax: typing.Union[plt.axes, None], optional:
        A matplotlib axes that the plot can be drawn on. 
        Defaults to :code:`None`.
    
    - hue_order: typing.Union[typing.List[str], None], optional:
        The order of the hue and legend. 
        Defaults to :code:`None`.
    
    - style_order: typing.Union[typing.List[str], None], optional:
        The order of the style and legend. 
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
        passed to :code:`plt.scatter`. From here,
        you can change a variety of the bar
        attributes.
    
    
    Returns
    --------
    
    - out: plt.axes: 
        The axes containing the plot.
    
    
    '''

    assert (not x is None) and (not y is None) and (not z is None),\
        "Please ensure you specify x, y, and z."
    
    assert (type(size) == int) or (type(size) == str),\
        "Please ensure size is an int or str."

    legend_title=""
    if not hue is None:
        legend_title += f"{hue}"
        if hue_order is None:
            hue_order = data[hue].unique()
        hue_order_dict = {h: i for i,h in enumerate(hue_order)}
        if (type(cmap) == str) or (cmap is None):
            cmap = plt.get_cmap(cmap)
            cmap = [cmap(i) for i in np.linspace(0,1,len(hue_order))]
        elif isinstance(cmap, mcs.Colormap):
            cmap = [cmap(i) for i in np.linspace(0,1,len(hue_order))]

        data_list = [
            {'data': data.query(f"`{hue}`==@h"), 'hue':h} 
            for h in hue_order
            ]
    else:
        if (type(cmap) == str) or (cmap is None):
            cmap = plt.get_cmap(cmap)
            cmap = [cmap(1)]
        elif isinstance(cmap, mcs.Colormap):
            cmap = [cmap(1)]
        hue_order_dict = None

        data_list = [{'data':data, 'hue': None}]

    if not style is None:
        if len(legend_title) > 0:
            legend_title += f", {style}"
        else:
            legend_title += f"{style}"
        if style_order is None:
            style_order = data[style].unique()
        style_order_dict = {s: i for i,s in enumerate(style_order)}
        data_list = [
            {'data': d['data'].query(f"`{style}`==@s"), 'hue': d['hue'], 'style': s}
            for d in data_list
            for s in style_order
            ]
    else:
        style_order_dict = None
        data_list = [
            {'data': d['data'], 'hue': d['hue'], 'style': None}
            for d in data_list
            ]
    
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10,10), subplot_kw={'projection': '3d'})

    markers = mm.MarkerStyle().filled_markers

    handles = []
    for data_dict in data_list:
        
        if len(data_dict['data']) == 0:
            continue
        
        label = ""
        if not hue_order is None:
            label += str(data_dict['hue'])
            if not cmap is None:
                kwargs['color'] = [cmap[::-1][hue_order_dict[data_dict['hue']]]]
        else:
            kwargs['color'] = cmap[0]

        if not style_order is None:
            if len(label) > 0:
                label += ", " + str(data_dict['style'])
            else:
                label += str(data_dict['style'])
            kwargs['marker'] = markers[style_order_dict[data_dict['style']]%len(markers)]
        else:
            kwargs['marker'] = markers[0]

        if not 'edgecolor' in kwargs:
            kwargs['edgecolors'] = 'face'

        ax.scatter(
            xs=data_dict['data'][x], 
            zs=data_dict['data'][z],
            ys=data_dict['data'][y], 
            label=label,
            s=50*(data_dict['data'][size])/(np.max(data_dict['data'][size].values)) \
                if type(size) == str else size,
            **kwargs,
            )

        handles.append(
            mlines.Line2D(
                [], [], color=kwargs['color'][0], marker=kwargs['marker'], linestyle='None',
                markersize=10, label=label
                )
            )

    if legend:
        ax.legend(handles=handles, title=legend_title)

    return ax