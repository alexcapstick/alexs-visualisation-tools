import pandas as pd
import numpy as np
import matplotlib.markers as mm
import matplotlib.colors as mcs
import matplotlib.pyplot as plt
import seaborn as sns
import typing


def scatter3dplot(
    data:pd.DataFrame,
    x:typing.Union[str, None]=None, 
    y:typing.Union[str, None]=None, 
    z:typing.Union[str, None]=None, 
    hue:typing.Union[str, None]=None, 
    size:typing.Union[str, None]=25, 
    style:typing.Union[str, None]=None,
    ax:typing.Union[plt.axes, None]=None, 
    hue_order:typing.Union[typing.List[str], None]=None, 
    style_order:typing.Union[typing.List[str], None]=None, 
    cmap:typing.Union[mcs.Colormap, str, None]='RdBu_r',
    legend:bool=True,
    **kwargs,
    ):
    '''
    Note: This currently only supports categorical hue values.
    Please also ensure that ax is a 3D projection axes.

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

    for data_dict in data_list:
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
            s=data_dict['data'][size] if type(size) == str else size,
            **kwargs,
            )

    if legend:
        ax.legend(title=legend_title)

    return ax