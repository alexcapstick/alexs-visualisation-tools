import matplotlib.pyplot as plt
import seaborn as sns
import contextlib
import matplotlib
import numpy as np
from cycler import cycler

# colours
tol_muted = [
    '#332288', 
    '#88CCEE', 
    '#44AA99', 
    '#117733', 
    '#999933', 
    '#DDCC77', 
    '#CC6677', 
    '#882255',
    '#AA4499'
    ]

ibm = [
    "#648fff",
    "#fe6100",
    "#dc267f", 
    "#785ef0",
    "#ffb000",
    ]


# colour map
def set_colour_map(colours:list=tol_muted):
    '''
    Sets the default colour map for all plots.
    
    
    
    Examples
    ---------
    
    The following sets the colourmap to :code:`tol_muted`:

    .. code-block::
    
        >>> set_colour_map(colours=avt.tol_muted)
    
    
    Arguments
    ---------
    
    - colours: list, optional:
        Format that is accepted by 
        :code:`cycler.cycler`. 
        Defaults to :code:`tol_muted`.
    
    '''
    custom_params = {"axes.prop_cycle": cycler(color=colours)}
    matplotlib.rcParams.update(**custom_params)

# context functions
@contextlib.contextmanager
def temp_colour_map(colours=tol_muted):
    '''
    Temporarily sets the default colour map for all plots.
    

    Examples
    ---------
    
    The following sets the colourmap to :code:`tol_muted` for
    the plotting done within the context:

    .. code-block::
    
        >>> with set_colour_map(colours=avt.tol_muted):
        ...     plt.plot(x,y)
    
    
    Arguments
    ---------
    
    - colours: list, optional:
        Format that is accepted by 
        :code:`cycler.cycler`. 
        Defaults to :code:`tol_muted`.
    
    '''
    set_colour_map(colours=colours)


@contextlib.contextmanager
def paper_theme(colours=ibm):
    with matplotlib.rc_context():
        plt.style.use('seaborn-poster')
        custom_params = {
            
            "axes.spines.right": False, 
            "axes.spines.top": False, 
            "axes.edgecolor" : 'black',
            'axes.linewidth': 2,
            'axes.grid': True,
            'axes.axisbelow': True,
            "axes.prop_cycle": cycler(color=colours),

            'grid.alpha': 0.5,
            'grid.color': '#b0b0b0',
            'grid.linestyle': '--',
            'grid.linewidth': 2,

            "font.family": "Times New Roman",
            
            'xtick.major.width': 2,
            'ytick.major.width': 2,
            
            'boxplot.whiskerprops.linestyle': '-',
            'boxplot.whiskerprops.linewidth': 2,
            'boxplot.whiskerprops.color': 'black',
            
            'boxplot.boxprops.linestyle': '-',
            'boxplot.boxprops.linewidth': 2,
            'boxplot.boxprops.color': 'black',

            'boxplot.meanprops.markeredgecolor': 'black',

            'boxplot.capprops.color': 'black',
            'boxplot.capprops.linestyle': '-',
            'boxplot.capprops.linewidth': 1.0,
            
            'legend.title_fontsize': 16,
            'legend.fontsize': 16,

            }
        

        matplotlib.rcParams.update(**custom_params)

        yield