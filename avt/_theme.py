import matplotlib.pyplot as plt
import seaborn as sns
import contextlib
import matplotlib
import numpy as np
from cycler import cycler


@contextlib.contextmanager
def paper_theme():
    with matplotlib.rc_context():
        plt.style.use('seaborn-poster')
    
        colours = [
            "#648fff",
            "#fe6100",
            "#dc267f", 
            "#785ef0",
            "#ffb000",
            ]

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