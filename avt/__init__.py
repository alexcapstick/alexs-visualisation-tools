from .__version__ import __version__, __author__, __doc__, __title__, __copyright__, __author_email__

from ._boxplot import boxplot
from ._catplot import clockplot, bar_labels, timefreqheatmap
from ._lineplot import stackplot
from ._matrixplot import cfmplot
from ._save_fig import save_fig
from ._scatterplot import scatter3dplot
from ._theme import paper_theme


__all__ =[
    'boxplot',
    'stackplot',
    'clockplot',
    'bar_labels',
    'timefreqheatmap',
    'cfmplot',
    'save_fig',
    'scatter3dplot',
    'paper_theme',
]