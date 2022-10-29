from .__version__ import __version__, __author__, __doc__, __title__, __copyright__, __author_email__

from ._catplot import clockplot, bar_labels, timefreqheatmap
from ._lineplot import stackplot
from ._matrixplot import cfmplot
from ._scatterplot import scatter3dplot


__all__ =[
    'stackplot',
    'clockplot',
    'bar_labels',
    'timefreqheatmap',
    'cfmplot',
    'scatter3dplot'
]