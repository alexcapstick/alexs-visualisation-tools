from .__version__ import (
    __version__,
    __author__,
    __doc__,
    __title__,
    __copyright__,
    __author_email__,
)

from ._boxplot import boxplot
from ._catplot import clockplot, bar_labels, timefreqheatmap, waterfallplot, radarplot
from ._lineplot import stackplot, parallelplot
from ._matrixplot import cfmplot
from ._save_fig import save_fig
from ._scatterplot import scatter3dplot
from ._theme import paper_theme, tol_muted, ibm, set_colour_map, temp_colour_map


__all__ = [
    "boxplot",
    "stackplot",
    "parallelplot",
    "clockplot",
    "bar_labels",
    "timefreqheatmap",
    "waterfallplot",
    "radarplot",
    "cfmplot",
    "save_fig",
    "scatter3dplot",
    "paper_theme",
    "tol_muted",
    "ibm",
    "set_colour_map",
    "temp_colour_map",
]
