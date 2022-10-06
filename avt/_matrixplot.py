'''
This module includes functions for generating matrix graphs.

Examples can be found here: 
https://github.com/alexcapstick/alexs-visualisation-tools/blob/main/examples/matrix.ipynb

'''

import typing
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def cfmplot(
    cfm:np.ndarray, 
    cbar:bool=True, 
    color:str='Blues', 
    categories:typing.List[str]='auto', 
    xlabel:bool=True, 
    ylabel:bool=True, 
    summary_statistics:bool=True,
    ax:typing.Union[None, plt.axes]=None,
    ) -> plt.axes:
    '''
    Draw a confusion matrix plot from a numpy array.
    
    This function was based on code from 
    https://github.com/DTrimarchi10/confusion_matrix/blob/master/cfm_matrix.py.
    
    
    Examples
    ---------
    
    When using this function to plot multi-label task confusion matrices,
    we will get something similar to the following:

    .. code-block::

        >>> cfm = np.array(
            [[10,  0,  0],
            [ 0, 12,  0],
            [ 0,  1, 15]],
            dtype=int64
            )
        >>> cfm_plot(cfm)

    .. image:: figures/cfmplot-multi_label.png
        :width: 600
        :align: center
        :alt: Alternative text
    
    If we have a binary task, the summary statistics are
    more extensive:

    .. code-block::

        >>> cfm = np.array(
            [[15,  0],
            [ 0, 10]], 
            dtype=int64
            )
        >>> cfm_plot(cfm)

    .. image:: figures/cfmplot-binary_label.png
        :width: 600
        :align: center
        :alt: Alternative text




    Arguments
    ---------
    
    - cfm: np.ndarray: 
        A numpy array representing the confusion matrix.
    
    - cbar: bool, optional:
        Whether to add a colour bar. 
        Defaults to :code:`True`.
    
    - color: str, optional:
        The cmap that can be used for colors. 
        Defaults to :code:`'Blues'`.
    
    - categories: typing.List[str], optional:
        The categories in the predicted and true axes. 
        Defaults to :code:`'auto'`.
    
    - xlabel: bool, optional:
        whether to include a label on the x axis.
        Defaults to :code:`True`.
    
    - ylabel: bool, optional:
        whether to include a label on the y axis. 
        Defaults to :code:`True`.
    
    - summary_statistics: bool, optional:
        Whether to add summary statistics to the
        label on the x axis. 
        Defaults to :code:`True`.
    
    - ax: typing.Union[None, plt.axes], optional:
        Axes in which to draw the plot, 
        otherwise use the currently-active Axes. 
        Defaults to :code:`None`.
    
    
    
    Returns
    --------
    
    - out: plt.axes: 
        Axes object with the confusion matrix.
    
    
    '''

    if ax is None:
        ax = plt.gca()

    if type(cfm).__name__ == 'Series':
        cfm = cfm.sum()

    group_percentages = ["{0:.2%}".format(value) for value in cfm.flatten()/np.sum(cfm)]
    group_counts = ["{0:0.0f}\n".format(value) for value in cfm.flatten()]

    box_labels = [f"{counts}{percs}" for counts, percs in zip(group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cfm.shape[0],cfm.shape[1])

    if summary_statistics:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cfm) / float(np.sum(cfm))

        #if it is a binary confusion matrix, show some more stats
        if len(cfm)==2:
            #Metrics for Binary Confusion Matrices
            precision = cfm[1,1] / sum(cfm[:,1])
            recall    = cfm[1,1] / sum(cfm[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    ax = sns.heatmap(
        cfm,
        annot=box_labels,
        fmt="",
        cmap=color,
        cbar=cbar,
        xticklabels=categories,
        yticklabels=categories,
        ax=ax,
        )

    if xlabel:
        ax.set_xlabel('Predicted label' + stats_text)
    else:
        ax.set_xlabel(stats_text)
    
    if ylabel:
        ax.set_ylabel('True label')

    return ax