import matplotlib.pyplot as plt
import seaborn as sns

def boxplot(*args, **kwargs):
    '''
    This is a wrapper for the seaborn boxplot
    function that includes some default formatting.

    By default, all lines will have width :code:`2`, be black
    and the boxplot width is :code:`0.75`.
    '''
    boxprops = dict(
        linestyle='-',
        linewidth=2.0,
        edgecolor='black',
        )
    if 'boxprops' in kwargs:
        boxprops.update(kwargs['boxprops'])
    kwargs['boxprops'] = boxprops

    capprops = dict(
        linestyle='-',
        linewidth=2.0,
        color='black',
        )
    if 'capprops' in kwargs:
        capprops.update(kwargs['capprops'])
    kwargs['capprops'] = capprops

    medianprops = dict(
        linestyle='-',
        linewidth=2.0,
        color='black',
        )
    if 'medianprops' in kwargs:
        medianprops.update(kwargs['medianprops'])
    kwargs['medianprops'] = medianprops

    whiskerprops = dict(
        linestyle='-',
        linewidth=2.0,
        color='black',
        )
    if 'whiskerprops' in kwargs:
        whiskerprops.update(kwargs['whiskerprops'])
    kwargs['whiskerprops'] = whiskerprops

    if 'width' not in kwargs:
        kwargs['width'] = 0.75

    return sns.boxplot(
        *args, 
        **kwargs,
        )