import pandas as pd
import numpy as np

def interpolate_nans(x:pd.Series):
    '''
    Interpolate a pandas series.


    Arguments
    ---------

    - x: pd.Series:
        The series to interpolate. The index
        will be taken as the x values and the
        values will be taken as the y values.


    Returns
    ---------

    - res: pd.Series:
        The interpolated pandas data series.

    '''
    index = x.index
    is_nan = pd.isna(x.values)
    res = x.values * 1.0
    res[is_nan] = np.interp(
        x.index[is_nan], 
        x.index[~is_nan], 
        x.values[~is_nan], 
        left=0,
        )
    res = pd.Series(data=res, index=index)
    return res



def update_with_defaults(kwargs_dict, default_dict):
    kwargs_dict = kwargs_dict.copy()
    for key, value in default_dict.items():
        if key not in kwargs_dict:
            kwargs_dict[key] = value
    return kwargs_dict