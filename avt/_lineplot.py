import matplotlib.pyplot as plt

def stackplot(data, x=None, y=None, ax=None, colors=None, legend=True, **kwargs,):
    
    if ax is None:
        fig, ax = plt.subplots(1,1)
    
    ax.stackplot(
        data[x].values, 
        data[y].values.T,
        labels=y,
        colors=colors,
        **kwargs,
        )
    
    if legend:
        ax.legend()

    return ax