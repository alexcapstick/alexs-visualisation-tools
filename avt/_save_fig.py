import matplotlib.pyplot as plt

def save_fig(fig:plt.figure, file_name:str, **kwargs) -> None:
    '''
    This function saves a pdf, png, and svg of the figure,
    with :code:`bbox_inches='tight'` and :code:`dpi=300`.
    

    Arguments
    ---------

    - fig: plt.figure:
        The figure to save.
    
    - file_name: str:
        The file name, including path, to save the figure at.
        This should not include the extension, which will 
        be added when each file is saved.

    '''
    fig.savefig(f'{file_name}.pdf', bbox_inches='tight', **kwargs)
    fig.savefig(f'{file_name}.png', dpi=300, bbox_inches='tight', **kwargs)
    fig.savefig(f'{file_name}.svg', bbox_inches='tight', **kwargs)