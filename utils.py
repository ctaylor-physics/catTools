# cat-tools is the start of a package where I can keep track of my handy functions. 
# This is to be a new and improved version of my previous functional codeblock RiometryTools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

### Re-used files and such

STYLE_PATH = "/home/cat-work/.config/matplotlib/paper.mplstyle"

### Functions
def calcFigSize(name="PRD", columns="twocol"):
    """
    Calculates figure sizes based on single column or two column publication style for LaTex friendly plots
    """
    pt = 1./72.27 # Hundreds of years of history... 72.27 points to an inch.

    jour_sizes = {"PRD": {"onecol": 246.*pt, "twocol": 510.*pt},
                "CQG": {"onecol": 374.*pt}, # CQG is only one column
                # Add more journals below. Can add more properties to each journal
                }

    my_width = jour_sizes[name][columns]
    # Our figure's aspect ratio
    golden = (1 + 5 ** 0.5) / 2

    figs = (my_width, my_width/golden)
    return figs


def get_colors(n, cmap='tab20', as_cmap=False):
    """
    Return a list of `n` colors sampled evenly from the given matplotlib colormap.

    Useful when you want to use the same color per-iteration across multiple plots.

    Parameters
    - n (int): number of distinct colors required.
    - cmap (str or Colormap): name of a matplotlib colormap or a Colormap instance.
    - as_cmap (bool): if True, return a ListedColormap instead of a list of RGBA colors.

    Returns
    - list of RGBA tuples (length n) or a `matplotlib.colors.ListedColormap` when `as_cmap=True`.

    Example
    colors = get_colors(5, cmap='tab10')
    for i, c in enumerate(colors):
        ax.plot(x, y[i], color=c)
    """
    if n <= 0:
        return [] if not as_cmap else mpl.colors.ListedColormap([])

    if isinstance(cmap, str):
        cmap_obj = mpl.cm.get_cmap(cmap)
    else:
        cmap_obj = cmap

    # sample n values evenly across [0, 1)
    samples = np.linspace(0, 1, n, endpoint=False)
    colors = [cmap_obj(s) for s in samples]

    if as_cmap:
        return mpl.colors.ListedColormap(colors)
    return colors

def normalize(sequence):
    return ( sequence - np.min(sequence) ) / (np.max(sequence) - np.min(sequence))
