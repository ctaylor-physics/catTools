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

