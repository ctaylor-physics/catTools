import numpy as np
import matplotlib.pyplot as plt
from utils import STYLE_PATH, calcFigSize, get_colors


# Format plot
figs = calcFigSize(name="CQG",columns='onecol')
plt.style.use(STYLE_PATH)

axes_names = ['Antenna' , '50MHz Pk ($\Delta$ dB)']
data = np.array([['9X'	, -46],
    ['11X'	, -18],
    ['14X'	, -16],
    ['16Y'	, -16],
    ['20Y'	, -41],
    ['23X'	, -6],
    ['25X'   , 3],
    ['28Y'	, -18],
    ['30Y'	, -2],
    ['31Y'	, -17],
    ['32X'	, 3],
    ['32Y'	, 3],
    ['33Y'	, 4],
    ['34X'	, -37],
    ['34Y'	, -42],
    ['39X'	, -42],
    ['39Y'	, -36],
    ['43X'	, -35],
    ['43Y'	, -41],
    ['46Y'	, -41],
    ['55X'	, -41],
    ['61Y'	, -19],
    ['62Y'	, -15],
    ['64Y'	, 3],
    ['open'  , -43]])
names = data[:,0]
values = np.array([int(x) for x in data[:,1]])

fig = plt.figure()
plt.scatter(np.arange(len(data)), values, s=3, c='k')
plt.xticks(ticks=np.arange(len(data)), labels=names, rotation = -30, fontsize=4.5)
plt.hlines(y=4, xmin=0, xmax=25, colors='green', label='ref antenna')
plt.hlines(y=-43,  xmin=0, xmax=25, colors='r', label='open')
plt.xlabel(axes_names[0])
plt.ylabel(axes_names[1])
plt.ylim(-50,10)
plt.legend()
plt.show()