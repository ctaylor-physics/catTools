import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from utils import calcFigSize, STYLE_PATH

### This is to try plotting the VLA beam from Perley 2016


def get_beam(r, row):
    return row.A0 + (row.A2*r**2) + (row.A4*r**4) + (row.A6*r**6)

data = [
    [2052, 1.000, -1.429, 7.52, -1.47, 21.10],
    [2180, 1.000, -1.389, 7.06, -1.33, 21.38],
    [2436, 1.000, -1.377, 6.90, -1.27, 21.45],
    [2564, 1.000, -1.381, 6.92, -1.26, 21.42],
    [2692, 1.000, -1.402, 7.23, -1.40, 21.29],
    [2820, 1.000, -1.433, 7.62, -1.54, 21.09],
    [2948, 1.000, -1.433, 7.46, -1.42, 21.03],
    [3052, 1.000, -1.467, 8.05, -1.70, 20.87],
    [3180, 1.000, -1.497, 8.38, -1.80, 20.66],
    [3308, 1.000, -1.504, 8.37, -1.77, 20.58],
    [3436, 1.000, -1.521, 8.63, -1.88, 20.49],
    [3564, 1.000, -1.505, 8.37, -1.75, 20.57],
    [3692, 1.000, -1.521, 8.51, -1.79, 20.44],
    [3820, 1.000, -1.534, 8.57, -1.77, 20.33],
    [3948, 1.000, -1.516, 8.30, -1.66, 20.43],
]

columns = [
    "Freq_MHz",
    "A0",
    "A2_x1e-3",
    "A4_x1e-7",
    "A6_x1e-10",
    "HWHM_arcmin"
]

df = pd.DataFrame(data, columns=columns)
df["Freq_GHz"] = df['Freq_MHz'] / 1000
df["A2"] = df["A2_x1e-3"] * 1e-3
df["A4"] = df["A4_x1e-7"] * 1e-7
df["A6"] = df["A6_x1e-10"] * 1e-10

# df.to_csv('/home/cat-work/work/SETI/vla_Sband_beam.csv')

# Test for center of the band
row = df.iloc[7]

## Radial distance defined as r = R_arcmin x v_GHz
r = np.linspace(-50,50,100)
P_r = get_beam(r, row)
r_squared = np.sqrt((r[:,np.newaxis]**2 + r[np.newaxis,:]**2)) 

P_rsq = get_beam(r_squared, row)

# Format plot
figs = calcFigSize(name="CQG",columns='onecol')
plt.style.use(STYLE_PATH)

# ## Single profile
fig, ax = plt.subplots(1,1, figsize=figs, constrained_layout=True)
ax.set_title(f"{row.Freq_MHz} MHz 1-D Beam Profile")
ax.plot(r, P_r, c='r')
ax.set_xlabel('Radial Distance (arcmin)')
ax.set_xlim(0,50)
ax.set_ylabel('Power Response')
ax.set_ylim(0,1)
plt.show()

# %%
## 2D Profile
circ_loc = (P_rsq.shape[0] / 2 ) - 1
hpbw_circle = Circle((circ_loc,circ_loc), row.HWHM_arcmin, linestyle=':', edgecolor='k', fill=False)

fig, ax = plt.subplots(1,1, figsize=figs, constrained_layout=True)
ax.set_title(f"{row.Freq_MHz} MHz 2-D Beam Profile")
ax.imshow(P_rsq, origin='lower', cmap = 'seismic', vmin=0, vmax=1)
ax.add_patch(hpbw_circle)
# ax.set_xticks(r)
# ax.set_yticks(r)

plt.show()



