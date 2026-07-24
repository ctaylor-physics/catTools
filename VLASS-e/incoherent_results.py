import os
import shutil
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, UTC
from astropy.time import Time
from scipy.spatial import cKDTree
import astropy.units as u
from astropy.coordinates import SkyCoord

from catTools.utils import STYLE_PATH, calcFigSize
from seticore.viewer import read_hits, read_stamps
from cosmic_database_analysis import sarfi



DIR='/home/cat-work/work/SETI/incoherentHits'
RDIR = '/home/cat-work/work/SETI/incoherentHits/results'
PDIR = '/home/cat-work/work/SETI/incoherentHits/results/diagnosePlots'
FDIR = '/home/cat-work/work/SETI/incoherentHits/results/finalPlots'

groupnames = ["perfect_match", "oneoff_freq", "oneoff_dr"]

def build_final_table():
    ## Perf
    perfect = pd.read_csv(os.path.join(DIR, 'results/perfect_match', 'stamp_diagnostics.csv'))
    perfect_info = pd.read_csv(os.path.join(DIR,"incoh_us_perfect_match_allstamps.csv"))
    pt = perfect[(perfect.signal_score == True) & (perfect.sarfi_score.isna())]
    p_merge = perfect_info.merge(pt,how='inner',on='id').drop(columns='Unnamed: 0')
    ## One frequency bin off
    oneoff_freq = pd.read_csv(os.path.join(DIR, 'results/oneoff_freq', 'stamp_diagnostics.csv'))
    oof_info = pd.read_csv(os.path.join(DIR,"incoh_us_oneoff_freq_allstamps.csv"))
    ooft = oneoff_freq[(oneoff_freq.signal_score == True) & (oneoff_freq.sarfi_score.isna())]
    oof_merge = oof_info.merge(ooft,how='inner',on='id').drop(columns='Unnamed: 0')
    ## One drift rate bin off
    oneoff_dr = pd.read_csv(os.path.join(DIR, 'results/oneoff_dr', 'stamp_diagnostics.csv'))
    oodr_info = pd.read_csv(os.path.join(DIR,"incoh_us_oneoff_dr_allstamps.csv"))
    oodrt = oneoff_dr[(oneoff_dr.signal_score == True) & (oneoff_dr.sarfi_score.isna())]
    oodr_merge = oodr_info.merge(oodrt,how='inner',on='id').drop(columns='Unnamed: 0')

    final_table = pd.concat(
        [
        p_merge.reset_index(drop=True),
        oof_merge.reset_index(drop=True),
        oodr_merge.reset_index(drop=True),
        ],
        axis=0
    )

    final_table.to_csv(os.path.join(DIR, 'results/incoh_us_final_sample.csv'))
    return

def collect_plots(final_table):
    final_table['show_ants_file'] = None
    for i, row in final_table.iterrows():
            plot_fn = os.path.join(PDIR,f"{row.source_name_y}_id{row.id}_{round(row.signal_frequency,5)}MHz_show_antennas.png")
            print(f"Looking for: {plot_fn}")
            if os.path.isfile(plot_fn):
                print('found it, appending')
                final_table.at[i,'show_ants_file'] = plot_fn
                out_fn = os.path.join(FDIR,f"{row.source_name_y}_id{row.id}_{round(row.signal_frequency,5)}MHz_show_antennas.png")
                shutil.copy(plot_fn, out_fn)
    return

### Step 1: Make combined table of hits from three groups
ft_filename = '/home/cat-work/work/SETI/incoherentHits/results/incoh_us_final_sample.csv'
final_table = pd.read_csv(ft_filename)

### Step 2: Collect plots for each remaining hit

# collect_plots(final_table) # Done Already!

### Step 3: Collect stamps for hits that pass manual inspection
# No stamps pass manual inspection, they are all RFI.
# We could include an example plot, but I am not sure it would be useful. 
final_table['mjd_seconds'] = ( final_table.tstart - final_table.tstart.min() )* 86400

fig, axs = plt.subplots(1,3)
axs[0].scatter(final_table.signal_frequency, final_table.signal_driftRate)
axs[0].set_xlabel('Signal Frequency (MHz)')
axs[0].set_ylabel('Signal DriftRate (Hz/s)')

axs[1].scatter(final_table.signal_frequency, final_table.mjd_seconds)
axs[1].set_xlabel('Signal Frequency (MHz)')
axs[1].set_ylabel('MJD seconds separation')

axs[2].scatter(final_table.ra, final_table.dec)
axs[2].set_xlabel('RA')
axs[2].set_ylabel('DEC')

plt.show()
# %%
# Min and Max separations?
ramin = final_table.ra.idxmin()
ramax = final_table.ra.idxmax()
a = SkyCoord(ra=final_table.ra.iloc[ramin] * u.hourangle, dec=final_table.dec.iloc[ramin] * u.deg)
b = SkyCoord(ra=final_table.ra.iloc[ramax] * u.hourangle, dec=final_table.dec.iloc[ramax] * u.deg)
print(a.separation(b))

decmin = final_table.dec.idxmin()
decmax = final_table.dec.idxmax()
c = SkyCoord(ra=final_table.ra.iloc[decmin] * u.hourangle, dec=final_table.dec.iloc[decmin] * u.deg)
d = SkyCoord(ra=final_table.ra.iloc[decmax] * u.hourangle, dec=final_table.dec.iloc[decmax] * u.deg)
print(c.separation(d))

# %%
