### Code to filter through the hits that I gathered from the Meerkat sample 
import os
import glob
import time
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timezone, UTC
from astropy.time import Time
import seaborn as sns
from scipy.stats import median_abs_deviation

from catTools.utils import STYLE_PATH, calcFigSize
import catTools.cosmic_utils as cosmicu

from seticore import hit_capnp, stamp_capnp
from seticore.viewer import read_hits, read_stamps
from cosmic_database_analysis import sarfi

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
plt.style.use(STYLE_PATH)
figs = calcFigSize(name="CQG",columns='onecol')

DIR='/mnt/d/data1/meerkat_test_data/galacticPlane'


# metadata_csvs = sorted(glob.glob(DIR + '/mkstamp_metadata_blpn*.csv'))
# all_stamps = pd.DataFrame()
# for file in metadata_csvs:
#     temp = pd.read_csv(file)
#     all_stamps=pd.concat([all_stamps, temp], ignore_index=True)
all_stamps = pd.read_csv(DIR+'/mkstamp_metadata_combined.csv', index_col=0)

# RFI occupancy bands below
occupancy = pd.read_csv('/mnt/d/data1/meerkat_test_data/galacticPlane/meerkat_uhf_lband_rfi_occupancy_estimate.csv')
occupancy['approx_occupancy_fraction_mean'] = occupancy[['approx_occupancy_fraction_low', 'approx_occupancy_fraction_midpoint', 'approx_occupancy_fraction_high']].mean(axis=1)


### Step 1: Create a database of stamp information w/o the data field
def plot_summary_hits(df, save=False, out_fn = 'mkstamp_summary.png'):

    fig, axs = plt.subplots(1,2, sharex=True, layout='constrained', figsize=figs)
    fig.suptitle('Meerkat Hits Summary')
    axs[0].scatter(df.signal_frequency, df.signal_snr, c='r', s=2)
    axs[0].set_xlabel('Frequency (MHz)')
    axs[0].set_ylabel('SNR')

    axs[1].scatter(df.signal_frequency, df.signal_driftRate, c='b', s=2)
    axs[1].set_xlabel('Frequency (MHz)')
    axs[1].set_ylabel('Drift Rate (Hz/s)')
    if save:
        plt.savefig(os.path.join(DIR+out_fn))
    plt.show()
    return

### Step 1.1: Plot just Frequency, SNR, and Occupied Bands

def plot_summary_occ(df, save=False, out_fn = 'mkstamp_summary_occupancy.png'):
    # build colormap
    norm = Normalize(vmin=0.0, vmax=1.0)
    cmap = plt.colormaps["Reds"]

    # define figure
    fig, axs = plt.subplots(1,1, layout='constrained', figsize=figs)
    axs.scatter(df.signal_frequency, df.signal_snr, c='k', s=2)
    axs.set_xlabel('Frequency (MHz)')
    axs.set_ylabel('SNR')
    axs.set_ylim(-1,1000)

    # add RFI bands
    for row in occupancy.itertuples():
        axs.axvspan(
            row.start_frequency_mhz,
            row.stop_frequency_mhz,
            color=cmap(norm(row.approx_occupancy_fraction_mean)),
            alpha=0.55,
            linewidth=0,
            zorder=0,
        )

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=axs, pad=0.03)
    cbar.ax.set_ylim(0, occupancy.approx_occupancy_fraction_mean.max())
    cbarticks = np.linspace(0,occupancy.approx_occupancy_fraction_mean.max(), 6)
    cbar.set_label("Approximate occupancy fraction")
    if save:
        plt.savefig(os.path.join(DIR+out_fn))
    plt.show()
    return

### Step 1.2: Histograms
def dr_hist(dr):
    fig = plt.figure(figsize=figs, layout='constrained')
    plt.hist(
    dr.signal_driftRate,
    bins = np.arange(-10,11,1),
    edgecolor='k',
    facecolor='g',
    linewidth=1
    )
    plt.title('drift rate histogram')
    plt.xlabel('drift rate Hz/s')
    plt.ylabel('counts')
    plt.xticks(np.arange(-10,11,1),labels=np.arange(-10,11,1).astype(str),)
    plt.yscale('log')
    plt.show()

    return

def freq_hist(df, width=1., out_fn = None): #primarily for single band hits/stamps
    f_0 = np.floor(df_UHF_unique.signal_frequency.min() / width) * width
    f_1 = np.ceil(df_UHF_unique.signal_frequency.max() / width) * width
    f_bins = np.arange(int(df.signal_frequency.min()), int(df.signal_frequency.max()), width)

    fig, axs = plt.subplots(1,2,sharex=True, figsize=figs, layout='constrained')
    axs[0].scatter(df.signal_frequency, df.signal_snr, c='r', s=2)
    axs[0].set_xlabel('Frequency (MHz)')
    axs[0].set_ylabel('SNR')

    counts,_,_ = axs[1].hist(
                df.signal_frequency,
                bins = f_bins,
                edgecolor='k',
                facecolor='g',
                linewidth=0.3
                )
    axs[1].set_title('1MHz Occupancy')
    axs[1].set_xlabel('Frequency (MHz)')
    axs[1].set_ylabel('counts')
    # axs[1].yscale('log')
    if out_fn:
        plt.savefig(out_fn)

    plt.show()

    print(f"n_counts above 25 with width={width}MHz: {len(counts[counts > 25])}")
    print(f"Median Counts: {np.median(counts)}")
    print(f"STD of Counts: {np.std(counts)}")
    print(f"MAD Counts: {median_abs_deviation(counts)}")

    return

def freq_hist_comp(df_base, df_filt, width=0.25, out_fn = None): #primarily for single band hits/stamps
    f_0 = np.floor(df_base.signal_frequency.min() / width) * width
    f_1 = np.ceil(df_base.signal_frequency.max() / width) * width
    f_bins = np.arange(f_0, f_1, width)

    # plot1
    fig, axs = plt.subplots(1,1,sharex=True, figsize=figs, layout='constrained')
    axs.scatter(df_base.signal_frequency, df_base.signal_snr, c='r', s=1, zorder=1)
    axs.scatter(df_filt.signal_frequency, df_filt.signal_snr, c='g', s=1, zorder=2)
    axs.set_title('Raw Hits')
    axs.set_xlabel('Frequency (MHz)')
    axs.set_ylabel('SNR')

    if out_fn:
        plt.savefig(out_fn+'_scatter.png')

    plt.show()

    # plot2
    fig, axs = plt.subplots(1,1,sharex=True, figsize=figs, layout='constrained')
    counts,_,_ = axs.hist(
            df_base.signal_frequency,
            bins = f_bins,
            facecolor='r',
            zorder=1
            )

    counts2,_,_ = axs.hist(
                df_filt.signal_frequency,
                bins = f_bins,
                facecolor='g',
                zorder=2
                )
        
    axs.set_title(f'{width} MHz Occupancy')
    axs.set_xlabel('Frequency (MHz)')
    axs.set_ylabel('counts')

    if out_fn:
        plt.savefig(out_fn+'_hist.png')

    plt.show()

    return



def sns_histpercent(df, width=1):
    f_0 = np.floor(df_UHF_unique.signal_frequency.min() / width) * width
    f_1 = np.ceil(df_UHF_unique.signal_frequency.max() / width) * width
    f_bins = np.arange(int(df.signal_frequency.min()), int(df.signal_frequency.max()), width)

    sns.histplot(
        df.signal_frequency, 
        bins=f_bins, 
        stat='percent'
    )
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Percentage')
    plt.show()

    return



### Step 2: Filter the stamps based on my preconceptions of what is likely to be RFI
# Drift Rate:               Zeros removed, Cut max range values
# Signal-to-Noise Ratio:    SNR < 100

## No filtering
print(f"Starting length: {len(all_stamps)}")

## snr
all_stamps_snr = cosmicu.snr_Thresh(all_stamps)
print(f"SNR Filter Cut: {len(all_stamps_snr)}")

## drift rate - not very helpful at this stage. 
## Approximate frequency ranges: UHF (580-1015 MHz, 800 MHz), L (900-1670 MHz, 1300 MHz), S (1750-3500 MHz, 2600 MHz)
## If filtering for detectable exoplanets: 53 nHz * f = [42.4,  68.9, 137.8] Hz/s (i.e. no filtering)
## If filtering for Kepler derived sample: 0.44nHz * f = [0.352, 0.572, 1.144] Hz/s
# all_stamps_snr_dr = cosmicu.dr_Thresh(all_stamps_snr, upper = 1.15, lower=-1.15)
# print(f"SNR and tight DR Cut: {len(all_stamps_snr_dr)} (Just for testing, not used)")

## rfi 
occupancy.drop(occupancy.index[8], inplace=True) #aircraft transponders are low occ across band, high occ in narrow windows. 
all_stamps_snr_rfi = cosmicu.rfi_Clean(all_stamps_snr, occupancy, start_col='start_frequency_MHz', stop_col='stop_frequency_MHz')
print(f"RFI Filter Cut: {len(all_stamps_snr_rfi)}")


### Step 3: Break up into bands
foff_L, foff_S, foff_UHF = all_stamps_snr_rfi.foff.unique() 
df_UHF = all_stamps_snr_rfi[all_stamps_snr_rfi.foff == foff_UHF]
df_L = all_stamps_snr_rfi[all_stamps_snr_rfi.foff == foff_L]
df_S = all_stamps_snr_rfi[all_stamps_snr_rfi.foff == foff_S]

print("\nLength by band:")
print(f"UHF: {len(df_UHF)} \t L_band: {len(df_L)} \t S_band: {len(df_S)}")

### Step 4: Uniqueness filter (didnt get a lot of juice from this)
# Same freq, dr but different beam
# unique, snr-only
def band_uniqueness(df):

    foff = df.foff.iloc[0]
    dr_off = df.signal_driftRate.abs().min()
    unique_stamps = cosmicu.find_unique_signals(df,
                                                freq_tol=foff,
                                                freq_col='signal_frequency',
                                                drift_tol=dr_off,
                                                drift_col='signal_driftRate',
                                                target_col='sourceName',
    )
    return unique_stamps.reset_index(drop=True)

df_UHF_unique = band_uniqueness(df_UHF)
df_L_unique = band_uniqueness(df_L)
df_S_unique = band_uniqueness(df_S)

print(f"Uniqueness Filter Cut by band:")
print(f"UHF: {len(df_UHF_unique)} \t L_band: {len(df_L_unique)} \t S_band: {len(df_S_unique)}")

### CHECKPOINT
# df_UHF_unique.to_csv(os.path.join(DIR, 'mkstamp_UHF_unique.csv'))
# df_L_unique.to_csv(os.path.join(DIR, 'mkstamp_L_unique.csv'))
# df_S_unique.to_csv(os.path.join(DIR, 'mkstamp_S_unique.csv'))

## Testing some histograms, I think the sweetspot might be 0.5 MHz 

### Step 5
## Filtering based on occupancy in some frequency channel size seems optimal. 

def filter_by_frequency_occupancy(
    df,
    freq_col="signal_frequency",
    bin_width=1.0,
    max_hits_per_bin=100,
):
    """
    Remove hits located in frequency bins whose total hit count exceeds
    max_hits_per_bin.
    """

    if bin_width <= 0:
        raise ValueError("bin_width must be positive.")

    result = df.copy()

    valid = np.isfinite(result[freq_col].to_numpy())

    if not valid.any():
        result["frequency_bin"] = pd.NA
        result["bin_hit_count"] = 0
        return result.iloc[0:0].copy(), result

    freq_min = (
        np.floor(result.loc[valid, freq_col].min() / bin_width)
        * bin_width
    )
    freq_max = (
        np.ceil(result.loc[valid, freq_col].max() / bin_width)
        * bin_width
    )

    bin_edges = np.arange(
        freq_min,
        freq_max + bin_width,
        bin_width,
    )

    result["frequency_bin"] = pd.cut(
        result[freq_col],
        bins=bin_edges,
        right=False,
        include_lowest=True,
    )

    result["bin_hit_count"] = (
        result.groupby("frequency_bin", observed=False)[freq_col]
              .transform("size")
              .fillna(0)
              .astype(int)
    )

    filtered = result.loc[
        valid & result["bin_hit_count"].le(max_hits_per_bin)
    ].copy()

    return filtered, result

UHF_filt, UHF_result = filter_by_frequency_occupancy(df_UHF_unique,
                                                     bin_width=0.25,
                                                     max_hits_per_bin=25)

L_filt, L_result = filter_by_frequency_occupancy(df_L_unique,
                                                 bin_width=0.25,
                                                 max_hits_per_bin=25)

S_filt, S_result = filter_by_frequency_occupancy(df_S_unique,
                                                 bin_width=0.25,
                                                 max_hits_per_bin=25)



freq_hist_comp(df_UHF_unique,
               UHF_filt,
               width=0.25,
               out_fn=os.path.join(DIR, 'UHF_unique_filtering'))
freq_hist_comp(df_L_unique,
               L_filt,
               width=0.25,
               out_fn=os.path.join(DIR, 'L_unique_filtering'))
freq_hist_comp(df_S_unique,
               S_filt,
               width=0.25,
               out_fn=os.path.join(DIR, 'S_unique_filtering'))

# plot_summary_hits(df_UHF_unique)

print(f"Occupancy Based RFI Filter Cut by band:")
print(f"UHF: {len(UHF_filt)} \t L_band: {len(L_filt)} \t S_band: {len(S_filt)}")
print(f"Total Remaining: {len(UHF_filt) + len(L_filt) + len(S_filt)}")


### CHECKPOINT
UHF_filt.to_csv(os.path.join(DIR, 'mkstamp_UHF_unique_filt.csv'))
L_filt.to_csv(os.path.join(DIR, 'mkstamp_L_unique_filt.csv'))
S_filt.to_csv(os.path.join(DIR, 'mkstamp_S_unique_filt.csv'))

