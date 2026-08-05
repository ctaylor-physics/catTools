import os 
import sys 
import numpy as np
import pandas as pd

from astropy.time import Time
from astropy.table import Table, Column
from astropy.io import votable

from scipy.signal import find_peaks
from scipy.spatial import cKDTree

import datetime as datetime

import matplotlib.pyplot as plt


### takes in a pickle filename, clips out data for the input mjd
def dayClip(fn, mjd_date, outfn):
    # Read in k2-18b dataset
    dataframe = pd.read_pickle(fn)

    # Slice the first day of observing (I already know the first day is 60220)
    singleDay = dataframe.loc[(dataframe['tstart'] > (mjd_date-1)) & (dataframe['tstart'] < mjd_date)]
    cols = singleDay.columns

    # Add datetime column
    time = Time(singleDay['tstart'].values, format='mjd', scale='utc')
    singleDay['tstart_datetime'] = time.to_datetime()

    # Save out Progress to avoid loading the whole meatball 
    singleDay.to_pickle(outfn)
    return

# Convert ecsv to VOTable and write as xml 
def toVOTable(tabin, outname):
    outable = votable.from_table(tabin)
    return votable.writeto(outable, outname)


def are_tables_the_same(fn1, fn2):
    df1 = pd.read_pickle(fn1)
    df2 = pd.read_pickle(fn2)
    if df1.equals(df2):
        print('identical dataframes')
    else:
        print('non-identical dataframes')
    return

def calculate_EIRP(S_jy, BW, distance):
    """
    Give a distance, get an estimated EIRP and ratio to Arecibo transmitter
    Sensitivity in Janskys (S_jy) for a given observing config
    Bandwidth in Hz (BW)
    """
    S_jy = S_jy * 10**(-26)# Jy = 1 * 10^-26 W/Hz/m^2
    dist = distance *  3.08567e16 # 1 pc = 3.085 * 10^16 m
    Fmin = S_jy*BW
    EIRP = 4 * np.pi * Fmin * dist**2 # Watts
    perArecibo = EIRP / (2e13) # Unitless

    print(f"Equivalent Isotropic Radiated Power: {EIRP:e} W")
    print(f"Ratio of EIRP/Arecibo: {perArecibo}")

    ## Maximum distance to detect Arecibo:
    singleAreciboDistance = np.sqrt((2e13)/(4*np.pi*Fmin)) / (3.08567e16) # pc
    print(f"Single Arecibo Distance: {singleAreciboDistance} pc")
    return

def johnson_EIRP(SEFD, BW, distance, t):
    print(f"{(10*4*np.pi*(10 * 3.08567e16)**2 )*(9200 * 10**(-26)) * np.sqrt(3/(2*600)):e}")

def look_for_combs(stamp_data, name, thresh, cepstrum_offset, out_fn = None):
    """
    This is a notepad for a function that would take in a stamp dynamic spectra
    and look for frequency combs present in the image
    
    I noticed that RFI looks like an obvious frequency comb sometimes. 
    The input is from the stamp tutorial code. 
    inputs:
        ant_stamp_data: Antenna stamp data of form [timestep, channel]
        threshold: threshold of comb peaks (maybe amp ~100 based on Voyager signal)
    output:
        score whether peaks exist above the threshold
    """

    log_s = np.log(stamp_data.mean(axis=0))
    cepstrum = np.abs(np.fft.rfft(log_s))**2
    quefrency = np.fft.rfftfreq(len(log_s), d=2)
    peaks, props = find_peaks(cepstrum[cepstrum_offset:], height = 2* thresh, threshold=thresh)

    if len(peaks) > 0:
        if out_fn is not None:
            title = f"{name}\n{len(peaks)} sharp peaks above height {2 * thresh}. max peak = {np.max(props["peak_heights"]).round()}"
            fig, ax = plt.subplots(1,2, layout='constrained')
            fig.suptitle(title)
            ax[0].plot(log_s)
            ax[0].set_ylabel(f'Log(signal)')
            ax[0].set_xlabel(f'Channel')

            ax[1].set_xlabel(f"quefrency")
            ax[1].set_ylabel(f"Cepstrum amplitude")
            ax[1].plot(cepstrum[cepstrum_offset:])
            ax[1].scatter(np.array(peaks), cepstrum[peaks+cepstrum_offset], c='r')
            plt.savefig(out_fn + f"_combplot_{name}.png")
            plt.close(fig)
        return 1
    else: 
        return 2

def log_with_pandas(data_dict, file_path=None):
    # Create a DataFrame from the new data
    new_data_df = pd.DataFrame([data_dict])

    # Check if file exists to decide whether to write header
    if file_path is not None:            
        if os.path.exists(file_path):
            new_data_df.to_csv(file_path, mode='a', index=False, header=False)
        else:
            new_data_df.to_csv(file_path, mode='w', index=False, header=True)

### Functions for Filtering Hits-like tables
def is_rfi(frequency, ranges, start_col, stop_col):
    # Check if frequency is inside any RFI range
    # From Chenoa Tremblay k2-18b
    return np.any((frequency >= ranges[start_col]) & (frequency <= ranges[stop_col]))

def rfi_Clean(df, rfi_df, start_col = 'start_frequency', stop_col = 'stop_frequency'):
    """
    This function is a preliminary RFI screen for incoming data.
        It is from Chenoa's K2-18b code
    """

    # Build and apply mask
    mask = df['signal_frequency'].apply(lambda f: not is_rfi(f, rfi_df, start_col, stop_col))
    df_clean = df[mask].reset_index(drop=True)

    return df_clean


def snr_Thresh(df, upper=100, lower=10):
    """
    snr threshold of dataframe from hits pkl.
    Inputs: dataframe, upper limit, lower limit
    """
    if 'signal_snr' not in df.columns:
        raise ValueError("Cannot find 'signal_snr' column.")

    return df.loc[(df['signal_snr']<upper) & (df['signal_snr']>lower)]


def dr_Thresh(df, upper, lower):
    """
    Makes a drift rate cut of dataframe from hits pkl (always excludes zero). 
    Inputs: dataframe, upper limit, lower limit
    """
    if 'signal_drift_rate' in df.columns:
        colname = 'signal_drift_rate'
    elif 'signal_driftRate' in df.columns:
        colname = 'signal_driftRate'
    else: 
        raise ValueError("Drift Rate Column not found.")
    
    return df.loc[(df[colname]<upper) & (df[colname]>lower) & (df[colname]!=0)]

def different_Source_Filter(df, group_cols=['signal_frequency', 'signal_driftRate'], target='sourceName'):
    """
    Filters a dataframe based on the group columns for uniqueness in the target column. 
    Ex: The default settings would find frequency-driftRate pairs and reject hits that occur in multiple sources

    This is an absolute filter that assumes drift rates and frequencies would be exactly the same
    without considering the possibility of float/rounding errors. 

    Inputs: 
        df: Hits/Stamps style dataframe
        group_cols: column names to group by
        target: target uniqueness column name

    """
    n_source = (
                df.groupby(group_cols)[target].transform("nunique")
                )
    df_unique_source = df.loc[n_source.eq(1)].copy()

    return df_unique_source

def find_unique_signals(df_target,
                        freq_tol = 3e-5,
                        freq_col = 'signal_frequency',
                        drift_tol = 0.0001,
                        drift_col = 'signal_driftRate',
                        target_col = 'sourceName',
                        ):
    # Normalize both dimensions by their tolerances so that
    # a Chebyshev distance of 1.0 = "within tolerance box"
    points_source = np.column_stack([
        df_target[freq_col].values / freq_tol,
        df_target[drift_col].values / drift_tol
    ])

    source_names = df_target[target_col].to_numpy()

    # Build KD-tree on df_other (done once, O(m log m))
    tree = cKDTree(points_source)

    # Query: find any neighbor within the tolerance box
    matches = tree.query_ball_point(points_source, r=1.0, p=np.inf)

    # Keep Rows that only appear at the same source for nearest neighbors. 
    keep = np.fromiter(
        (
            np.all(source_names[idx] == source_names[i])
            for i, idx in enumerate(matches)
        ),
        dtype=bool,
        count=len(df_target),
    )

    return df_target.loc[keep].copy()

    # # Rows with NO match in df_other
    # no_match_mask = np.array([len(m) == 0 for m in matches])
    # df_unique_rows = df_target[no_match_mask].reset_index(drop=True)

    # #Check if Incoherent in sourceName.unique()
    # # if incoherent:
    # #     # Rows with match in df_other. Return the common hits in both catalogs so it is easier to compare.
    # #     match_mask = ~no_match_mask
    # #     df_common_rows = df_target[match_mask].reset_index(drop=True)

    # #     other_indices = [m[0] if len(m) > 0 else None for m in matches]
    # #     other_common_rows = df_other.iloc[[other_indices[i] for i in np.where(match_mask)[0]]].reset_index(drop=True)

    # #     full_common_rows = pd.concat([df_common_rows, other_common_rows])

    # #     return df_unique_rows, full_common_rows

    # return df_unique_rows

