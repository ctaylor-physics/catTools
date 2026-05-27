import os 
import sys 
import numpy as np
import pandas as pd

from astropy.time import Time
from astropy.table import Table, Column
from astropy.io import votable

from scipy.signal import find_peaks

import datetime as datetime

import matplotlib.pyplot as plt


### takes in a pickle filename, clips out data for the input mjd
def dayClip(fn, mjd_date):
    # Read in k2-18b dataset
    dataframe = pd.read_pickle(fn)

    # Slice the first day of observing (I already know the first day is 60220)
    singleDay = dataframe.loc[(dataframe['tstart'] > (mjd_date-1)) & (dataframe['tstart'] < mjd_date)]
    cols = singleDay.columns

    # Add datetime column
    time = Time(singleDay['tstart'].values, format='mjd', scale='utc')
    singleDay['tstart_datetime'] = time.to_datetime()

    # Save out Progress to avoid loading the whole meatball 
    outfn = '/home/cat-work/work/SETI/K2-18b-hits_mjd' + str(int(mjd_date))+'.pkl'
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


