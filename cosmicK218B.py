import os 
import sys 
import numpy as np
import pandas as pd
from astropy.time import Time
import datetime as datetime
import matplotlib.pyplot as plt
from utils import calcFigSize,STYLE_PATH
import cosmic_utils

# Source File From Chenoa
raw_fn = '/home/cat-work/work/SETI/K2-18b-hits_02-14-24-2.pkl'

single_day_fn = '/home/cat-work/work/SETI/K2-18b-hits_mjd60220.pkl'

### "Step 0"
## Separate K2-18b from K2-18 (host star)


### "Step 1/2"
def drThresh(df, upper, lower):
    """
    Makes a drift rate cut of dataframe from hits pkl (always excludes zero). 
    Inputs: dataframe, upper limit, lower limit
    """
    return df.loc[(df['signal_drift_rate']<upper) & (df['signal_drift_rate']>lower) & (df['signal_drift_rate']!=0)]

### "Step 3"
def snrThresh(df, upper=100, lower=10):
    """
    snr threshold of dataframe from hits pkl.
    Inputs: dataframe, upper limit, lower limit
    """

    return df.loc[(df['signal_snr']<upper) & (df['signal_snr']>lower)]

### "Step 4"
def split_and_unique(df, source_names):
    """
    Unique combined signals from K2-18 and K2-18b. 
    Takes in dataframe with all sources, splits data according to list of targets desired
    """
    for src in source_names:
        temp_df = plotData.loc[( plotData['source_name'] == src )]


    return # incomplete
        

### "Step 4"
## Make sure planet isn't behind the star

### "Step 5"
## Chenoa here would exclude things that repeat on different days at the same freq
## For this exercise we will do the opposite


def main():
    ### This was created using dayClip from cosmic_utils
    singleDay = pd.read_pickle(single_day_fn)

    ### Remove SNR values above 100?
    singleDay_snr = snrThresh(singleDay, 100, 10)
    singleDay_snr_dr = drThresh(singleDay_snr, 1.9, -1.9)

    ### This captures a single time step in the file, here the first one. 
    singleStep = singleDay_snr_dr.loc[singleDay_snr['tstart_datetime']<'2023-10-03 19:50:00']

    ### Plot Style for Publication
    # figs = calcFigSize(name="CQG",columns='onecol')
    # plt.style.use(STYLE_PATH)

    ## Reassign this value as we make cuts
    plotData = singleDay_snr_dr

    ### Lets make a suite of plots to characterize the remaining data

    title = 'Test Plot of K2-18b'

    fig, ax = plt.subplot_mosaic("""
                                AB
                                """, sharex=False, sharey=False)
    fig.suptitle(title)

    ax['A'].set_title('frequency-snr')
    ax['A'].scatter( plotData['signal_frequency'] ,  plotData['signal_snr'], c='b')
    ax['A'].set_xlabel("Signal Frequency(MHz)")
    ax['A'].set_ylabel("Signal SNR")

    ax['B'].set_title('driftrate-snr')
    ax['B'].scatter( plotData['signal_drift_rate'] ,  plotData['signal_snr'], c='k')
    ax['B'].set_xlabel("Signal Drift Rate (Hz)")
    ax['B'].set_ylabel("Signal SNR")

    plt.show()
    return plotData

if __name__ == "__main__":
    plotData = main()


    ### Since the VLASS observations are continuous slews, you can reject signals that are present or not present in other antennas.
    ## Another method could be to compare the 'hits' between sources that are observed in the same drifting track or over some fixed duration.
    ## This is similar to the uniqueness filter that Chenoa has used for scans that use k2-18 or K2-18b at the phase center. 
    







