import os 
import sys 
import numpy as np
import pandas as pd
from astropy.time import Time
import datetime as datetime
import matplotlib.pyplot as plt
from utils import calcFigSize,STYLE_PATH

# Source File From Chenoa
raw_fn = '/home/cat-work/work/SETI/K2-18b-hits_02-14-24-2.pkl'

single_day_fn = '/home/cat-work/work/SETI/K2-18b-hits_mjd60220.pkl'

### takes in a dataframe  
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

### takes an snr threshold of dataframe from hits pkl.
def snrThresh(df, upper=100, lower=10):
    return df.loc[(singleDay['signal_snr']<upper) & (singleDay['signal_snr']>lower)]

### takes a drift rate cut of dataframe from hits pkl. Excludes zero. 
def drThresh(df, upper, lower):
    return df.loc[(singleDay['signal_drift_rate']<upper) & (singleDay['signal_drift_rate']>lower) & (singleDay['signal_drift_rate']!=0)]

### This was created using the function above
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










