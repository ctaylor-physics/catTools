import os 
import sys 
import numpy as np
import pandas as pd
from astropy.time import Time
import datetime as datetime
import matplotlib.pyplot as plt
from utils import calcFigSize,STYLE_PATH

# Source File From Chenoa
fn = '/home/cat-work/work/SETI/K2-18b-hits_02-14-24-2.pkl'


def dayClip(fn):

    # Read in k2-18b dataset
    dataframe = pd.read_pickle(fn)

    # # Here is everything that is in it
    # datasets = dataframe.columns
    # print(datasets)

    # Lets make a suite of plots to characterize the 
    # Slice the first day of observing (I already know the first day is 60220)
    singleDay = dataframe.loc[dataframe['tstart'] < 60221,:]
    cols = singleDay.columns

    # Add datetime column
    time = Time(singleDay['tstart'].values, format='mjd', scale='utc')
    singleDay['tstart_datetime'] = time.to_datetime()

    # Save out Progress to avoid loading the whole meatball 
    singleDay.to_pickle('/home/cat-work/work/SETI/K2-18b-hits_mjd60220.pkl')
    return


### This was created using the function above
singleDay = pd.read_pickle('/home/cat-work/work/SETI/K2-18b-hits_mjd60220.pkl')

### Remove SNR values above 100?
snrThresh = singleDay.loc[singleDay['signal_snr']<100]

singleStep = snrThresh.loc[snrThresh['tstart_datetime']<'2023-10-03 19:50:00']

### Plot Style for Publication
# figs = calcFigSize(name="CQG",columns='onecol')
# plt.style.use(STYLE_PATH)

## Reassign this value as we make cuts
plotData = singleStep
title = 'Test Plot of K2-18b'

fig, ax = plt.subplot_mosaic("""
                             BD
                             """, sharex=False, sharey=False)
fig.suptitle(title)
# ax['A'].set_title('time-frequency')
# ax['A'].scatter( plotData['tstart_datetime'],  plotData['signal_frequency'], c='r')
# ax['A'].set_xlabel("Time (MJD)")
# ax['A'].set_ylabel("Signal Frequency (MHz)")

ax['B'].set_title('frequency-snr')
ax['B'].scatter( plotData['signal_frequency'] ,  plotData['signal_snr'], c='b')
ax['B'].set_xlabel("Signal Frequency(MHz)")
ax['B'].set_ylabel("Signal SNR")

# ax['C'].set_title('time-snr')
# ax['C'].scatter( plotData['tstart_datetime'] ,  plotData['signal_snr'], c='g')
# ax['C'].set_xlabel("Time (dt)")
# ax['C'].set_ylabel("Signal SNR")

ax['D'].set_title('driftrate-snr')
ax['D'].scatter( plotData['signal_drift_rate'] ,  plotData['signal_snr'], c='k')
ax['D'].set_xlabel("Signal Drift Rate (Hz)")
ax['D'].set_ylabel("Signal SNR")


plt.show()










