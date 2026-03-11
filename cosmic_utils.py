import os 
import sys 
import numpy as np
import pandas as pd

from astropy.time import Time
from astropy.table import Table, Column
from astropy.io import votable

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

def calculate_COSMIC_EIRP(distance):
    """
    Give a distance, get an estimated EIRP and ratio to Arecibo transmitter
    """
    S_vla = 13.92 * 10**(-26)# Jy = 1 * 10^-26 W/Hz/m^2
    BW = 8 # Hz
    dist = distance *  3.08567e16 # 1 pc = 3.085 * 10^16 m
    Fmin = S_vla*BW
    EIRP = 4 * np.pi * Fmin * dist**2 # Watts
    perArecibo = EIRP / (2e13) # Unitless

    print(f"Equivalent Isotropic Radiated Power: {EIRP} W")
    print(f"Ratio of EIRP/Arecibo: {perArecibo}")

    ### Maximum distance to detect Arecibo:
    # singleAreciboDistance = np.sqrt((2e13)/(4*np.pi*Fmin)) / (3.08567e16) # pc
    # print(f"Single Arecibo Distance: {singleAreciboDistance} pc")
    return

