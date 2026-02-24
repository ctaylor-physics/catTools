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


### Testing function made with shat to find candidates
def candidate_bins_intersection(
    df: pd.DataFrame,
    sources: list[str],
    freq_tol: float,
    drift_tol: float,
    *,
    freq_col: str = "signal_frequency",
    drift_col: str = "signal_drift_rate",
    source_col: str = "source_name",
    edge_guard: bool = True,
):
    """
    Returns candidate (freq_bin, drift_bin) cells that contain at least one row
    from every source in `sources`, using tolerances as bin widths.

    If edge_guard=True, also considers neighbor bins (±1) to reduce misses near bin edges.
    """
    sub = df[df[source_col].isin(sources)].copy()

    # Bin indices (integer grid). floor is stable; rounding can be used too but floor is predictable.
    fbin = np.floor(sub[freq_col].to_numpy() / freq_tol).astype(np.int64)
    dbin = np.floor(sub[drift_col].to_numpy() / drift_tol).astype(np.int64)

    sub["_fbin"] = fbin
    sub["_dbin"] = dbin

    if edge_guard:
        # Expand each row to 9 neighbor cells: (fbin+df, dbin+dd) for df,dd in {-1,0,1}
        # This is the main cost, but it's still vectorized and often OK if you filter early by sources.
        shifts = np.array([(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)], dtype=np.int64)

        # Repeat rows 9x without python loops
        sub9 = sub.loc[sub.index.repeat(len(shifts))].copy()

        # Apply shifts
        rep = np.tile(shifts, (len(sub), 1))
        sub9["_fbin"] = sub9["_fbin"].to_numpy() + rep[:, 0]
        sub9["_dbin"] = sub9["_dbin"].to_numpy() + rep[:, 1]
        work = sub9
    else:
        work = sub

    # For each grid cell, count unique sources present
    counts = (
        work.groupby(["_fbin", "_dbin"], sort=False)[source_col]
            .nunique()
            .reset_index(name="n_sources")
    )

    # Candidate cells: all requested sources present
    candidates = counts[counts["n_sources"] == len(sources)][["_fbin", "_dbin"]]

    return candidates

