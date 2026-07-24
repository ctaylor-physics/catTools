import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, UTC
from astropy.time import Time
from scipy.spatial import cKDTree

from seticore.viewer import read_hits, read_stamps
from cosmic_database_analysis import sarfi

def find_nearby_signals(df1,
                        df2,
                        freq_col = "signal_frequency",
                        drift_col = "signal_driftRate",
                        freq_tol = 7.62939453125e-06,
                        drift_tol = 0.9239311256105938):
    """
    For COSMIC
    delta_nu_min = 7.62939453125e-06 (MHz)
    delta_dr_min = 0.9239311256105938 (Hz/s)
    """
    ### Normalize both dimensions by their tolerances so that a Chebyshev distance of 1.0 = "within tolerance box"
    # Reference Variable
    points_df1 = np.column_stack([
        df1[freq_col].values / freq_tol,
        df1[drift_col].values / drift_tol
    ])
    # Query Variable
    points_df2 = np.column_stack([
        df2[freq_col].values / freq_tol,
        df2[drift_col].values / drift_tol
    ])

    # Try without these first:
    # df1['tstart'].values,
    # df1['ra'].values,
    # df1['dec'].values,

    ### Build KD-tree on Reference Variable
    tree = cKDTree(points_df1)

    ### Query for nearest neighbors
    dist, inx = tree.query(points_df2, k=1)

    ### Copy and augment the recovered dataset

    matched = df2.copy()
    matched["matched_index"] = df1.index[inx]
    matched[f"matched_{freq_col}"] = df1.iloc[inx][freq_col].to_numpy()
    matched[f"matched_{drift_col}"] = df1.iloc[inx][drift_col].to_numpy()
    matched["nn_distance_scaled"] = dist

    matched[f"delta_{freq_col}"] = (
        matched[freq_col] - matched[f"matched_{freq_col}"]
    )

    matched[f"delta_{drift_col}"] = (
        matched[drift_col] - matched[f"matched_{drift_col}"]
    )

    try:
        matched[f"matched_freq_start"] = df1.iloc[inx]['start_freq_inx'].to_numpy()
        matched[f"matched_freq_end"] = df1.iloc[inx]['stop_freq_inx'].to_numpy()
    except KeyError:
        print('no frequency indices to be found')
        pass

    return matched


DIR = "/home/cat-work/work/SETI/incoherentHits"
fn_seed = "incoh_unique_signals_scan_ids.csv"
ross_csv = "20260624incoh_unique_signals_scan_ids.csv.stamp_rels.csv"

all_stamps_csv = "stamp_metadata_v1.csv"

incoh = pd.read_csv(os.path.join(DIR,fn_seed)) # from Chenoa
stamp_rels = pd.read_csv(os.path.join(DIR,ross_csv)) # limited qt that Ross gave me
all_stamps = pd.read_csv(os.path.join(DIR,all_stamps_csv)) # full db scrub

# change all_stamps colunm to mjd instead of timestamp
incoh.rename(columns={"ra_hours":"ra", "dec_degrees":"dec", "signal_drift_rate":"signal_driftRate"}, inplace=True)

all_stamps.rename(columns={"tstart":"tstart_unix"}, inplace=True)
all_stamps['tstart'] = all_stamps.tstart_unix.apply(lambda x: Time(datetime.fromtimestamp(x, tz=UTC)).mjd)
all_stamps['tstart_round'] = all_stamps['tstart'].round(11)
combo = incoh.merge(all_stamps,
                    how='left',
                    on=["tstart",
                        "ra",
                        "dec",
                        "signal_frequency",
                        "signal_driftRate"],
                    suffixes=("","_stampdb"),
                    indicator=True
)
both = combo[combo["_merge"] == "both"]
unmatched = combo[combo["_merge"] == "left_only"]
print(f"Hits found in Both, no edits {len(both)}")

# cKD_tree to find nearest neighboring points
matched = find_nearby_signals(all_stamps,incoh) 
perfect_match = matched[matched.nn_distance_scaled == 0.0] # len = 257
one_off = matched[matched.nn_distance_scaled == 1.0]
oo_freq = one_off[one_off.delta_signal_frequency != 0.0] # len = 16
oo_dr = one_off[one_off.delta_signal_driftRate != 0.0] # len = 2808

anti_matches = matched[matched.nn_distance_scaled > 1.0] # len = 866

# Next I need to combine the perfect matches and one_offs with the proper all_stamps index for locations
perfect_match_allstamps = pd.concat(
    [
        perfect_match.reset_index(drop=True),
        all_stamps.loc[perfect_match.matched_index].reset_index(drop=True).add_suffix('_allstamps')
    ],
axis=1
)


oo_freq_allstamps = pd.concat(
    [
        oo_freq.reset_index(drop=True),
        all_stamps.loc[oo_freq.matched_index].reset_index(drop=True).add_suffix('_allstamps')
    ],
axis=1
)

oo_dr_allstamps = pd.concat(
    [
        oo_dr.reset_index(drop=True),
        all_stamps.loc[oo_dr.matched_index].reset_index(drop=True).add_suffix('_allstamps')
    ],
axis=1
)
# rename cols for code on cosmic-s2
perfect_match_allstamps.rename(columns={"filepath_allstamps":"stamp_file_uri", "stamp_file_local_enum_allstamps":"stamp_file_local_enum"}, inplace=True)
oo_freq_allstamps.rename(columns={"filepath_allstamps":"stamp_file_uri", "stamp_file_local_enum_allstamps":"stamp_file_local_enum"}, inplace=True)
oo_dr_allstamps.rename(columns={"filepath_allstamps":"stamp_file_uri", "stamp_file_local_enum_allstamps":"stamp_file_local_enum"}, inplace=True)


# perfect_match_allstamps.to_csv(os.path.join(DIR,"incoh_us_perfect_match_allstamps.csv"))
# oo_freq_allstamps.to_csv(os.path.join(DIR,"incoh_us_oneoff_freq_allstamps.csv"))
# oo_dr_allstamps.to_csv(os.path.join(DIR,"incoh_us_oneoff_dr_allstamps.csv"))


outliers = matched[(matched.nn_distance_scaled != 0.0) & (matched.nn_distance_scaled != 1.0)]
# %% 
fig = plt.figure()
plt.scatter(
            outliers.matched_signal_driftRate,
            outliers.nn_distance_scaled
            )
plt.xlabel('Frequency (MHz)')
plt.ylabel('NN Distance')
plt.yscale('log')
plt.show()
# %%
