import os 
import sys 
import numpy as np
import pandas as pd
from astropy.time import Time
import datetime as datetime
import matplotlib.pyplot as plt
from utils import calcFigSize,STYLE_PATH
import cosmic_utils
import utils
import argparse
from scipy.spatial import cKDTree
import time


# Practice source File From Chenoa
raw_fn = '/home/cat-work/work/SETI/K2-18b/K2-18b-hits_02-14-24-2.pkl'
single_day_fn = '/home/cat-work/work/SETI/K2-18b/K2-18b-hits_mjd60220.pkl'

def is_rfi(frequency, ranges):
    # Check if frequency is inside any RFI range
    # From Chenoa Tremblay k2-18b
    return np.any((frequency >= ranges['start_frequency']) & (frequency <= ranges['stop_frequency']))

### "Step 0"
def find_ObsId(df, outfile=None):
    """
    Incoming dataframes will be mixed between the target source, and the other sources from the same ObsId
     here we will split each of the sources from the others, then record the ObsIds associated with each source. 

    This is used as a preliminary step to identify the ObsIds that need to be collected from the
     cosmic database on cosmic-head

    Takes in dataframe with all sources and outfile name, writes a log file.
    """
    if not outfile:
        print('no outfile name provided')
        return
    
    ObsIds = df['observation_id'].unique()

    logfile = outfile+'.log'

    with open(logfile, 'w') as f:
        for id in ObsIds:
            ## Log the Ids and corresponding sources
            temp_df = df.loc[( df['observation_id'] == id )]
            f.write(f"{id:8s} {temp_df['source_name'].unique()}\n")

    return 

### "Step 1"
def rfi_Clean(df):
    """
    This function is a preliminary RFI screen for incoming data.
        It is from Chenoa's K2-18b code
    """
    ## S-band RFI From CRICKETS project
    # Load cleaned RFI CSV
    rfi_fn = '/home/cat-work/work/SETI/cosmicStellarHosts/Full_Crickets_CleanedUp.pkl'
    rfi_ranges = pd.read_pickle(rfi_fn).iloc[:, 3:] #drops kurtosis column

    # Build and apply mask
    mask = df['signal_frequency'].apply(lambda f: not is_rfi(f,rfi_ranges))
    df_clean = df[mask].reset_index(drop=True)

    return df_clean

### "Step 2"
def snr_Thresh(df, upper=100, lower=10):
    """
    snr threshold of dataframe from hits pkl.
    Inputs: dataframe, upper limit, lower limit
    """

    return df.loc[(df['signal_snr']<upper) & (df['signal_snr']>lower)]

### "Step 3"
def dr_Thresh(df, upper, lower):
    """
    Makes a drift rate cut of dataframe from hits pkl (always excludes zero). 
    Inputs: dataframe, upper limit, lower limit
    """
    return df.loc[(df['signal_drift_rate']<upper) & (df['signal_drift_rate']>lower) & (df['signal_drift_rate']!=0)]

### "Step 4"
## this might throw trouble depending on if the source_name is a string or not. 
def split_Sources(df, outfile=None, type = 'others', target=None):
    """
    Takes a csv dataframe for a given Observation ID and splits off sources for cross examination
     You can use this in "others" mode to just separate the target from the rest of the sources, 
     Or, in "all" mode to write dataframes for each source that was beamformed on,
     Incoherent Beam is always separated. 
    Writes the new dataframes at the outfile directory based on source name.
    """
    # Quick operations check
    if outfile is None:
        print('Please use proper grammar: no outfile name provided')
        return
    
    # Split off the data
    if type == 'all':
        for src in df.source_name.unique():
            tempdf = df.loc[df.source_name == src]
            tempdf.to_pickle(outfile+'_'+str(src)+'.pkl')
    else:
        if target is None:
            print('Please use proper grammar: specify a target gaia_id')
            return
        # target
        df.loc[df.source_name == target].to_pickle(outfile+'_'+str(target)+'.pkl')
        # incoherent
        df.loc[df.source_name == 'Incoherent'].to_pickle(outfile+'_incoherent.pkl')
        # others
        df.loc[(df.source_name == target) & (df.source_name == 'Incoherent')].to_pickle(outfile+'_others.pkl')
    
    return

        
### "Step 5"
def find_unique_signals(df_target, df_other, freq_tol = 3e-5, drift_tol = 0.0001, incoherent=False):
    # Normalize both dimensions by their tolerances so that
    # a Chebyshev distance of 1.0 = "within tolerance box"
    points_other = np.column_stack([
        df_other['signal_frequency'].values / freq_tol,
        df_other['signal_drift_rate'].values / drift_tol
    ])
    points_k218b = np.column_stack([
        df_target['signal_frequency'].values / freq_tol,
        df_target['signal_drift_rate'].values / drift_tol
    ])

    # Build KD-tree on df_other (done once, O(m log m))
    tree = cKDTree(points_other)

    # Query: find any neighbor within the tolerance box (Chebyshev / L-inf norm)
    # p=np.inf gives the L∞ norm, which is equivalent to checking both axes independently
    matches = tree.query_ball_point(points_k218b, r=1.0, p=np.inf, workers=-1)

    # Rows with NO match in df_other
    no_match_mask = np.array([len(m) == 0 for m in matches])
    df_unique_rows = df_target[no_match_mask].reset_index(drop=True)

    if incoherent:
        # Rows with match in df_other. Return the common hits in both catalogs so it is easier to compare.
        match_mask = ~no_match_mask
        df_common_rows = df_target[match_mask].reset_index(drop=True)

        other_indices = [m[0] if len(m) > 0 else None for m in matches]
        other_common_rows = df_other.iloc[[other_indices[i] for i in np.where(match_mask)[0]]].reset_index(drop=True)

        full_common_rows = pd.concat([df_common_rows, other_common_rows])

        return df_unique_rows, full_common_rows

    # df_unique_rows.to_pickle("k218b_unique_signals_c-band.pkl")
    return df_unique_rows

### Daughter function to process a single source
def process_source(dataframe, 
                   target,
                   out_dir=None,
                   snr_thresh = (10, 100),
                   dr_thresh=(-50, 50),
                   frequency_tol = 3e-5,
                   dr_tol = 1e-4):
    """
    This is a daughter function to process a single target source in an observation id dataframe. 

    Inputs:
        dataframe: ObservationHits dataframe for a given ObsId, or not
        target: Gaia source to compare against other beams in same observation
        out_dir: outgoing directory where the intermediate products will be stored. 
        snr_thresh: Signal-to-Noise threshold, default is minimum 10, maximum 100
        dr_thresh: Signal Drift Rate threshold, default is +/-50 Hz/s
        frequency_tol: tolerance to cross-match hits in frequency space, default is 3e-5 MHz
        dr_tol: tolerance to cross-match hits in drift rate space, default is 1e-4 Hz/s

    Returns: (none)
        Saves intermediate files along the way

    """

    if out_dir is None:
        raise ValueError('must provide an outgoing directory for data products')
    else: 
        os.makedirs(str(out_dir), exist_ok=True)

    filepath_prefix = f"{out_dir}/src{target}"

    ## Step 2
    # SNR Cut
    try:
        obs_snr = pd.read_pickle(filepath_prefix+f'_{snr_thresh[1]}snr.pkl')
    except:
        obs_snr = snr_Thresh(dataframe)
        obs_snr.to_pickle(filepath_prefix+f'_{snr_thresh[1]}snr.pkl')

    ## Step 3
    # Drift Rate Cut
    try:
        obs_snr_dr = pd.read_pickle(filepath_prefix+f'_{snr_thresh[1]}snr_{dr_thresh[1]}dr.pkl')
    except:
        obs_snr_dr = dr_Thresh(obs_snr, 50, -50)
        obs_snr_dr.to_pickle(filepath_prefix+f'_{snr_thresh[1]}snr_{dr_thresh[1]}dr.pkl')
    
    ## Step 4
    # Split
    target_df = obs_snr_dr.loc[( obs_snr_dr['source_name'] == target )]
    others_df = obs_snr_dr.loc[( obs_snr_dr['source_name'] != target )]
    incoherent_df = obs_snr_dr.loc[( obs_snr_dr['source_name'] == 'Incoherent' )]

    ## Step 5
    # Target Unique Signals with Others
    t_unique = find_unique_signals(target_df, others_df, freq_tol = frequency_tol, drift_tol = dr_tol)
    if len(t_unique) > 0:
        print(f"Number of unique signals found for target {target} vs. Others: {len(t_unique)}")
        t_unique.to_pickle(filepath_prefix+f'_{snr_thresh[1]}snr_{dr_thresh[1]}dr_unique.pkl')
    else: 
        print("No Unique 'Others' Signals Found!")

    ## Step 6
    # Target Unique/Common with Incoherent
    ti_unique, ti_common = find_unique_signals(t_unique, incoherent_df, freq_tol = frequency_tol, drift_tol = dr_tol, incoherent=True)
    if len(ti_unique) > 0:
        print(f"Number of unique signals found for target {target} vs. Incoherent: {len(t_unique)}")
        ti_unique.to_pickle(filepath_prefix+f'_{snr_thresh[1]}snr_{dr_thresh[1]}dr_unique_Incoh.pkl')
    elif len(ti_common) > 0:
        print(f"Number of unique signals found for target {target} vs. Incoherent: {len(t_unique)}")
        ti_common.to_pickle(filepath_prefix+f'_{snr_thresh[1]}snr_{dr_thresh[1]}dr_common_Incoh.pkl')
    else:
        print("No useful 'Incoherent' Signals Found!")

    return 

### Parent function
def process_Observation_Id(csv_filename, target=None):
    """
    This is intended to process a csv file for a single observation through the workflow.
    Inputs - 
        csv_filename: CSV filepath for a single obs id hits file
        target_source:
            - Single source to get "Target vs Incoherent vs Others"
            - An array/list of target sources will be iteratively done with the process above. 
            - 'All' or 'all' will process all the sources in an observation by collecting them independently
    """
    # Setup
    parent_dir = os.path.dirname(csv_filename)
    try:
        obs_df = pd.read_pickle(csv_filename)
    except:
        obs_df = pd.read_csv(csv_filename)

    ## Step 1
    #RFI Filter
    try:
        obs_rfi = pd.read_pickle(csv_filename)
    except:
        obs_rfi = rfi_Clean(obs_df)
        obs_rfi.to_pickle(f"{os.path.splitext(csv_filename)[0]}_rfi.pkl")

    incomingObsId = obs_df.observation_id.unique()
    incomingSources = obs_df.source_name.unique()
    
    # if len(incomingObsId) > 1:
    #     raise ValueError(f"more than one observation id contained: {incomingObsId}")
    
    ofilepath = parent_dir+f"/obsId_{incomingObsId[0]}"

    # Straighten out the target list
    if target is None:
        raise ValueError('Please provide a target for this query')
    elif isinstance(target, list): #list input
        target_source=target
    elif np.isin(target, ["All","all"]):
        target_source = list(obs_df.source_name.unique())
    elif isinstance(target,np.ndarray): #array input
        target_source=target.tolist()
    # elif isinstance(target,str): # single value
    #     target_source = [target]
    else:
        raise ValueError('i cannot use the structure of the incoming target variable')
    
    # Make sure the target is actually in this dataset
    if not np.isin(target_source, incomingSources).all():
        raise ValueError(f"missing source in requested search\n targets: {target_source}\n {np.isin(target_source, incomingObsId)}")
    
    # Go
    print(f"Processing sources: {target_source}")
    print('#########################################')
    t0 = time.perf_counter()
    for src in target_source:
        process_source(obs_rfi,
                        src,
                        out_dir = ofilepath,
                        snr_thresh = (10, 100),
                        dr_thresh = (-50, 50),
                        frequency_tol = 3e-5,
                        dr_tol = 1e-4)


        t_lap = time.perf_counter() - t0
        print(f"{t_lap:.5f} - Gaia ID {src} completed")

    print('done')
    return


### "Step X"
## Make sure planet isn't behind the star
## I think this is a later step in this case, right? 
## Because some of these will be multi-planet systems. 
# Inputs: Gaia_dr2_id, VLASS_t0, VLASS_t1
# 1. Get TOIs for all planets around star
# 2. Use astroquery to get period, T0, and transit duration
# 3. Propagate the period into the future to see if it overlaps with the input window
# Return: Boolean describing if the planet is close to secondary transit, start and end time of nearest secondary transit(?) 

### "Step Y"
## Chenoa here would exclude things that repeat on different days at the same freq
## For this exercise we will do the opposite

### "Step Z"
## Get the relevant stamp files. 



def main(args):
    ### Verify inputs
    if not os.path.isfile(args.filename):
        raise OSError('file does not exist')

    ### First set of ObsIds are here:
    # ObsId_files = "/home/cat-work/work/SETI/cosmicStellarHosts/databaseHits/observationIds_10pc/*.csv"

    # practice_File = "/home/cat-work/work/SETI/cosmicStellarHosts/databaseHits/observationIds_10pc/cat_ObsIdHits_v1_31524.csv"
    # trial_target = '2824770686019003904' # one of two gaia ids in this file
    # trial_target2 = ['2824770686019003904', '2824770686019004032']
    # obs31524 = pd.read_csv(practice_File)

    process_Observation_Id(args.filename, args.target)

    return 


if __name__ == "__main__":
    ### Since VLASS observations are continuous slews, you can reject based on other beams formed in the Obs.
    ## I.E. a signal must be unique to a single beam pointing to be interesting, if it is in other beams then it is local (RFI)
    
    ### The direction that I am going with this is to have it be command line operable with argparse (example below from commissioning)
    ## Where the inputs are something like a csv/pkl file of hits for a given ObsId/Target, with whatever limits I need
    ## and it returns a csv/pkl file containing unique signals to be investigated in the stamps collection on the Berkeley cluster
    
    ### Here is what I got from Talon's queries for me:
    ## This is done from the cosmic head node, using the env called "database_v1"
    ## > cosmicdb_inspect ObservationHit -w source_name in_csv ./stellarHosts_idsList_decCut.csv:Gaia_ID --pandas-output-filepath ./craig_gaia_hits_v1.csv
    ## Data location: /home/cosmic/dev/COSMIC-VLA-StampInspection/craig_gaia_hits_v1.csv


    parser = argparse.ArgumentParser(
            description='This code is designed to apply a set of filters to incoming COSMIC-VLASS data to identify interesting candidate signals. ',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    parser.add_argument('filename', type=str,
                        help='csv filename to process')
    # parser.add_argument('-o', '--observation_id', type=str,
    #                     help='VLASS Observation Id to be processed in this csv.')
    parser.add_argument('-t', '--target', nargs='+', type=str, default=None, 
                        help='target or a space delimited list of targets to be processed or you can specify "all" to process each source in the csv independently (Use with caution as many Obs Ids have lots of sources within!). ')

    args = parser.parse_args()
    print(args)
    main(args)

    # df = main(do_plot=False)


    ## For specifically mutually exclusive ("only choose one") options
    # wgroup = parser.add_mutually_exclusive_group(required=False)
    # wgroup.add_argument('-t', '--bartlett', action='store_true',
    #                     help='apply a Bartlett window to the data')
    # wgroup.add_argument('-b', '--blackman', action='store_true',
    #                     help='apply a Blackman window to the data')
    # wgroup.add_argument('-n', '--hanning', action='store_true',
    #                     help='apply a Hanning window to the data')






