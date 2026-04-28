import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cosmic_utils import look_for_combs, log_with_pandas
from scipy.stats import median_abs_deviation
from seticore import viewer
from cosmic_database_analysis import sarfi

def snr_windows(data, mask): #mask should be stamp.signal_mask()
    """
    This would take in a stamp and compute the edge SNR, core SNR, and signal.
    A large discrepancy between the edge and core SNR could indicate that there is comb-like structure (RFI)
    Input:
        stamp real array data of form [antenna][timestep][channel] averaged in pol and real/imag axes
        stamp mask [stamp.signal_mask()]
    Output:
        tuple of (edge_snr, core_snr, signal strength)
    """
    signal = (data * mask).sum()

    ## Edge
    # Calculate the noise based on the first and last 20 column sums
    left_column_sums = data[:, :20].sum(axis=1)
    right_column_sums = data[:, -20:].sum(axis=1)
    column_sums = np.concatenate((left_column_sums, right_column_sums))
    mean = column_sums.mean()
    std = column_sums.std()
    edge_snr = (signal - mean) / std

    ## Core
    # Calculate the noise based on the core 40 columns around the masked signal
    inv_signal = (data * ~mask)
    middle = int(data.shape[-1]/2)
    core_sum = inv_signal[:, middle-20:middle+21].sum(axis=1) # assumes in the absolute minimum that the 'middle' channel is masked
    core_mean = core_sum.mean()
    core_std = core_sum.std()
    core_snr = (signal - core_mean) / core_std

    return (edge_snr, core_snr, signal)

def find_outlier_antennas(obs_info, out_dir = None):

    """
    For a single stamp this function makes a comparison of the SNR measured between each antenna
        If one antenna has snr 5 sigma above mean -> reject
        elif =<5 antennas have inverted snrs (core>edge) -> I dont think this is helpful unless the box is bigger and normalized?
        else flag antennas with combs as such and mark good antennas without. 
    input:
      data for a stamp, all antennas
    output: 
        Antenna score: 0 = reject for overpowered
                       1 = combs detected, questionable calibration likely
                       2 = signal possibly detected
    """

    ## Fetch Stamp
    stamps_gen = viewer.read_stamps(obs_info.stamp_file_uri, find_recipe=True)
    for index, stamp in enumerate(stamps_gen):
        if index == obs_info.stamp_file_local_enum:
            assert(stamp != None)
            assert(stamp.recipe != None)
            break

    ## Get some filename information
    srcName = obs_info.source_name
    print(f"stamp source name: {srcName}")
    out_fn = os.path.join(out_dir,
                          f"{srcName}_id{obs_info.id}_{round(obs_info.signal_frequency,5)}MHz")
    ## ant_pow[antenna][timestep][channel] Sums along pol and real/image axes
    ant_pow = np.square(stamp.real_array()).sum(axis=(2, 4)).transpose(2,0,1) 
    sig_mask = stamp.signal_mask()
    antenna_titles = np.array(stamp.recipe.antenna_names)

    ## Get some statistics on the data for flagging antennas
    antdata_snrs = np.array([ snr_windows(ant, sig_mask) for ant in ant_pow ])

    ## Average, MAD, and edge comparison
    mean_snr = np.mean(antdata_snrs, axis=0) # mean along antennas
    std_snr = np.std(antdata_snrs, axis=0) #std along antennas
    mad_snr = median_abs_deviation(antdata_snrs, axis=0)
    comp_edge = np.subtract.outer(antdata_snrs[:,0],antdata_snrs[:,0]) #outer comparison of edge_snrs
    # comp_edge_core = antdata[:,0] - antdata

    ## Single Antenna Above the Lot 25 times the median absolute deviation is my initial thoughts from the voyager example, might fail heroically. 
    # This will also give an exceedingly sharp spike in the log_signal and a decaying logarithm in cepstrum
    single_bad_ants = ( comp_edge > (25 * mad_snr[0] )).sum(axis=1)

    ## Initial indication is that we maybe can find the cepstrum peaks above say ~50?
    ## In the fuzzy looking antenna data there are lots that extend above even 100 
    ## Highest SNR for Voyager only gets to ~30 in amplitude on the cepstrum.
    antenna_score = np.array([ look_for_combs(ant, name, thresh=50, cepstrum_offset=5, out_fn=out_fn) for ant, name in zip(ant_pow, antenna_titles)  ])

    ## Signal score: 1 = no extreme outlier antennas, 0 = hot antenna detected
    signal_score = True

    ## Anomalous Antenna In Set
    if np.any(single_bad_ants):
        signal_score = False
        antenna_score[np.argwhere(single_bad_ants)[0]] = 0
        print(f"hot antenna detected, plotting")

    sarfi_score = sarfi.is_SARFI(stamp)


    if out_dir is not None:
        # segregate antennas
        bad = np.where(antenna_score == 0)[0]
        comb = np.where(antenna_score == 1)[0]
        ok = np.where(antenna_score == 2)[0]
        ticklocs = np.arange(len(antenna_titles))

        # plot 
        fig = plt.figure(layout='constrained')
        plt.suptitle(f"Target: {stamp.stamp.sourceName}")
        plt.title(f"Good Antennas:{len(np.argwhere(antenna_score))}     OutlierAntenna:{signal_score}", fontsize=8)
        plt.scatter(ticklocs[bad], antdata_snrs[bad, 0], label='bad') # Edge SNR
        plt.scatter(ticklocs[comb], antdata_snrs[comb, 0], label='comb detected') # Edge SNR
        plt.scatter(ticklocs[ok], antdata_snrs[ok, 0], label='OK') # Edge SNR
        # plt.scatter(antenna_titles, antdata_snrs[:, 1], label='core', alpha = 0.3) # Core SNR
        plt.ylabel("SNR")
        plt.xlabel("VLA Antenna")
        plt.xticks(ticks=ticklocs, labels = antenna_titles, rotation=45)
        plt.legend()
        plt.savefig(out_fn + f"_antenna_score.png")
        plt.close(fig)

        stamp.show_antennas(title=f"{srcName}", save_to = out_fn + "_show_antennas.png")
    
    outliers = {"source_name": srcName,
                "id": obs_info.id,
                "freq": obs_info.signal_frequency,
                "antenna_score": antenna_score,
                "total_score": antenna_score.sum(),
                "signal_score": signal_score,
                "sarfi_score": sarfi_score,
                "stamp_file_uri": obs_info.stamp_file_uri, # I should also add the show_antennas filename os.path.basename()
                }
    return outliers #number of plausible antennas, score of signal
"""
def practice_data():
    
    # Data for the stamps is stored: 
    #    data[timestep][channel][polarization][antenna][real or imag] = pracStamp.real_array()

    #The test stamps are 33: Voyager signal in all ants
    #                    34: Ant 1 has a bright signal but the others not-so-much, also a 24, 26, 28 look bad
    #                    25: Obvious comb-like features in antenna 24, 26

    
    # Voyager 14-Aug-2025 21:57:29.3 - 21:58:02.9 
    DIR = '/home/cat-work/work/SETI/stampTutorial/voyager'
    voyager_fn = "TCOS0001_sb49105488_1_1.60901.90559458333.4.1.AC.C480.0000.raw.seticore.0000.stamps" 
    STAMPS_PATH = os.path.join(DIR, voyager_fn)
    DIAG_PATH = os.path.join(DIR, 'diagnostic_plots')
    os.makedirs(DIAG_PATH, exist_ok=True)


    print(f"Reading stamps at: {STAMPS_PATH}")
    paths = [STAMPS_PATH,STAMPS_PATH,STAMPS_PATH]
    local_enums = [33,34,25]
    hit_ids = [111,222,333]
    log_fn = os.path.join(DIAG_PATH, "stamp_diagnostics.csv")


    for path,enum,id in zip(paths,local_enums,hit_ids):
        outliers = find_outlier_antennas(stamp_path = path,
                                        stamp_enum = enum,
                                        hit_enum = id,
                                        out_dir = DIAG_PATH)
        log_with_pandas(outliers,
                        log_fn)
"""

def main():
    # This is on cosmic-storage-1
    DIR = '/srv/cosmicfs0/scratch/ctaylor/'
    # STAMPS_FN1 = os.path.join(DIR, "uniqueFinalHits10pc_scan_ids_v3.storage1.stamp_rels.csv")
    # STAMPS_FN2 = os.path.join(DIR, "uniqueFinalHits10pc_scan_ids_v3.storage2.stamp_rels.csv")
    OBS_INFO_FN = os.path.join(DIR, "uniqueFinalHits10pc_storage1.csv")

    DIAG_PATH = os.path.join(DIR, 'diagnostic_plots')
    os.makedirs(DIAG_PATH, exist_ok=True)
    log_fn = os.path.join(DIAG_PATH, "stamp_diagnostics.csv")


    #stamps1 = pd.read_csv(STAMPS_FN1)
    # stamps2 = pd.read_csv(STAMPS_FN2)
    obs_info = pd.read_csv(OBS_INFO_FN)

    # for path,enum_s,enum_h in zip( stamps1.stamp_file_uri.iloc[1:2],
    #                                stamps1.stamp_file_local_enum.iloc[1:2],
    #                                stamps1.hit_file_local_enum.iloc[1:2] ):

    for i, row in obs_info.iterrows():
        outliers = find_outlier_antennas(row, out_dir = DIAG_PATH)
        log_with_pandas(outliers,log_fn)


if __name__ == "__main__":
    main()


# The vision here is to load in all the stamps in the first step,
# then write a log file that has information about every hit.
# Observations with the highest values of antenna score are the most likely candidates to find a signal
# something like a csv output with columns like
# Stamp file, antenna_score, total_score, outlier, 

### This section will ingest a list of the files and stamps that we need to collect,
##  then the iterator will retrieve them one by one and run the pipeline on them. 




# pracStamp.show_classic_incoherent(show_signal=True)
# stamp.show_antennas(title=f'Voyager?') # This figure is too big
