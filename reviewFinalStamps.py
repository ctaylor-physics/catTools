import os
import sys
import ast
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn
from utils import STYLE_PATH, calcFigSize, get_colors

import glob
from cosmic_utils import look_for_combs, log_with_pandas
from scipy.stats import median_abs_deviation
from scipy import signal

from seticore import viewer
from cosmic_database_analysis import sarfi

from blri.dsp import upchannelise

PLOT_PATH = '/home/cat-work/work/SETI/cosmicStellarHosts/databaseHits/observationIds_10pc'
STAMP_PATH = '/home/cat-work/work/SETI/cosmicStellarHosts/databaseHits/observationIds_10pc/final_10pc_candidates'

def dedrift_stamp(stamp, upchannelisation_rate = 1, mask=False): 
    """
    Input:
        a stamp array as [antenna, channel, timestep, pol]
    and an upchannelization optionally
    Returns:
        the dedrifted stamp
    """
    if mask is True:
        sig_mask = stamp.signal_mask()
        stamp_power = np.square(stamp.real_array()).sum(axis= 4) * sig_mask[:,:,np.newaxis, np.newaxis]
        stamp_power = stamp_power.transpose(3,1,0,2)
    else:
        stamp_power = np.square(stamp.real_array()).sum(axis= 4).transpose(3,1,0,2) 

    stamp_shape = stamp_power.shape

    # compute the dedispersion maths 
    if stamp.stamp.signal.driftRate != 0.0:
        drift_rate = stamp.stamp.signal.driftRate
        foff = stamp.stamp.foff

        print(f'dr:{drift_rate}, foff:{foff}, dt:{stamp.stamp.tsamp}')
        df = abs(foff) * 1e6 # / upchannelisation_rate
        dt = stamp.stamp.tsamp # * upchannelisation_rate

        offset_bins = (np.round(abs(drift_rate) * np.arange(stamp_shape[2]) * dt / df)).astype(int)
        offset_bins = offset_bins - offset_bins[0]
        
        # Test points
        # print((abs(drift_rate) * np.arange(stamp_shape[2]) * dt / df))
        # print(offset_bins)

        if np.sign(drift_rate) == np.sign(foff):
            shift_direction = -1
        else:
            shift_direction = 1
        
        ddStamp = np.zeros_like(stamp_power)
        for i,dist in enumerate(offset_bins):
            ddStamp[:,:,i,:] = np.roll(stamp_power[:,:,i,:], dist*shift_direction, axis=1)
         
    else:
        ddStamp = stamp_power

    # upchannelise if desired
    if upchannelisation_rate != 1:
        ddStamp = upchannelise(ddStamp,upchannelisation_rate)
    
    return ddStamp



def correlate_stamp(ddStampData):
    """
    From seticore/viewer.py:
    Returns a data[2 * time, antenna] array of complex values.
    We have two points for each timestep because we have multiple polarizations.
    Then Return a matrix where [i, j] is the correlation coefficient between
    antennas i and j.
    
    This version is a combination of masked_antenna_vales() and correlations()
    that I have adapted to take a masked version of dedrift_stamp(mask=True)
    Returns masked_ant_values (for testing) and cross-corrs
    """
    antennas, channels, timesteps, pols = ddStampData.shape
    print(antennas, channels, timesteps, pols)
    ddStampData = ddStampData.transpose(2,1,3,0)
    mav = []
    for ant in range(antennas):
        pol0 = ddStampData[:,:,0,ant]
        pol1 = ddStampData[:,:,1,ant]
        mav.append(np.concatenate((pol0[pol0 != 0],pol1[pol1 != 0])))
    mav = np.array(mav).transpose(1,0)

    ccorrs = np.zeros((antennas, antennas))
    for i in range(antennas):
            for j in range(i, antennas):
                vi = mav[:, i]
                vj = mav[:, j]
                cc = abs(np.vdot(vi, vj)) / (np.linalg.norm(vi) * np.linalg.norm(vj))
                ccorrs[i, j] = cc
                ccorrs[j, i] = cc
    return mav, ccorrs



# Next steps:
# 1. Are the start and stop frequencies the same in all antenna
# 2. Does the amplitude vs time stay relatively consistent across all antenna
# 3. De-drift the signals and coherently add the pulses together. Does that make it look weird?
# 4. If any survive, lets try to image them? 

def main():
    #args: filename
    filename = os.path.join(PLOT_PATH, 'uniqueFinalHits10pc_shortlist.csv')

    ## Step 1: Load Data and Assemble Stamps
    obs_info_final = pd.read_csv(filename)

    final_stamps = []
    for i,row in obs_info_final.iterrows():
        ## Fetch Stamp
        stampfn = os.path.join(STAMP_PATH, os.path.basename(row.stamp_file_uri_oi))
        stamps_gen = viewer.read_stamps(stampfn, find_recipe=True)
        for index, stamp in enumerate(stamps_gen):
            if index == row.stamp_file_local_enum:
                assert(stamp != None)
                assert(stamp.recipe != None)
                final_stamps.append(stamp)
                break

    ## Step 2: Check Correlation Score
    # Each stamp gets a mean, median, std of correlations    
    corr_results = np.zeros((len(final_stamps),3))

    for i,stamp in enumerate(final_stamps):
        # get observation info for stamp
        row_info = obs_info_final.iloc[i]
        ant_score = np.fromstring(row_info.antenna_score.strip('[]'), sep = ' ')
        flag_rc = np.where(np.logical_or(ant_score == 0, ant_score == 1))
        
        # Get correlation
        corr = stamp.correlations()
        
        # Flag Correlation
        np.fill_diagonal(corr, np.nan)
        corr[flag_rc, :] = corr[:, flag_rc] = np.nan

        # Store Statistics
        corr_results[i,:] = np.nanmean(corr), np.nanmedian(corr), np.nanstd(corr)

        ## Plot images of this:
        # print(f"Gaia ID {obs_info_final.source_name_oi.iloc[0]} ({stamp.stamp.signal.frequency} MHz)")
        # print(f"    Mean Correlation: {np.nanmean(corr)}\n    STD Correlation: {np.nanstd(corr)}\n")
        # fig = plt.figure()
        # plt.title(f"Id: {row_info.id} \n Source: {row_info.source_name_oi})
        # plt.imshow(corr, origin='lower')
        # plt.show()


    # Looking for correlation score: mean, median > 0.5 & std < 0.1
    obs_info_final[['corr_mean', 'corr_median','corr_std']] = corr_results

    corr_filter = obs_info_final[(obs_info_final['corr_mean'] > 0.5) & 
                                 (obs_info_final['corr_median'] > 0.5) & 
                                 (obs_info_final['corr_std'] < 0.1)]
    if len(corr_filter) == 0:
        print('No Correlated Signals Detected!')
        return
    
    ## Step 3: Does the time series follow a cut of the beam profile?
    # get remaining stamps
    remaining_stamps = final_stamps[-1]
    dd_timeseries = []
    for stamp in remaining_stamps:
        ddStamp = np.abs(dedrift_stamp(stamp,1, mask=False))
        timeseries = ddStamp.sum(axis=(1,3)) # [antenna, timestamp]
        dd_timeseries.append(timeseries)

    # correlate with the S-Band VLA Beam pattern, this might just be for rigor
    timecorr = signal.correlate()
    peak_corr = np.argmax()

    # plot the antenna responses, overlay the beam cut that was taken
    # some of this is setup in vlaBeam.py
    # still would need to get the beam phase center or source offset, duration in beam for scan
    # then plot over the corresponding shape 

    return


obs_info_final = pd.read_csv(os.path.join(PLOT_PATH, 'uniqueFinalHits10pc_shortlist.csv'))

final_stamps = []
for i,row in obs_info_final.iterrows():
    ## Fetch Stamp
    stampfn = os.path.join(STAMP_PATH, os.path.basename(row.stamp_file_uri_oi))
    stamps_gen = viewer.read_stamps(stampfn, find_recipe=True)
    for index, stamp in enumerate(stamps_gen):
        if index == row.stamp_file_local_enum:
            assert(stamp != None)
            assert(stamp.recipe != None)
            final_stamps.append(stamp)
            break

# %%
### Voyager Test File
# Voyager 14-Aug-2025 21:57:29.3 - 21:58:02.9 
VDIR = '/home/cat-work/work/SETI/stampTutorial/voyager'
voyager_fn = "TCOS0001_sb49105488_1_1.60901.90559458333.4.1.AC.C480.0000.raw.seticore.0000.stamps" 
VSTAMPS_PATH = os.path.join(VDIR, voyager_fn)
vstamps = []
vstamps_gen = viewer.read_stamps(VSTAMPS_PATH, find_recipe=True)
for index, stamp in enumerate(vstamps_gen):
    if index in {25,33,34}:
        assert(stamp != None)
        assert(stamp.recipe != None)
        vstamps.append(stamp)
        #break
# %%

stamp = final_stamps[3]
stamp = vstamps[1]


# Get arrays
stamp_power = np.square(stamp.real_array()).sum(axis= 4).transpose(3,1,0,2)
ddStamp = np.abs(dedrift_stamp(stamp,1, mask=False))
ddStamp2 = np.abs(dedrift_stamp(stamp,2, mask=False))

v_timeseries = ddStamp.sum(axis=(1,-1))
P_center = np.array([0.01728   , 0.03699497, 0.05682462, 0.07757318, 0.09987174,
                    0.12419309, 0.15086616, 0.18008982, 0.21194622, 0.24641365,
                    0.28337887, 0.32264885, 0.36396212, 0.40699951, 0.45139445,
                    0.49674269, 0.54261154, 0.5885486 , 0.63408999, 0.67876801,
                    0.72211837, 0.7636868 , 0.80303528, 0.83974764, 0.87343467,
                    0.90373883, 0.93033823, 0.95295035, 0.97133501, 0.98529701,
                    0.99468814, 0.99940875, 0.99940875, 0.99468814, 0.98529701,
                    0.97133501, 0.95295035, 0.93033823, 0.90373883, 0.87343467,
                    0.83974764, 0.80303528, 0.7636868 , 0.72211837, 0.67876801,
                    0.63408999, 0.5885486 , 0.54261154, 0.49674269, 0.45139445,
                    0.40699951, 0.36396212, 0.32264885, 0.28337887, 0.24641365,
                    0.21194622, 0.18008982, 0.15086616, 0.12419309, 0.09987174,
                    0.07757318, 0.05682462, 0.03699497, 0.01728   ])
fig = plt.figure()
plt.title('target timeseries from Voyager tracking through phase center')
plt.plot(v_timeseries[0,:] * P_center)
plt.xlabel('Time (0-8s)')
plt.ylabel('Power')
plt.show()

# Stamp power * mask
sp_masked = stamp_power.sum(axis=(-1)).transpose(0,2,1) * stamp.signal_mask()[np.newaxis, :,:]


### Antenna Cross-Correlations Investigation

# For this to be an appropriate metric, I need to mask out antennas that are bad or decorrelated.
# I can do this by looking up the antenna-scores and flagging out rows/columns that include ants 
# that only scored a 1. (Chenoa recommended this, but I expect it won't matter all that much 0.0)

# What do the correlations look like?
for i,stamp in enumerate(final_stamps):
    row_info = obs_info_final.iloc[i]
    ant_score = np.fromstring(row_info.antenna_score.strip('[]'), sep = ' ')
    flag_rc = np.where(np.logical_or(ant_score == 0, ant_score == 1))
    # Get correlation
    corr = stamp.correlations()
    # Flag Correlation
    np.fill_diagonal(corr, np.nan)
    corr[flag_rc, :] = corr[:, flag_rc] = np.nan
    print(f"Gaia ID {obs_info_final.source_name_oi.iloc[0]} ({stamp.stamp.signal.frequency} MHz)")
    print(f"    Mean Correlation: {np.nanmean(corr)}\n    STD Correlation: {np.nanstd(corr)}\n")
    # fig = plt.figure()
    # plt.imshow(corr, origin='lower')
    # plt.show()


# # This needs some testing with Voyager and SETIgen to have more confidence.
# test_corr = stamp.correlations()

# # Change to upchannelized? I mean dedrift works fine I guess. 
# ddStamp_masked = dedrift_stamp(stamp,1, mask=True)
# mav, ddcorrelations = correlate_stamp(ddStamp_masked)

# fig, axs = plt.subplots(1,2, constrained_layout=True)
# axs[0].imshow(test_corr, origin='lower')
# axs[0].set_title('regular')
# axs[1].imshow(ddcorrelations, origin='lower')
# axs[1].set_title('dd')
# plt.show()


"""
### Sloped Chirp
obs_info_classified = pd.read_csv(os.path.join(PLOT_PATH, "uniqueFinalHits10pc_classified.csv"))
sloped_chirps = obs_info_classified[obs_info_classified["classification"] == 'sloped chirps']
chirp_stamps = []
for i,row in sloped_chirps.iterrows():
    ## Fetch Stamp
    stampfn = os.path.join(STAMP_PATH, os.path.basename(row.stamp_file_uri_oi))
    stamps_gen = viewer.read_stamps(stampfn, find_recipe=True)
    for index, stamp in enumerate(stamps_gen):
        if index == row.stamp_file_local_enum:
            assert(stamp != None)
            assert(stamp.recipe != None)
            chirp_stamps.append(stamp)
            break
        
# # Plot Signal Stamps
# fig, axs = plt.subplots(3,3, constrained_layout=True)
# for i in range(3):
#     k = i+5
#     a = stamp_power[k,:,:,:].sum(axis=-1)
#     b = ddStamp[k,:,:,:].sum(axis=-1)
#     c = ddStamp2[k,:,:,:].sum(axis=-1)

#     axs[i, 0].imshow(a.T)
#     axs[i, 0].set_title('regular')
#     axs[i, 1].imshow(b.T)
#     axs[i, 1].set_title('dd')
#     axs[i, 2].imshow(c.T)
#     axs[i, 2].set_title('dd upchannel')
# plt.show()
"""

# %%
