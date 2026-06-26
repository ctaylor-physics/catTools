import os
import sys
import ast
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn
from utils import STYLE_PATH, calcFigSize, get_colors, normalize

import glob
from cosmic_utils import look_for_combs, log_with_pandas
from scipy.stats import median_abs_deviation
from scipy import signal

from seticore import viewer
from cosmic_database_analysis import sarfi

from blri.dsp import upchannelise

PLOT_PATH = '/home/cat-work/work/SETI/cosmicStellarHosts/databaseHits/observationIds_10pc'
STAMP_PATH = '/home/cat-work/work/SETI/cosmicStellarHosts/databaseHits/observationIds_10pc/final_10pc_candidates'
STYLE_PATH2 = '/home/cat-work/.config/matplotlib/2stamp_corr.mplstyle'

def get_beam(r,row):
    return row.A0 + (row.A2*r**2) + (row.A4*r**4) + (row.A6*r**6)

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
        stamp_power = np.square(stamp.real_array()).sum(axis= 4) 
        stamp_power = np.ma.array(data = stamp_power, mask = np.broadcast_to(sig_mask[:,:,np.newaxis, np.newaxis], stamp_power.shape))
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

def snr_timeseries(stamp):
    # Getn antenna power [ant, time, chan]
    data = np.square(stamp.real_array()).sum(axis=(2, 4)).transpose(2, 0, 1)
    signal_mask = stamp.signal_mask()

    # Gather time-dependent noise 
    left_column = data[:,:, :30]
    right_column = data[:,:, -30:]
    column_noise = np.concatenate((left_column, right_column), axis=-1) # Outer noise per antenna per time
    column_means = np.mean(column_noise, axis=-1)
    column_std = np.std(column_noise, axis=-1)

    data_signals = ( data * signal_mask ).sum(axis=-1)
    snrs = (data_signals - column_means) / column_std
    return snrs, data_signals

def plot_2ant_corr(stamp,row_info, ant1, ant2):

    ## Build Power Array [ant,chan,time,pol]
    stamp_power = np.square(stamp.real_array()).sum(axis= 4).transpose(3,1,0,2)
    middle = int(stamp_power.shape[1]/2)
    stamp_power = stamp_power[:,middle-32:middle+32,:,:].sum(axis=-1)
    stamp_shape = stamp_power.shape

    ## Get correlations
    ant_score = np.fromstring(row_info.antenna_score.strip('[]'), sep = ' ')
    flag_rc = np.where(np.logical_or(ant_score == 0, ant_score == 1))

    corr = stamp.correlations()
    np.fill_diagonal(corr, np.nan)
    corr[flag_rc, :] = corr[:, flag_rc] = np.nan
    print(f"Mean Correlation: {np.nanmean(corr)}\nSTD Correlation: {np.nanstd(corr)}\n")

    ## Get antenna indices
    ant_names = stamp.recipe.antenna_names
    ant_names.append('ea09')
    ant_names = np.array(sorted(ant_names))
    print(ant_names.shape)
    ant1_inx = np.where(ant_names == ant1)[0]
    ant2_inx = np.where(ant_names == ant2)[0]
    first_image = stamp_power[ant1_inx]
    first_min = np.min(first_image)
    first_max = np.max(first_image)*1.1

    ## Format plot
    print(f"full_frequency {stamp.stamp.signal.frequency} MHz")

    figs = calcFigSize(name="PRD",columns='twocol')
    plt.style.use(STYLE_PATH2)

    fig, axs = plt.subplots(1,3,figsize=figs, constrained_layout=True)
    plt.suptitle(f"Gaia ID {row_info.source_name_oi} - {stamp.stamp.signal.frequency:.3f} MHz", y=0.82)
    ### ant1
    axs[0].set_title(f'Stamp: {ant1}') # , fontsize=6)
    axs[0].imshow(stamp_power[ant1_inx].T, origin='lower', vmin = first_min, vmax = first_max)

    # set up ticks
    yticks = np.arange(0,stamp_shape[2],stamp_shape[2]//5)
    axs[0].set_yticks(yticks)
    axs[0].set_yticklabels([f"{tick*stamp.stamp.tsamp:.2f}" for tick in yticks] , fontsize=8)
    axs[0].set_ylabel("Time (s)", fontsize=8)

    xticks = np.arange(0,stamp_shape[1], stamp_shape[1]//5)
    axs[0].set_xticks(xticks)
    axs[0].set_xticklabels([f"{tick*stamp.stamp.foff*1e6:.1f}" for tick in xticks], rotation=-25, fontsize=8)
    axs[0].set_xlabel(f"Frequency\n(kHz+{middle*stamp.stamp.foff + stamp.stamp.fch1:.5f} MHz)", fontsize=9)

    
    ### ant2
    axs[1].set_title(f'Stamp: {ant2}') # , fontsize=6)
    axs[1].imshow(stamp_power[ant2_inx].T, origin='lower', vmin = first_min, vmax = first_max)
    axs[1].set_yticks(yticks)
    axs[1].set_yticklabels([f"{tick*stamp.stamp.tsamp:.2f}" for tick in yticks], fontsize=8)
    axs[1].set_yticks([])
    axs[1].set_xticks(xticks)
    axs[1].set_xticklabels([f"{tick*stamp.stamp.foff*1e6:.1f}" for tick in xticks], rotation=-25, fontsize=8)
    axs[1].set_xlabel(f"Frequency\n(kHz+{middle*stamp.stamp.foff + stamp.stamp.fch1:.5f} MHz)", fontsize=9)

    ### ant3
    plt.tick_params(axis='both', bottom=False, left=False)
    label_lox = np.array([0,stamp_shape[0]//2, -1])
    elip_lox = np.array([1,stamp_shape[0]//2 - 1,stamp_shape[0]//2 + 1, -2])
    short_ant_names = np.array([ant[2:] for ant in ant_names])
    short_ant_names2 = np.full(stamp_shape[0], '', dtype=object)
    short_ant_names2[label_lox] = short_ant_names[label_lox]
    short_ant_names2[elip_lox] = '...'

    corr_cmap = plt.cm.plasma
    corr_cmap.set_bad(color='grey', alpha=0.5)

    axs[2].set_title(f'Antenna Correlation Matrix') # , fontsize=8)
    im_corr = axs[2].imshow(corr, origin='lower', vmin=0, vmax=1, cmap = corr_cmap)
    axs[2].set_xlabel(f"Antenna Number", fontsize=9)
    axs[2].set_ylabel(f"Antenna Number", fontsize=9)
    axs[2].set_xticks(np.arange(0,stamp_shape[0],1))
    axs[2].set_xticklabels(short_ant_names2, fontsize=8)
    axs[2].set_yticks(np.arange(0,stamp_shape[0],1))
    axs[2].set_yticklabels(short_ant_names2, rotation=90, fontsize=8)
    cb = plt.colorbar(im_corr, ax=axs[2], fraction=0.05, pad=0.04)
    cb.ax.tick_params(labelsize=8)
    plt.savefig(os.path.join(PLOT_PATH, f'paper_plots/{row_info.source_name_oi}_{ant1}_{ant2}_2.png'))
    plt.show()

    # could create a custom cmap with 'bad' values to represent np.nan using: 
    # cmap = plt.cm.viridis
    # map.set_bad(color='white') # Set NaN to white

    return

def plot_timeseries_beam(stamp):
    vla_sband_beam = pd.read_csv('/home/cat-work/work/SETI/vla_Sband_beam.csv')
    
    antenna_powers = np.square(stamp.real_array()).sum(axis=(2, 4)).transpose(2, 0, 1)
    snr_and_signals = np.array([stamp.snr_and_signal(antenna_power) for antenna_power in antenna_powers])

    ### Dedispersed Stamp Nonsense
    ddStamp = dedrift_stamp(stamp,1, mask=True)
    ddStamp_data = ddStamp.data.sum(axis=-1)
    print(ddStamp_data.shape)
    ddStamp_mask = ddStamp.mask[:,:,:,0].astype(bool)

    # adds 1 index to each direction for better capture
    beefed_mask = ( ddStamp_mask | np.roll(ddStamp_mask, 1, axis=1) | np.roll(ddStamp_mask, -1, axis=1) ) 
    noise_mask = ~beefed_mask
    
    noise_data = np.where(noise_mask, ddStamp_data, np.nan)
    noise_mean = np.nanmean(noise_data, axis=1)
    noise_std = np.nanstd(noise_data, axis=1, ddof=1)

    n_signal = np.sum(beefed_mask, axis=1)
    n_noise = np.sum(noise_mask, axis=1)

    gross_signal = np.sum(ddStamp_data * beefed_mask, axis=1)
    gross_noise = noise_mean * n_signal

    snr = (gross_signal - gross_noise) / (noise_std * np.sqrt(n_signal))

    array_signal = np.sum(gross_signal, axis=0)
    array_noise = np.sum(gross_noise, axis=0)

    array_noise_var = np.sum((noise_std * np.sqrt(n_signal))**2, axis=0)
    array_snr = (array_signal - array_noise) / np.sqrt(array_noise_var)

    print("Per-antenna mean SNR:", np.nanmean(snr, axis=-1))
    print("Array-combined mean SNR:", np.nanmean(array_snr))


    # print(f' signal {np.mean(gross_signal, axis=-1)}')
    # print(f' noise {np.mean(gross_noise, axis=-1)}')
    # print(f' snr {np.mean(snr, axis=-1)}')


    ### VLA Beam
    freq_inx = np.abs( vla_sband_beam.Freq_MHz - stamp.stamp.signal.frequency).argmin()
    r = np.linspace(-41,41,ddStamp_data.shape[-1])
    power_r = get_beam(r, vla_sband_beam.iloc[freq_inx])

    ### PLOT
    figs = calcFigSize(name="CQG",columns='onecol')
    plt.style.use(STYLE_PATH)

    fig, ax = plt.subplots(1,1)
    for i in range(ddStamp.shape[0]//5):
        print(snr_and_signals[i][0])
        plotarr = snr[i]
        ax.plot(plotarr, c='k')

    ax.plot(power_r, c='r', linestyle='--')
    plt.show()

    fig, ax = plt.subplots(1,1)
    for i in range(ddStamp.shape[0]//6):
        plotarr = normalize(gross_signal[i])
        ax.plot( plotarr , c='k', alpha = 0.75)
    ax.plot( plotarr , c='k', alpha = 0.75, label = 'Single antenna responses')

    ax.plot(power_r, c='darkorange', linewidth = 1.5, label='S-Band Primary Beam Response')
    ax.set_ylabel('Normalized Power')
    ax.set_xlabel('Elapsed Time (s)')
    ax.set_xticks(np.arange(9) * 8, labels=np.arange(9).astype(str))
    ax.legend()
    plt.show()

    return gross_signal, gross_noise

def plot_timeseries_beam2(stamp):
    signal_mask = stamp.signal_mask()

    # SNRS by antenna and time
    snrs, signals = snr_timeseries(stamp)

    # SNRS Incoherent vs Time
    incoherent = np.square(stamp.real_array()).sum(axis=(2, 3, 4))
    incoherent_noise = np.concatenate((incoherent[:,:30], incoherent[:,-30:]), axis=-1)
    inc_noise_mean = np.mean(incoherent_noise, axis=-1)
    inc_noise_std = np.std(incoherent_noise, axis=-1)
    inc_snr = ((incoherent * signal_mask).sum(axis=-1) - inc_noise_mean) / inc_noise_std

    ### VLA Beam
    vla_sband_beam = pd.read_csv('/home/cat-work/work/SETI/vla_Sband_beam.csv')
    freq_inx = np.abs( vla_sband_beam.Freq_MHz - stamp.stamp.signal.frequency).argmin()
    r = np.linspace(-41,41,snrs.shape[-1])
    power_r = get_beam(r, vla_sband_beam.iloc[freq_inx])

    fig, ax = plt.subplots(1,1)
    ax.plot(np.arange(64), power_r * stamp.stamp.signal.snr, c='darkorange', linewidth=1.5, label='S-Band Primary Beam Response')
    for i in range(snrs.shape[0]//6):
        plotarr = snrs[i]
        # ax.scatter(np.arange(64), plotarr, c='k', s=4)
        ax.plot(np.arange(64), plotarr, c='k', alpha = 0.7)

    ax.plot(np.arange(64), plotarr, c='k', alpha = 0.7, label = 'Single antenna responses')

    # ax.scatter(np.arange(64), inc_snr, c='b', label='incoherent beam', s=4)
    ax.plot(np.arange(64), inc_snr, c='b', label='incoherent beam', alpha = 0.8)
    ax.set_ylabel('Signal-to-noise')
    ax.set_xlabel('Elapsed Time (s)')
    ax.set_xticks(np.arange(9) * 8, labels=np.arange(9).astype(str))
    ax.set_ylim(-5,35)
    plt.legend()
    plt.show()
    
    return

# This is Figure 6 from Tremblay 2025
def plot_coherent_timeseries(stamp, src_name):
    
    coh_beam = stamp.beamform_power(0)
    coh_timeseries = coh_beam.sum(axis=-1)
    coh_norm = normalize(coh_timeseries)

    incoherent = np.square(stamp.real_array()).sum(axis=(2, 3, 4))
    incoh_timeseries = incoherent.sum(axis=-1)
    incoh_norm = ( incoh_timeseries - np.min(coh_timeseries) ) / ( np.max(coh_timeseries) - np.min(coh_timeseries) )

    ### VLA Beam
    vla_sband_beam = pd.read_csv('/home/cat-work/work/SETI/vla_Sband_beam.csv')
    freq_inx = np.abs( vla_sband_beam.Freq_MHz - stamp.stamp.signal.frequency).argmin()
    r = np.linspace(-41,41,64)
    power_r = get_beam(r, vla_sband_beam.iloc[freq_inx])

    ### PLOT
    figs = calcFigSize(name="CQG",columns='onecol')
    plt.style.use(STYLE_PATH)

    fig, ax = plt.subplots(1,1)
    ax.set_title(f"Gaia ID {src_name} - {stamp.stamp.signal.frequency:.3f} MHz", fontsize=14)
    ax.plot(coh_norm, 'k', label='Coherent')
    ax.plot(incoh_norm, 'b', label='Incoherent')
    ax.plot(np.arange(64), power_r, c='darkorange', linewidth=1.5, label='S-Band\n Primary Beam', zorder=1)

    ax.set_ylabel('Normalized Power', fontsize=14)
    ax.set_xlabel('Elapsed Time (s)', fontsize=14)
    ax.set_xticks(np.arange(9) * 8, labels=np.arange(9).astype(str), fontsize=12)
    ax.set_yticks(np.linspace(0,1,5),labels=np.linspace(0,1,5).astype(str), fontsize=12)
    ax.legend(fontsize=10)
    plt.savefig(os.path.join(PLOT_PATH, f'paper_plots/{src_name}_primary_beam_plot.png'))
    plt.show()

    print('fontsize x-label', ax.xaxis.label.get_fontsize())
    print('fontsize title', ax.title.get_fontsize())
    print('fontsize tick labels', ax.xaxis.get_tick_params())
    
    
    return


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

    ## FOR THE PAPER:
    # plots of antennas (A) ea01 and (B) ea19 with (C) cc spectra for final_stamps[3] (2506MHz) - with light curve under?
    # plots of antennas (A) ea27 and (B) ea28 with (C) cc spectra for final_stamps[5] (3755MHz)
    # plots of antennas 
    # for each of the candidates above: plot all dedispersed time series w/ normalized beam profile overlay
    
    
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

##########################
### Working Stuff Here ###
##########################
obs_info_final = pd.read_csv(os.path.join(PLOT_PATH, 'uniqueFinalHits10pc_shortlist.csv'))


### Collect Stamps
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

## Paper Plots
plot_2ant_corr(final_stamps[3], obs_info_final.iloc[3], 'ea01', 'ea15')
# plot_2ant_corr(final_stamps[5], obs_info_final.iloc[5], 'ea27', 'ea28')
# plot_coherent_timeseries(final_stamps[0], obs_info_final.iloc[0].source_name_oi)


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

stamp = final_stamps[3]
stamp = vstamps[1]



# # Get arrays
# ddStamp = np.abs(dedrift_stamp(stamp,1, mask=False))
# ddStamp2 = np.abs(dedrift_stamp(stamp,2, mask=False))

# v_timeseries = ddStamp.sum(axis=(1,-1))
# P_center = np.array([0.01728   , 0.03699497, 0.05682462, 0.07757318, 0.09987174,
#                     0.12419309, 0.15086616, 0.18008982, 0.21194622, 0.24641365,
#                     0.28337887, 0.32264885, 0.36396212, 0.40699951, 0.45139445,
#                     0.49674269, 0.54261154, 0.5885486 , 0.63408999, 0.67876801,
#                     0.72211837, 0.7636868 , 0.80303528, 0.83974764, 0.87343467,
#                     0.90373883, 0.93033823, 0.95295035, 0.97133501, 0.98529701,
#                     0.99468814, 0.99940875, 0.99940875, 0.99468814, 0.98529701,
#                     0.97133501, 0.95295035, 0.93033823, 0.90373883, 0.87343467,
#                     0.83974764, 0.80303528, 0.7636868 , 0.72211837, 0.67876801,
#                     0.63408999, 0.5885486 , 0.54261154, 0.49674269, 0.45139445,
#                     0.40699951, 0.36396212, 0.32264885, 0.28337887, 0.24641365,
#                     0.21194622, 0.18008982, 0.15086616, 0.12419309, 0.09987174,
#                     0.07757318, 0.05682462, 0.03699497, 0.01728   ])
# # %%
# vmask = stamp.signal_mask()
# stamp_power = np.square(stamp.real_array()).sum(axis= 4).transpose(3,1,0,2)
# stamp_power_beam = stamp_power * P_center[np.newaxis, np.newaxis,:,np.newaxis]
# stamp_power_masked = stamp_power * vmask.swapaxes(1,0)[np.newaxis, :,:,np.newaxis]

# stamp_complex = stamp.complex_array().transpose(3,1,0,2)
# stamp_complex_masked = stamp_complex * vmask.swapaxes(1,0)[np.newaxis, :,:,np.newaxis]
# stamp_complex_beam = stamp_complex_masked * P_center[np.newaxis, np.newaxis,:,np.newaxis]

# cols = np.array([18,20,22])
# vcorr = stamp.correlations()
# _, vcorr_check = correlate_stamp(stamp_complex_masked)
# _, vcorr_beam = correlate_stamp(stamp_complex_beam)
# np.fill_diagonal(vcorr, np.nan)
# np.fill_diagonal(vcorr_check, np.nan)
# np.fill_diagonal(vcorr_beam, np.nan)
# vcorr[cols,:] = vcorr[:,cols] = np.nan
# vcorr_check[cols,:] = vcorr_check[:,cols] = np.nan
# vcorr_beam[cols,:] = vcorr_beam[:,cols] = np.nan


# fig, axs = plt.subplots(1,3, sharex=True, sharey=True)
# axs[0].imshow(vcorr, origin='lower')
# axs[1].imshow(vcorr_check, origin='lower')
# axs[2].imshow(vcorr_beam, origin='lower')

# plt.show()

# print('vcorr',
#     np.nanmean(vcorr),
#     np.nanmedian(vcorr),
#     np.nanstd(vcorr))
# print('check',
#     np.nanmean(vcorr_check),
#     np.nanmedian(vcorr_check),
#     np.nanstd(vcorr_check))
# print('beam',
#     np.nanmean(vcorr_beam),
#     np.nanmedian(vcorr_beam),
#     np.nanstd(vcorr_beam))


# # %% 

# fig = plt.figure()
# plt.title('target timeseries from Voyager tracking through phase center')
# plt.plot(v_timeseries[0,:] * P_center)
# plt.xlabel('Time (0-8s)')
# plt.ylabel('Power')
# plt.show()

# # Stamp power * mask
# sp_masked = stamp_power.sum(axis=(-1)).transpose(0,2,1) * stamp.signal_mask()[np.newaxis, :,:]


### Antenna Cross-Correlations Investigation

# For this to be an appropriate metric, I need to mask out antennas that are bad or decorrelated.
# I can do this by looking up the antenna-scores and flagging out rows/columns that include ants 
# that only scored a 1. (Chenoa recommended this, but I expect it won't matter all that much 0.0)

# # What do the correlations look like?
# for i,stamp in enumerate(final_stamps):
#     row_info = obs_info_final.iloc[i]
#     ant_score = np.fromstring(row_info.antenna_score.strip('[]'), sep = ' ')
#     flag_rc = np.where(np.logical_or(ant_score == 0, ant_score == 1))
#     # Get correlation
#     corr = stamp.correlations()
#     # Flag Correlation
#     np.fill_diagonal(corr, np.nan)
#     corr[flag_rc, :] = corr[:, flag_rc] = np.nan
#     print(f"Gaia ID {obs_info_final.source_name_oi.iloc[0]} ({stamp.stamp.signal.frequency} MHz)")
#     print(f"    Mean Correlation: {np.nanmean(corr)}\n    STD Correlation: {np.nanstd(corr)}\n")


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



# ### Sloped Chirp
# obs_info_classified = pd.read_csv(os.path.join(PLOT_PATH, "uniqueFinalHits10pc_classified.csv"))
# sloped_chirps = obs_info_classified[obs_info_classified["classification"] == 'sloped chirps']
# chirp_stamps = []
# for i,row in sloped_chirps.iterrows():
#     ## Fetch Stamp
#     stampfn = os.path.join(STAMP_PATH, os.path.basename(row.stamp_file_uri_oi))
#     stamps_gen = viewer.read_stamps(stampfn, find_recipe=True)
#     for index, stamp in enumerate(stamps_gen):
#         if index == row.stamp_file_local_enum:
#             assert(stamp != None)
#             assert(stamp.recipe != None)
#             chirp_stamps.append(stamp)
#             break
        
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

