import os
import sys
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import astropy.units as u

from seticore import viewer
from utils import STYLE_PATH, calcFigSize, get_colors
from pfb_maker import gen_coarse_channel_response

import setigen 
import blimpy

import matplotlib
matplotlib.use("TkAgg")

MEERKAT_DATA = '/mnt/d/data1/meerkat_test_data/guppi_60450_05861_003791_JWST_0001.band00.beam00.h5'
MEERKAT_STAMPS = '/mnt/d/data1/meerkat_test_data/guppi_60450_05861_003791_JWST_0001.stamps'
MEERKAT_HITS = '/mnt/d/data1/meerkat_test_data/guppi_60450_05861_003791_JWST_0001.hits'

def db(x):
    """ Linear to dB space """
    if not isinstance(100,np.ndarray):
        x = np.array(x)
    return 10 * np.log10(np.abs(x.astype(np.float64))+1e-20)

def exp_sinusoid(t, A, k, omega, phi, C):
    return A * np.exp(k * t) * np.sin(omega * t + phi) + C

def fit_remaining_sinusoid(frequencies, spectrum):
    initial_guess = [1.0, 0.0, 2 * np.pi, 0.0, np.mean(spectrum)]
    popt, pcov = curve_fit(exp_sinusoid, np.arange(len(frequencies)), spectrum, p0=initial_guess)
    A_fit, k_fit, omega_fit, phi_fit, C_fit = popt
    fitted_line = exp_sinusoid(np.arange(len(frequencies)), A_fit, k_fit, omega_fit, phi_fit, C_fit)
    return fitted_line

def plot_hits():
    mkhit_signal = []
    mkhit_filterbank = []
    mk_hitgen = viewer.read_hits(MEERKAT_HITS)
    for hit in mk_hitgen:
        mkhit_signal.append(hit.signal.to_dict())
        mkhit_filterbank.append(hit.filterbank.to_dict())

    mkhit_signal = pd.DataFrame(mkhit_signal)
    mkhit_filterbank = pd.DataFrame(mkhit_filterbank)

    print(f"num hits {len(mkhit_signal)}")

    mk_stamps = []
    mk_stampgen = viewer.read_stamps(MEERKAT_STAMPS, find_recipe=True)
    for stamp in mk_stampgen:
        mk_stamps.append(stamp)

    print(f"num stamps {len(mk_stamps)}")

    # Format plot
    figs = calcFigSize(name="CQG",columns='onecol')
    plt.style.use(STYLE_PATH)

    fig, ax = plt.subplots(1,1)
    ax.scatter(mkhit_signal['frequency'], mkhit_signal['snr'], c='k')
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('snr')
    ax.set_ylim(0,100)
    plt.show()

    return

def plot_waterfall(chan_file):
    ## Plotting just waterfall plots from the h5 file to see what it looks like inside
    # HDF Stuff
    hd = h5py.File(MEERKAT_DATA)
    data = hd['data']
    tsamp = data.attrs['tsamp']
    fch1 = data.attrs['fch1']
    foff = data.attrs['foff']

    time_elapsed = np.arange(0, tsamp * data.shape[0], tsamp)
    frequencies = fch1 + np.arange(data.shape[-1]) * foff

    # Uncorrected
    ref_data = np.squeeze(data, axis=1)

    # Bliss PFB
    n_coarse_chan = 4
    coarse_chan_size = int(data.shape[-1] / n_coarse_chan)
    chan_response = np.fromfile(chan_file, dtype=np.float32)
    full_chan_resp = np.tile(chan_response, n_coarse_chan)
    data_corrected = np.squeeze(data, axis=1) / full_chan_resp
    avg_timeseries = np.mean(data_corrected, axis=-1)
    avg_spectrum = np.mean(data_corrected, axis=0)

    # Hanning pfb
    hann_response = gen_coarse_channel_response(fine_per_coarse=coarse_chan_size, num_coarse_channels=n_coarse_chan, taps_per_channel=16, window="hann" )
    full_hann_resp = np.tile(hann_response, n_coarse_chan)
    data_corrected_hann = np.squeeze(data, axis=1) / full_hann_resp
    avg_hann_spectrum = np.mean(data_corrected_hann, axis=0)

    # fitting hanning results
    polyfit_coeffs = np.polyfit(np.arange(len(frequencies)), avg_hann_spectrum, deg=6)
    polyvals_fit = np.polyval(polyfit_coeffs, np.arange(len(frequencies)))

    sinusoid_fit = fit_remaining_sinusoid(frequencies, avg_hann_spectrum)


    fig, ax = plt.subplot_mosaic(
        """
        CC
        AB
        """, sharex=True)
    ax["A"].set_title('hamming')
    ax["A"].plot(frequencies, avg_spectrum)
    ax["A"].set_xlabel('Freq (MHz)')
    
    ax["B"].set_title('hann')
    ax["B"].plot(frequencies, avg_hann_spectrum)
    ax["B"].plot(frequencies, polyvals_fit, c='cyan')
    ax["B"].plot(frequencies, sinusoid_fit, c='r')
    ax["B"].set_xlabel('Freq (MHz)')

    ax["C"].set_title('uncorrected')
    ax["C"].plot(frequencies, ref_data.mean(axis=0))
    ax["C"].set_xlabel('Freq (MHz)')
    plt.show()


    hd.close()

    return data_corrected, data_corrected_hann

def injectSignal(h5_source, injected_signals):
    """ 
    inputs:
        - an h5py source file
        - data structure describing the signals injected (same as ending)
    returns:
        - saves the new h5py file to a different filename
    """
    with h5py.File(h5_source, 'r') as hd:
        data_attrs = dict(hd['data'].attrs)
        data = hd['data'][:]
        mask = hd['mask'][:]

    # Build Main object
    wf = blimpy.Waterfall(h5_source)
    main_frame = setigen.Frame(waterfall=wf)

    win_means = []
    win_stds = []
    for i, row in injected_signals.iterrows():
        print(f'injection frequency {row.inj_f_start}')

        # left and right bounds of a section to window for statistics (twice as wide in freq as time)
        f_upper_bound = row.inj_f_start + (data_attrs['foff'] * n_drift_steps)
        f_lower_bound = row.inj_f_start - (data_attrs['foff'] * n_drift_steps)
        
        ### Inject Signals
        ## Do I need to adjust for the bandpass? Or can I use the local noise stats?

        # Build Window Object
        data_window = blimpy.Waterfall(h5_source,
                                    f_start = f_lower_bound,
                                    f_stop = f_upper_bound)
        window_frame = setigen.Frame(data_window)

        # get statistics from window, then adjust snr
        win_freq, _ = data_window.grab_data()
        inj_win_mean, inj_win_std = window_frame.get_noise_stats()
        win_means.append(inj_win_mean)  
        win_stds.append(inj_win_std)
        intensity_at_snr = window_frame.get_intensity(snr = row.inj_snr)
        
        # add_signal
        signal = main_frame.add_signal(
            path = setigen.constant_path(f_start = row.inj_f_start * u.MHz,
                                        drift_rate = row.inj_drift_rate * u.Hz/u.s),
            t_profile = setigen.constant_t_profile(level=intensity_at_snr), 
            f_profile = setigen.box_f_profile(width = row.inj_width * u.Hz),
            bounding_f_range = (f_lower_bound * u.MHz, f_upper_bound * u.MHz),
            # doppler_smearing = True, 
            # smearing_subsamples = 1
        )

    ### Write to outfile
    out_filename = h5_source.split('.h5')[0] + '_sigInject.h5'
    main_frame.save_h5(out_filename)

    ### Add Signal Metadata as new dataset
    with h5py.File(out_filename, 'a') as f:
        # Create Setigen Group
        grp = f.create_group("setigen")

        # Create datasets within the group
        grp.create_dataset("start_frequency", data= injected_signals.inj_f_start.values )
        grp.create_dataset("drift_rate", data= injected_signals.inj_drift_rate.values )
        grp.create_dataset("zscore", data= injected_signals.inj_snr.values )
        grp.create_dataset("signal_width", data= injected_signals.inj_width.values )
        grp.create_dataset("snr_db", data= db(injected_signals.inj_snr.values) )
        grp.create_dataset("noise_mean", data= np.array(win_means) ) # -> from window_frame.get_noise_stats()
        grp.create_dataset("noise_std", data= np.array(win_stds) ) # -> from window_frame.get_noise_stats()

    ### Plot the Signals Injected
    fig, ax = plt.subplots(1,1)
    ax.set_title(f"Injected Signals")
    ax.scatter(injected_signals.inj_f_start, injected_signals.inj_drift_rate)
    ax.set_xlabel(f"Start Frequency (MHz)")
    ax.set_ylabel(f"Drift Rate (Hz/s)")
    plt.show()


    return print('done')




if __name__ == "__main__":
    # data_corrected, hann_corrected = plot_waterfall(chan_file = '/mnt/d/data1/meerkat_test_data/channel_response_meerkat_131072_16.f32')
    
    # Test Claude Code:
    # chan_file = '/mnt/d/data1/meerkat_test_data/channel_response_meerkat_131072_16.f32'
    # chan_response = np.fromfile(chan_file, dtype=np.float32)

    ### Get file information
    h5_source = '/home/cat-work/work/swarmSETI/scripts_savin/data_setigen_check.h5'
    out_filename = h5_source.split('.h5')[0] + '_sigInject.h5'

    with h5py.File(h5_source, 'r') as hd:
        data_attrs = dict(hd['data'].attrs)
        data = hd['data'][:]
        mask = hd['mask'][:]

    # Build Main object
    # wf = blimpy.Waterfall(h5_source)
    # main_frame = setigen.Frame(waterfall=wf)

    ### Build Signal(s)
    ## We will need a dataframe with these columns, then we can iteratate by row
    
    # This would be chosen specifically for the test case, here I am just looking near the window
    test_channel = 216051
    inj_f_starts = (test_channel + np.array([1,2,3]) * 10) * data_attrs['foff'] + data_attrs['fch1'] # first signal is at f = 54.004850 MHz
    inj_drift_rates = np.arange(3) *  0.4 # Hz/s
    inj_snrs = np.array([100.,110.,120.])
    inj_width = data_attrs['foff'] * 1e6 # Hz adjusted from units of MHz which is a relic of the data generation.  
    inj_widths = np.full(3, inj_width)
    inj_snrs_db = db(inj_snrs) # -> leave this out? 
    n_drift_steps = 20 # all timesteps

    inj_data = {
        "inj_f_start": inj_f_starts,
        "inj_drift_rate": inj_drift_rates,
        "inj_snr": inj_snrs,
        "inj_width": inj_widths,
    }

    injection_test = pd.DataFrame(data = inj_data)
    injectSignal(h5_source, injection_test)

    """
    win_means = []
    win_stds = []
    for i, row in injection_test.iterrows():
        print(f'injection frequency {row.inj_f_start}')

        # left and right bounds of a section to window for statistics (twice as wide in freq as time)
        f_upper_bound = row.inj_f_start + (data_attrs['foff'] * n_drift_steps)
        f_lower_bound = row.inj_f_start - (data_attrs['foff'] * n_drift_steps)
        
        ### Inject Signals
        ## Do I need to adjust for the bandpass? Or can I use the local noise stats?

        # Build Window Object
        data_window = blimpy.Waterfall(h5_source,
                                    f_start = f_lower_bound,
                                    f_stop = f_upper_bound)
        window_frame = setigen.Frame(data_window)

        # get statistics from window, then adjust snr
        win_freq, _ = data_window.grab_data()
        inj_win_mean, inj_win_std = window_frame.get_noise_stats()
        win_means.append(inj_win_mean)  
        win_stds.append(inj_win_std)
        intensity_at_snr = window_frame.get_intensity(snr = row.inj_snr)
        
        # add_signal
        signal = main_frame.add_signal(
            path = setigen.constant_path(f_start = row.inj_f_start * u.MHz,
                                        drift_rate = row.inj_drift_rate * u.Hz/u.s),
            t_profile = setigen.constant_t_profile(level=intensity_at_snr), 
            f_profile = setigen.box_f_profile(width = row.inj_width * u.Hz),
            bounding_f_range = (f_lower_bound * u.MHz, f_upper_bound * u.MHz),
            # doppler_smearing = True, 
            # smearing_subsamples = 1
        )

    ### Write to outfile
    out_filename = h5_source.split('.h5')[0] + '_sigInject.h5'
    main_frame.save_h5(out_filename)

    ### Add Signal Metadata as new dataset
    with h5py.File(out_filename, 'a') as f:
        # Create Setigen Group
        grp = f.create_group("setigen")

        # Create datasets within the group
        grp.create_dataset("start_frequency", data= injection_test.inj_f_start.values )
        grp.create_dataset("drift_rate", data= injection_test.inj_drift_rate.values )
        grp.create_dataset("zscore", data= injection_test.inj_snr.values )
        grp.create_dataset("signal_width", data= injection_test.inj_width.values )
        grp.create_dataset("snr_db", data= db(injection_test.inj_snr.values) )
        grp.create_dataset("noise_mean", data= np.array(win_means) ) # -> from window_frame.get_noise_stats()
        grp.create_dataset("noise_std", data= np.array(win_stds) ) # -> from window_frame.get_noise_stats()


    ### Plot the Signals Injected
    """
    ### Validation Check
    with h5py.File(out_filename, 'r') as hd2:
        validation = hd2['data'][:]
    
    fig = plt.figure()
    plt.imshow(validation[:,0,216051-10:216051+60])
    plt.show()

    os.remove(out_filename)




    # This is the default signal from Ken's github page.
    # signal_morph = {
    #     "sig_frequency": 54.00485030152919, # placeholder
    #     "sig_max_drift": 2.0, #Hz/s
    #     "total_drift_cycles": 10.0, #idk what these mean yet
    #     "n_sig_per_cycle": 20.1,
    #     "drift_offset": -0.02,
    #     "sig_snr_linear": 100, 
    #     "sig_snr_db": 20,
    #     "sig_width_bins": 1, #Hz
    #     "do_snr_compensation": True
    # }
