import os
import sys
import numpy as np
import pandas as pd
import h5py
import time
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

def db(x):
    """ Linear to dB space """
    if not isinstance(100,np.ndarray):
        x = np.array(x)
    return 10 * np.log10(np.abs(x.astype(np.float64))+1e-20)

def box_t_profile_index(start_index, duration_bins, level, dt=None):
    """
    Rectangular pulse profile controlled by time-bin index.

    Inputs:
        start_index : int
            First time bin where the pulse is on.
        duration_bins : int
            Number of time bins for which the pulse is on.
        level : float
            Injected intensity level, e.g. frame.get_intensity(snr=snr).
        dt : float or astropy Quantity, optional
            Time resolution. If omitted, this assumes setigen passes t in units
            where adjacent samples differ by 1 index. Usually you should pass frame.dt.
    """

    def profile(t):
        # Convert astropy Quantity to plain seconds if needed
        if hasattr(t, "unit"):
            t_vals = t.to("s").value
            dt_val = dt.to("s").value if hasattr(dt, "unit") else dt
        else:
            t_vals = np.asarray(t)
            dt_val = dt

        # Convert time values to integer sample index
        if dt_val is None:
            t_index = np.rint(t_vals).astype(int)
        else:
            t_index = np.rint(t_vals / dt_val).astype(int)

        stop_index = start_index + duration_bins

        return level * ((t_index >= start_index) & (t_index < stop_index))

    return profile

def create_inj_signals(h5_source, n_sigs, seed):
    """
    Draw a set of randomly generated narrowband technosignatures from a given distribution
    Inputs:
        n_sigs: number of signals to generate
        seed: random number generator seed for reproduceability

    """
    # get info from file
    with h5py.File(h5_source, 'r') as hd:
        data_attrs = dict(hd['data'].attrs)
        data_shape = hd['data'].shape

    # setup shapes
    n_times = data_shape[0]
    n_chans = data_shape[-1]

    # build generator
    rng = np.random.default_rng(seed)

    ## Draw values 
    # drs (+/-3 is the best that we can do with bliss rn)
    drift_rates = rng.uniform(-3,3,n_sigs) # Hz/s

    # snrs
    snrs = rng.uniform(15,500, n_sigs) # we are searching for SNR>15

    # widths
    widths = rng.integers(2,7, n_sigs) # I've found 3 is a good start value, but I'd like to see recovery

    # freqs
    start_freq_inx = rng.integers(0, n_chans, n_sigs)
    start_freq = data_attrs['fch1'] + (start_freq_inx * data_attrs['foff'])

    # start_times and duration
    start_time_inx = rng.integers(0,n_times, n_sigs) # index
    durations = rng.normal(loc= n_times/2, scale= n_times/6, size= n_sigs)
    stop_time_inx = start_time_inx + durations 
    durations_s = durations * data_attrs['tsamp']

    # compute stop frequencies (index)
    freq_drift = durations_s * drift_rates # Hz
    f_drift_chans = freq_drift / (data_attrs['foff'] * 1e6) # number of bins in freq
    stop_freq_inx = (start_freq_inx + np.around(f_drift_chans)).astype(int)

    signals_dict = {
                "snr": snrs,
                "drift_rate": drift_rates,
                "width": widths,
                "start_freq_inx": start_freq_inx,
                "stop_freq_inx": stop_freq_inx,
                "start_freq_MHz": start_freq,
                "start_time_inx": start_time_inx,
                "stop_time_inx": stop_time_inx,
                "duration": durations,

            }
    signals = pd.DataFrame(signals_dict)

    # check for violations of the ranges provided, and remove those signals
    # if stop_freq_inx > n_chans -> remove
    # if stop_time_inx > n_times -> remove
    # if either is less than 0 -> remove
    signals = signals[(signals.stop_time_inx < (n_times-1)) &
                      (signals.stop_freq_inx < (n_chans-1)) & 
                      (signals.stop_time_inx > 0) &
                      (signals.stop_freq_inx > 0)
                      ]
    
    return signals

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
        # data = hd['data'][:]
        # mask = hd['mask'][:]

    st = time.time()
    # Build Main object
    wf = blimpy.Waterfall(h5_source)
    main_frame = setigen.Frame(waterfall=wf)
    print(f'Built Blimpy Waterfall {time.time() - st:.03f}')

    win_means = []
    win_stds = []
    print(f'Starting Signal Injection \n{"-" * 30}')
    print(f'Number of Signals Injecting: {len(injected_signals)}')
    for i, row in injected_signals.iterrows():
        # print(f'injection frequency {row.start_freq_MHz}')
        n_drift_steps = np.abs(row.start_freq_inx - row.stop_freq_inx)
        if n_drift_steps < 10:
            n_drift_steps = 10
        # left and right bounds of a section to window for statistics (twice as wide in freq as the drift itself)
        f_upper_bound = row.start_freq_MHz + (data_attrs['foff'] * n_drift_steps * 2)
        f_lower_bound = row.start_freq_MHz - (data_attrs['foff'] * n_drift_steps * 2)
        
        ### Inject Signals
        ## Do I need to adjust for the bandpass? Or can I use the local noise stats?

        # Build Window Object
        data_window = blimpy.Waterfall(h5_source,
                                    f_start = f_lower_bound,
                                    f_stop = f_upper_bound,)
                                    
        try:
            window_frame = setigen.Frame(data_window)
        except IndexError:
            print(f"upper bound {f_upper_bound}")
            print(f"lower bound {f_lower_bound}")
            print(f"drift_rate {row.drift_rate}")
            return
            
            

        # get statistics from window, then adjust snr
        win_freq, _ = data_window.grab_data()
        inj_win_mean, inj_win_std = window_frame.get_noise_stats()
        win_means.append(inj_win_mean)  
        win_stds.append(inj_win_std)
        intensity_at_snr = window_frame.get_intensity(snr = row.snr)
        
        # add_signal
        signal = main_frame.add_signal(
            path = setigen.constant_path(f_start = row.start_freq_MHz * u.MHz,
                                        drift_rate = row.drift_rate * u.Hz/u.s),
            # t_profile = setigen.constant_t_profile(level=intensity_at_snr), 
            t_profile = box_t_profile_index(row.start_time_inx, row.duration, level= intensity_at_snr, dt= data_attrs['tsamp']),
            f_profile = setigen.box_f_profile(width = row.width * u.Hz),
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
        grp.create_dataset("start_frequency", data= injected_signals.start_freq_MHz.values )
        grp.create_dataset("drift_rate", data= injected_signals.drift_rate.values )
        grp.create_dataset("zscore", data= injected_signals.snr.values )
        grp.create_dataset("signal_width", data= injected_signals.width.values )
        grp.create_dataset("snr_db", data= db(injected_signals.snr.values) )
        grp.create_dataset("noise_mean", data= np.array(win_means) ) # -> from window_frame.get_noise_stats()
        grp.create_dataset("noise_std", data= np.array(win_stds) ) # -> from window_frame.get_noise_stats()

    ### Plot the Signals Injected
    fig, ax = plt.subplots(1,1)
    ax.set_title(f"Injected Signals")
    ax.scatter(injected_signals.start_freq_MHz, injected_signals.drift_rate)
    ax.set_xlabel(f"Start Frequency (MHz)")
    ax.set_ylabel(f"Drift Rate (Hz/s)")
    plt.show()

    return print(f'done - ')


if __name__ == "__main__":
    ### Get file information
    h5_source = '/home/cat-work/work/swarmSETI/scripts_savin/data_setigen_check.h5'
    out_filename = h5_source.split('.h5')[0] + '_sigInject.h5'

    
    start = time.time()
    print(f"Starting Signal Generation")
    signals = create_inj_signals(h5_source, 1000, seed=2026)
    print(f"Signals Generated [Time Elapsed = {time.time() - start:0.3f}]")
    injectSignal(h5_source, signals)
    print(f"Signals Injected [Time Elapsed = {time.time() - start:0.3f}]")

    ### Validation Check
    with h5py.File(out_filename, 'r') as hd2:
        validation = hd2['data'][:]
    
    fig = plt.figure()
    plt.imshow(validation[:,0,216051-10:216051+60])
    plt.show()

    # os.remove(out_filename)




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