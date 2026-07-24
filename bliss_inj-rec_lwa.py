import os
import sys
import numpy as np
import pandas as pd
import h5py
import time
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import astropy.units as u
import shutil

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
    sigs_filename = h5_source.split('.h5')[0] + '_signals.csv'

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
    durations = rng.normal(loc=n_times / 2, scale=n_times / 6, size=n_sigs)
    durations = np.rint(durations).astype(int)
    stop_time_inx = start_time_inx + durations 
    durations_s = durations * data_attrs['tsamp']

    # compute stop frequencies (index)
    freq_drift = durations_s * drift_rates # Hz
    f_drift_chans = freq_drift / (data_attrs['foff'] * 1e6) # number of bins in freq
    stop_freq_inx = (start_freq_inx + np.around(f_drift_chans)).astype(int)

    signals_dict = {
                "SNR": snrs,
                "Drift_Rate": drift_rates,
                "width": widths,
                "start_freq_inx": start_freq_inx,
                "stop_freq_inx": stop_freq_inx,
                "Uncorrected_Frequency": start_freq,
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
    
    signals.to_csv(sigs_filename)

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

### Insert ChatGPT help:
def injectSignal_setigen_windowed(
    h5_source,
    injected_signals,
    overwrite=False,
    min_window_chans=64,
    drift_pad_factor=4,
    debug_first_n=5,
):
    """
    Memory-light signal injection using setigen.add_signal().

    This function:
      - Copies the source HDF5 file to *_sigInject.h5
      - Opens only a small BLIMPY Waterfall frequency window per signal
      - Builds a setigen.Frame from that small window
      - Uses setigen.add_signal()
      - Writes the modified small window back into the output HDF5 file
      - Logs only successfully injected signals to:
            *_signals_injected.csv
      - Writes only successfully injected signals to the HDF5 /setigen group

    Assumes imports already exist at top of script:
      os, shutil, time, numpy as np, pandas as pd, h5py,
      astropy.units as u, blimpy, setigen
    Also assumes db() and box_t_profile_index() are already defined.
    """

    def freq_from_index(fch1, foff, index):
        """
        Convert channel index to frequency in MHz.

        Assumes:
          fch1 in MHz
          foff in MHz/channel
        """
        return fch1 + foff * index

    def h5_data_slice(ndim, t_slice, f_slice):
        """
        Build slices for common waterfall shapes:
          2D: [time, frequency]
          3D: [time, beam/pol, frequency]
        """
        if ndim == 2:
            return (t_slice, f_slice)
        elif ndim == 3:
            return (t_slice, slice(None), f_slice)
        else:
            raise ValueError(f"Unexpected data ndim={ndim}. Expected 2 or 3.")

    out_filename = h5_source.replace(".h5", "_sigInject.h5")
    injected_csv = h5_source.replace(".h5", "_signals_injected.csv")

    if os.path.exists(out_filename):
        if overwrite:
            os.remove(out_filename)
        else:
            raise FileExistsError(
                f"{out_filename} already exists. Use overwrite=True to replace it."
            )

    print("Copying source file to output file:")
    print(f"  {out_filename}")
    shutil.copy2(h5_source, out_filename)

    successful_rows = []
    successful_noise_means = []
    successful_noise_stds = []

    start = time.time()

    with h5py.File(out_filename, "r+") as hd:
        ds = hd["data"]
        data_attrs = dict(ds.attrs)

        n_times = ds.shape[0]
        n_chans = ds.shape[-1]

        fch1 = float(data_attrs["fch1"])
        foff = float(data_attrs["foff"])
        tsamp = float(data_attrs["tsamp"])

        chan_bw_hz = abs(foff * 1e6)

        print(f"Data shape: {ds.shape}")
        print(f"n_times: {n_times}")
        print(f"n_chans: {n_chans}")
        print(f"foff: {foff} MHz/channel")
        print(f"channel bandwidth: {chan_bw_hz:.6f} Hz")
        print(f"Attempting to inject {len(injected_signals)} signals")
        print("-" * 50)

        for inj_number, row in injected_signals.reset_index(drop=True).iterrows():

            start_idx = int(round(row.start_freq_inx))
            stop_idx = int(round(row.stop_freq_inx))

            drift_chans = abs(stop_idx - start_idx)
            drift_chans = max(drift_chans, 10)

            width_chans = max(1, int(np.ceil(float(row.width) / chan_bw_hz)))

            pad_chans = max(
                min_window_chans // 2,
                drift_pad_factor * drift_chans,
                4 * width_chans,
            )

            chan0 = max(0, min(start_idx, stop_idx) - pad_chans)
            chan1 = min(n_chans, max(start_idx, stop_idx) + pad_chans + 1)

            # Enforce a minimum window width where possible
            if (chan1 - chan0) < min_window_chans:
                center = int(round(0.5 * (chan0 + chan1)))
                chan0 = max(0, center - min_window_chans // 2)
                chan1 = min(n_chans, chan0 + min_window_chans)
                chan0 = max(0, chan1 - min_window_chans)

            if chan1 <= chan0:
                print(f"Skipping signal {inj_number}: bad channel window")
                continue

            # Use chan1 as an exclusive edge to match Python slicing.
            # BLIMPY may still return +/- 1 channel, so write-back remains defensive.
            f_edge_a = freq_from_index(fch1, foff, chan0)
            f_edge_b = freq_from_index(fch1, foff, chan1)

            f_start = min(f_edge_a, f_edge_b)
            f_stop = max(f_edge_a, f_edge_b)

            try:
                data_window = blimpy.Waterfall(
                    out_filename,
                    f_start=f_start,
                    f_stop=f_stop,
                )

                window_frame = setigen.Frame(waterfall=data_window)

            except Exception as exc:
                print(f"Skipping signal {inj_number}: failed to build BLIMPY/SETIGEN window")
                print(f"  error={exc}")
                continue

            requested_nchan = chan1 - chan0
            returned_nchan = window_frame.data.shape[-1]

            if inj_number < debug_first_n:
                print(
                    f"signal {inj_number}: "
                    f"chan0={chan0}, chan1={chan1}, "
                    f"requested_nchan={requested_nchan}, "
                    f"returned_nchan={returned_nchan}, "
                    f"f_start={f_start}, f_stop={f_stop}, "
                    f"frame_shape={window_frame.data.shape}"
                )

            try:
                inj_win_mean, inj_win_std = window_frame.get_noise_stats()
                intensity_at_snr = window_frame.get_intensity(snr=float(row.snr))

            except Exception as exc:
                print(f"Skipping signal {inj_number}: failed to compute local noise stats")
                print(f"  error={exc}")
                continue

            duration_bins = int(round(row.duration))

            if duration_bins <= 0:
                print(f"Skipping signal {inj_number}: non-positive duration {duration_bins}")
                continue

            try:
                window_frame.add_signal(
                    path=setigen.constant_path(
                        f_start=float(row.start_freq_MHz) * u.MHz,
                        drift_rate=float(row.drift_rate) * u.Hz / u.s,
                    ),
                    t_profile=box_t_profile_index(
                        int(round(row.start_time_inx)),
                        duration_bins,
                        level=intensity_at_snr,
                        dt=tsamp,
                    ),
                    f_profile=setigen.box_f_profile(
                        width=float(row.width) * u.Hz
                    ),
                    bounding_f_range=(
                        f_start * u.MHz,
                        f_stop * u.MHz,
                    ),
                )

            except Exception as exc:
                print(f"Skipping signal {inj_number}: setigen.add_signal failed")
                print(f"  error={exc}")
                continue

            # ------------------------------------------------------------
            # Robust write-back block
            # ------------------------------------------------------------
            injected_window = np.asarray(window_frame.data)
            n_win_chans = injected_window.shape[-1]

            # Do not assume n_win_chans == chan1 - chan0.
            # BLIMPY may return one fewer or one extra channel because f_start
            # and f_stop are frequency bounds.
            write_chan0 = chan0
            write_chan1 = chan0 + n_win_chans

            # If this would exceed the dataset edge, trim the returned window.
            if write_chan1 > n_chans:
                overflow = write_chan1 - n_chans
                if overflow > 0:
                    injected_window = injected_window[..., :-overflow]
                write_chan1 = n_chans
                n_win_chans = injected_window.shape[-1]

            if n_win_chans <= 0:
                print(f"Skipping signal {inj_number}: zero-width write window after edge trim")
                continue

            if inj_number < debug_first_n:
                target_preview_shape = ds[
                    h5_data_slice(
                        ds.ndim,
                        slice(0, n_times),
                        slice(write_chan0, write_chan1),
                    )
                ].shape

                print(
                    f"  write-back: "
                    f"write_chan0={write_chan0}, write_chan1={write_chan1}, "
                    f"injected_shape={injected_window.shape}, "
                    f"target_shape={target_preview_shape}"
                )

            try:
                if ds.ndim == 2 and injected_window.ndim == 2:
                    ds[:, write_chan0:write_chan1] = injected_window.astype(
                        ds.dtype,
                        copy=False,
                    )

                elif ds.ndim == 3 and injected_window.ndim == 2:
                    # Common case for your file:
                    #   ds.shape = [time, 1, frequency]
                    #   injected_window.shape = [time, frequency]
                    for mid in range(ds.shape[1]):
                        ds[:, mid, write_chan0:write_chan1] = injected_window.astype(
                            ds.dtype,
                            copy=False,
                        )

                elif ds.ndim == 3 and injected_window.ndim == 3:
                    target_slice = h5_data_slice(
                        ds.ndim,
                        slice(0, n_times),
                        slice(write_chan0, write_chan1),
                    )

                    target_shape = ds[target_slice].shape

                    if injected_window.shape != target_shape:
                        raise RuntimeError(
                            "3D shape mismatch before write:\n"
                            f"  injected_window.shape={injected_window.shape}\n"
                            f"  target_shape={target_shape}"
                        )

                    ds[target_slice] = injected_window.astype(ds.dtype, copy=False)

                else:
                    raise RuntimeError(
                        "Unexpected shape combination:\n"
                        f"  ds.shape={ds.shape}\n"
                        f"  injected_window.shape={injected_window.shape}"
                    )

            except Exception as exc:
                print(f"Write-back failed for signal {inj_number}")
                print(f"  original chan0:chan1 = {chan0}:{chan1}")
                print(f"  write_chan0:write_chan1 = {write_chan0}:{write_chan1}")
                print(f"  requested_nchan = {requested_nchan}")
                print(f"  returned_nchan = {returned_nchan}")
                print(f"  injected_window.shape = {injected_window.shape}")
                print(f"  ds.shape = {ds.shape}")
                print(f"  error = {exc}")
                raise

            # Only now do we count the signal as actually injected.
            successful_rows.append(row.copy())
            successful_noise_means.append(inj_win_mean)
            successful_noise_stds.append(inj_win_std)

            if (inj_number + 1) % 25 == 0:
                print(
                    f"Processed {inj_number + 1} / {len(injected_signals)} | "
                    f"Injected successfully: {len(successful_rows)}"
                )

        # ------------------------------------------------------------
        # HDF5 metadata: successful injections only
        # ------------------------------------------------------------
        successful_signals = pd.DataFrame(successful_rows)

        if "setigen" in hd:
            del hd["setigen"]

        grp = hd.create_group("setigen")
        grp.attrs["n_injected"] = len(successful_signals)

        if len(successful_signals) > 0:
            grp.create_dataset(
                "start_frequency",
                data=successful_signals.start_freq_MHz.values,
            )
            grp.create_dataset(
                "drift_rate",
                data=successful_signals.drift_rate.values,
            )
            grp.create_dataset(
                "zscore",
                data=successful_signals.snr.values,
            )
            grp.create_dataset(
                "signal_width",
                data=successful_signals.width.values,
            )
            grp.create_dataset(
                "snr_db",
                data=db(successful_signals.snr.values),
            )
            grp.create_dataset(
                "noise_mean",
                data=np.array(successful_noise_means),
            )
            grp.create_dataset(
                "noise_std",
                data=np.array(successful_noise_stds),
            )
        else:
            print("Warning: no signals were successfully injected.")

    # Save only successfully injected signals.
    if len(successful_rows) > 0:
        successful_signals.to_csv(injected_csv, index=False)
    else:
        pd.DataFrame(columns=injected_signals.columns).to_csv(injected_csv, index=False)

    print("-" * 50)
    print(f"Injected successfully: {len(successful_rows)} / {len(injected_signals)}")
    print(f"Injected signals CSV:  {injected_csv}")
    print(f"Output HDF5:            {out_filename}")
    print(f"Done. Elapsed time: {time.time() - start:.3f} s")

    return out_filename

if __name__ == "__main__":
    ### Get file information
    # h5_source = '/home/cat-work/work/swarmSETI/scripts_savin/data_setigen_check.h5'
    h5_source = '/mnt/d/data1/lwa_seti/059654_002588958-waterfall_tun1_trimmed.h5'
    out_filename = h5_source.split('.h5')[0] + '_sigInject.h5'

    
    start = time.time()
    print(f"Starting Signal Generation")
    signals = create_inj_signals(h5_source, 1000, seed=2026)
    print(f"Signals Generated [Time Elapsed = {time.time() - start:0.3f}]")
    
    # injectSignal(h5_source, signals)
    out_filename = injectSignal_setigen_windowed(
        h5_source,
        signals,
        overwrite=True,
    )
    print(f"Signals Injected [Time Elapsed = {time.time() - start:0.3f}]")
    
    ### Validation Check
    with h5py.File(out_filename, 'r') as hd2:
        validation = hd2['data'][:]
    
    signals = pd.read_csv(h5_source.split('.h5')[0] + '_signals.csv')

    for i,row in signals.iloc[:3].iterrows():
        fig = plt.figure()
        plt.imshow(validation[:,0,int(row.start_freq_inx-20):int(row.stop_freq_inx+20)],
                   origin='lower',aspect='auto')
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
