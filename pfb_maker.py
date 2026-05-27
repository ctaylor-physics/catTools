import numpy as np
from scipy.signal import get_window


def _sinc_lpf(fc: float, num_taps: int) -> np.ndarray:
    """
    Mirrors bliss::sinc_lpf — centred sinc prototype filter.
    fc in range 0..1 (normalised frequency).
    np.sinc uses the normalised form sinc(x) = sin(πx)/(πx), matching the C++.
    """
    half_taps = (num_taps - 1) / 2.0
    t = np.arange(num_taps, dtype=np.float32) - half_taps
    return np.sinc(fc * t).astype(np.float32)


def gen_coarse_channel_response(
    fine_per_coarse: int = 131072,
    num_coarse_channels: int = 4,
    taps_per_channel: int = 16,
    window: str = "hamming",
) -> np.ndarray:
    """
    Python port of bliss::gen_coarse_channel_response, with generic window support.

    Mirrors the C++ pipeline exactly:
        1. Build sinc prototype filter
        2. Apply window (any scipy window name accepted)
        3. Zero-pad to full rate
        4. FFT → fftshift → magnitude squared
        5. Slice out centre region
        6. Fold (reshape + sum) adjacent channel leakage
        7. Normalise

    Parameters
    ----------
    fine_per_coarse      : fine channels per coarse channel  (default 131072)
    num_coarse_channels  : coarse channels in the recording  (default 4)
    taps_per_channel     : PFB taps per coarse channel       (default 16)
    window               : any scipy window name — 'hamming', 'hann',
                           'blackman', ('kaiser', 8.0), etc.  (default 'hamming')

    Returns
    -------
    H : np.ndarray, shape (fine_per_coarse,)
        Normalised PFB amplitude response for one coarse channel.
        Divide raw spectra by this to correct the PFB shape.
    """

    # ------------------------------------------------------------------ #
    # Step 1 — Prototype filter  (mirrors bliss::firdes)                  #
    # num_taps = taps_per_channel * num_coarse_channels                   #
    # fc       = 1 / num_coarse_channels                                  #
    # ------------------------------------------------------------------ #
    num_taps = taps_per_channel * num_coarse_channels
    fc = 1.0 / num_coarse_channels

    h_prototype = _sinc_lpf(fc, num_taps)
    w = get_window(window, num_taps).astype(np.float32)
    h = h_prototype * w

    # ------------------------------------------------------------------ #
    # Step 2 — Zero-pad to full rate                                      #
    # mirrors: h_padded.slice(0, 0, h.size(0)) = h                       #
    # ------------------------------------------------------------------ #
    full_res_length = num_coarse_channels * fine_per_coarse
    h_padded = np.zeros(full_res_length, dtype=np.float32)
    h_padded[:num_taps] = h

    # ------------------------------------------------------------------ #
    # Step 3 — fft_shift_mag_square                                       #
    # mirrors: bland::fft_shift_mag_square(h_padded)                      #
    # ------------------------------------------------------------------ #
    H = np.fft.fftshift(np.abs(np.fft.fft(h_padded)) ** 2)

    # ------------------------------------------------------------------ #
    # Step 4 — Slice centre region                                        #
    # mirrors: slice_spec{0, fine_per_coarse/2, full_res_length           #
    #                                           - fine_per_coarse/2}      #
    # number_coarse_channels_contributing is computed from pre-slice size #
    # ------------------------------------------------------------------ #
    n_contributing = H.shape[0] // fine_per_coarse - 1   # = num_coarse_channels - 1
    lo = fine_per_coarse // 2
    hi = full_res_length - fine_per_coarse // 2
    H = H[lo:hi]

    # ------------------------------------------------------------------ #
    # Step 5 — Fold: accumulate leakage from adjacent coarse channels     #
    # mirrors: H.reshape({n_contributing, fine_per_coarse})               #
    #          bland::sum(H, {0})                                         #
    # ------------------------------------------------------------------ #
    H = H.reshape(n_contributing, fine_per_coarse).sum(axis=0)

    # ------------------------------------------------------------------ #
    # Step 6 — Normalise                                                  #
    # ------------------------------------------------------------------ #
    H /= H.max()

    return H


# ------------------------------------------------------------------ #
# Usage                                                               #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    chan_file = '/mnt/d/data1/meerkat_test_data/channel_response_meerkat_131072_16.f32'
    chan_response = np.fromfile(chan_file, dtype=np.float32)


    configs = [
        ("hamming", 16, "Hamming 16-tap  ← bliss default"),
        ("hann",    16, "Hann    16-tap"),
        ("hamming",  4, "Hamming  4-tap  ← diagnosed from data"),
        ("hann",     4, "Hann     4-tap"),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    for window, taps, label in configs:
        H = gen_coarse_channel_response(
            fine_per_coarse=131072,
            num_coarse_channels=4,
            taps_per_channel=taps,
            window=window,
        )
        axes[0].plot(H, label=label, lw=0.8)
        axes[1].plot(20 * np.log10(np.maximum(H, 1e-10)), label=label, lw=0.8)

    for ax, title in zip(axes, ["Linear", "dB"]):
        ax.set_xlabel("Channel")
        ax.set_title(f"PFB response — {title}")
        ax.legend(fontsize=8)

    axes[0].scatter(np.arange(131072), chan_response, s=2, c='magenta')
    plt.tight_layout()
    plt.show()

    # Apply to your data:
    # H = gen_coarse_channel_response(131072, 4, 4, window="hann")
    # data_corrected = raw_spectrum / H


# Polyfit to remove trend???
# import numpy as np
# from scipy.ndimage import median_filter
# from numpy.polynomial import polynomial as P


# def derive_pfb_correction(
#     raw_spectrum,
#     pfb_assumed,
#     smooth_kernel=51,
#     poly_deg=6,
#     sigma_thresh=5.0,
#     return_diagnostics=False,
# ):
#     """
#     Empirically derive an improved PFB correction from a distorted spectrum.

#     After applying an imperfect PFB correction, the residual envelope encodes
#     the ratio pfb_true / pfb_assumed. This function estimates that ratio from
#     the corrected spectrum and folds it back into an improved PFB shape.

#     Parameters
#     ----------
#     raw_spectrum : np.ndarray
#         The RAW (uncorrected) spectrum, shape (n_channels,).
#     pfb_assumed : np.ndarray
#         Your current assumed PFB bandpass, shape (n_channels,). Must match
#         raw_spectrum in length.
#     smooth_kernel : int
#         Median filter width (in channels) for estimating the smooth envelope.
#         Should be >> ripple period but << band width. Default 51.
#     poly_deg : int
#         Degree of polynomial fit to the residual PFB envelope. Try 4–8.
#         Higher degrees track more complex shapes but risk overfitting RFI.
#         Default 6.
#     sigma_thresh : float
#         Channels deviating more than this many MAD-sigmas from the smooth
#         envelope are masked as RFI before fitting. Default 5.0.
#     return_diagnostics : bool
#         If True, return a dict of intermediate products for inspection.

#     Returns
#     -------
#     data_corrected : np.ndarray
#         raw_spectrum divided by the improved PFB estimate.
#     pfb_improved : np.ndarray
#         The improved PFB bandpass (pfb_assumed * residual envelope).
#     diagnostics : dict  (only if return_diagnostics=True)
#         Keys: 'corrected_initial', 'mask', 'pfb_residual', 'n_flagged'
#     """
#     assert len(raw_spectrum) == len(pfb_assumed), (
#         "raw_spectrum and pfb_assumed must have the same length"
#     )

#     n_chan = len(raw_spectrum)
#     chans = np.arange(n_chan, dtype=float)

#     # ------------------------------------------------------------------ #
#     # Step 1 — Apply your existing (imperfect) correction, then mask RFI  #
#     # ------------------------------------------------------------------ #
#     corrected_initial = raw_spectrum / pfb_assumed

#     smooth_envelope = median_filter(corrected_initial, size=smooth_kernel)
#     ratio = corrected_initial / np.where(smooth_envelope > 0, smooth_envelope, np.nan)

#     mad = np.nanmedian(np.abs(ratio - np.nanmedian(ratio)))
#     sigma = mad * 1.4826  # convert MAD → Gaussian σ equivalent
#     mask = np.abs(ratio - 1.0) < sigma_thresh * sigma  # True = good channels

#     n_flagged = np.sum(~mask)
#     print(f"[pfb_derive] Masked {n_flagged}/{n_chan} channels "
#           f"({100*n_flagged/n_chan:.1f}%) as RFI/outliers")

#     # ------------------------------------------------------------------ #
#     # Step 2 — Fit the smooth residual envelope on clean channels          #
#     # ------------------------------------------------------------------ #
#     coeffs = P.polyfit(chans[mask], corrected_initial[mask], deg=poly_deg)
#     pfb_residual = P.polyval(chans, coeffs)

#     # Normalise so the correction is neutral at the band centre
#     pfb_residual /= np.median(pfb_residual)

#     # ------------------------------------------------------------------ #
#     # Step 3 — Build improved PFB and apply to raw data                   #
#     # ------------------------------------------------------------------ #
#     pfb_improved = pfb_assumed * pfb_residual
#     data_corrected = raw_spectrum / pfb_improved

#     if return_diagnostics:
#         diagnostics = {
#             "corrected_initial": corrected_initial,
#             "mask": mask,
#             "pfb_residual": pfb_residual,
#             "n_flagged": n_flagged,
#         }
#         return data_corrected, pfb_improved, diagnostics

#     return data_corrected, pfb_improved
    