"""
Using Shat to help optimize the bin spacing to use in filtering the data for high occ

"""


import os
import numpy as np
import pandas as pd


def evaluate_bin_width(
    df,
    bin_width,
    freq_col="signal_frequency",
    offset_fraction=0.0,
    outlier_threshold=5.0,
):
    """
    Evaluate how well a frequency-bin width identifies unusually dense
    occupied bins.

    Empty bins are excluded from the background median and MAD calculation.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing hit frequencies.

    bin_width : float
        Frequency-bin width in the same units as freq_col.

    freq_col : str
        Name of the frequency column.

    offset_fraction : float
        Fraction of one bin width by which to shift the histogram edges.
        For example, 0.5 shifts the edges by half a bin.

    outlier_threshold : float
        Robust z-score threshold used to count outlier bins.

    Returns
    -------
    dict
        Histogram counts, edges, robust scores, and summary statistics.
    """

    if bin_width <= 0:
        raise ValueError("bin_width must be positive.")

    frequencies = (
        pd.to_numeric(df[freq_col], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy(dtype=float)
    )

    if frequencies.size == 0:
        raise ValueError(f"No valid frequencies found in {freq_col!r}.")

    offset = offset_fraction * bin_width

    lower_edge = (
        np.floor((frequencies.min() - offset) / bin_width)
        * bin_width
        + offset
    )

    upper_edge = (
        np.ceil((frequencies.max() - offset) / bin_width)
        * bin_width
        + offset
    )

    # Ensure at least one complete histogram bin.
    if np.isclose(lower_edge, upper_edge):
        upper_edge = lower_edge + bin_width

    bin_edges = np.arange(
        lower_edge,
        upper_edge + bin_width,
        bin_width,
    )

    counts, bin_edges = np.histogram(
        frequencies,
        bins=bin_edges,
    )

    # Define the background using only bins containing at least one hit.
    occupied = counts > 0
    occupied_counts = counts[occupied]

    if occupied_counts.size == 0:
        median_count = 0.0
        mad_count = 0.0
        robust_scale = 1.0
    else:
        median_count = float(np.median(occupied_counts))

        mad_count = float(
            np.median(
                np.abs(occupied_counts - median_count)
            )
        )

        robust_scale = 1.4826 * mad_count

        # MAD can be zero when many occupied bins have identical counts.
        if robust_scale == 0:
            robust_scale = np.sqrt(max(median_count, 1.0))

    # Empty bins receive NaN because they are not part of the occupied-bin
    # density comparison.
    robust_z = np.full(counts.shape, np.nan, dtype=float)

    robust_z[occupied] = (
        counts[occupied] - median_count
    ) / robust_scale

    if occupied_counts.size:
        max_robust_z = float(np.nanmax(robust_z))
        max_count = int(occupied_counts.max())
    else:
        max_robust_z = np.nan
        max_count = 0

    return {
        "bin_width": float(bin_width),
        "offset_fraction": float(offset_fraction),
        "median_count": median_count,
        "mad_count": mad_count,
        "robust_scale": float(robust_scale),
        "max_count": max_count,
        "max_robust_z": max_robust_z,
        "n_outlier_bins": int(
            np.count_nonzero(robust_z >= outlier_threshold)
        ),
        "n_total_bins": int(len(counts)),
        "n_occupied_bins": int(np.count_nonzero(occupied)),
        "occupied_fraction": float(np.mean(occupied)),
        "counts": counts,
        "edges": bin_edges,
        "robust_z": robust_z,
    }

def scan_bin_widths(
    df,
    candidate_widths,
    freq_col="signal_frequency",
    outlier_threshold=5.0,
    offsets=(0.0, 0.5),
):
    """
    Compare candidate frequency-bin widths using occupied-bin robust
    count statistics.

    Each width is tested with multiple histogram alignments. By default,
    the ordinary alignment and a half-bin shift are evaluated.

    The stable peak score is the lowest maximum robust z-score across
    the tested alignments. This favors widths whose strongest outliers
    persist when the bin edges move.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing hit frequencies.

    candidate_widths : iterable of float
        Candidate bin widths to test.

    freq_col : str
        Name of the frequency column.

    outlier_threshold : float
        Robust z-score above which an occupied bin is classified as
        unusually dense.

    offsets : iterable of float
        Histogram-edge offsets expressed as fractions of bin_width.

    Returns
    -------
    pandas.DataFrame
        One summary row per candidate bin width.
    """

    results = []

    for width in candidate_widths:
        if width <= 0:
            raise ValueError(
                f"All candidate widths must be positive; received {width}."
            )

        evaluations = [
            evaluate_bin_width(
                df=df,
                bin_width=width,
                freq_col=freq_col,
                offset_fraction=offset,
                outlier_threshold=outlier_threshold,
            )
            for offset in offsets
        ]

        peak_scores = np.array(
            [
                evaluation["max_robust_z"]
                for evaluation in evaluations
            ],
            dtype=float,
        )

        valid_peak_scores = peak_scores[
            np.isfinite(peak_scores)
        ]

        if valid_peak_scores.size:
            stable_peak_score = float(
                np.min(valid_peak_scores)
            )
            mean_peak_score = float(
                np.mean(valid_peak_scores)
            )
        else:
            stable_peak_score = np.nan
            mean_peak_score = np.nan

        results.append({
            "bin_width": float(width),

            # Conservative score: how strong is the peak under the
            # least favorable tested bin alignment?
            "stable_peak_score": stable_peak_score,

            # Average peak strength across alignments.
            "mean_peak_score": mean_peak_score,

            "mean_outlier_bins": float(np.mean([
                evaluation["n_outlier_bins"]
                for evaluation in evaluations
            ])),

            "median_count": float(np.mean([
                evaluation["median_count"]
                for evaluation in evaluations
            ])),

            "mean_max_count": float(np.mean([
                evaluation["max_count"]
                for evaluation in evaluations
            ])),

            "mean_occupied_bins": float(np.mean([
                evaluation["n_occupied_bins"]
                for evaluation in evaluations
            ])),

            "mean_total_bins": float(np.mean([
                evaluation["n_total_bins"]
                for evaluation in evaluations
            ])),

            "mean_occupied_fraction": float(np.mean([
                evaluation["occupied_fraction"]
                for evaluation in evaluations
            ])),
            "mean_robust_scale": float(np.mean([
            evaluation["robust_scale"]
            for evaluation in evaluations
            ])),
        })

    return (
        pd.DataFrame(results)
        .sort_values(
            "stable_peak_score",
            ascending=False,
            na_position="last",
        )
        .reset_index(drop=True)
    )


DIR = '/mnt/d/data1/meerkat_test_data/galacticPlane'

df_UHF_unique = pd.read_csv(os.path.join(DIR, 'mkstamp_UHF_unique.csv'))
df_L_unique = pd.read_csv(os.path.join(DIR, 'mkstamp_L_unique.csv'))
df_S_unique = pd.read_csv(os.path.join(DIR, 'mkstamp_S_unique.csv'))

test_widths = np.array([0.05, 0.1, 0.25, 0.5, 1.0])
bin_width_scores = scan_bin_widths(df_S_unique, candidate_widths=test_widths)
print(bin_width_scores)

count_threshold = (
    bin_width_scores["median_count"]
    + 5 * bin_width_scores["mean_robust_scale"]
)
print(count_threshold)

### Notes:
# 
# This worked pretty good. Given the distrubution of hits between UHF, L-band, and S-band