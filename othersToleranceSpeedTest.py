import pandas as pd
import numpy as np
import time
from scipy.spatial import cKDTree

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # ✅ Import tqdm

def claudeKDTree(df_k218b, df_other):
    # Define tolerances
    freq_tol = 3e-5
    drift_tol = 0.0001

    # Normalize both dimensions by their tolerances so that
    # a Chebyshev distance of 1.0 = "within tolerance box"
    points_other = np.column_stack([
        df_other['signal_frequency'].values / freq_tol,
        df_other['signal_drift_rate'].values / drift_tol
    ])
    points_k218b = np.column_stack([
        df_k218b['signal_frequency'].values / freq_tol,
        df_k218b['signal_drift_rate'].values / drift_tol
    ])

    # Build KD-tree on df_other (done once, O(m log m))
    tree = cKDTree(points_other)

    # Query: find any neighbor within the tolerance box (Chebyshev / L-inf norm)
    # p=np.inf gives the L∞ norm, which is equivalent to checking both axes independently
    matches = tree.query_ball_point(points_k218b, r=1.0, p=np.inf, workers=-1)

    # Rows with NO match in df_other
    no_match_mask = np.array([len(m) == 0 for m in matches])
    df_unique_rows = df_k218b[no_match_mask].reset_index(drop=True)

    print(len(df_unique_rows))
    # df_unique_rows.to_pickle("k218b_unique_signals_c-band.pkl")
    return df_unique_rows

def chenoaTree(df_k218b, df_other):
    # Define tolerances
    freq_tol = 3e-5
    drift_tol = 0.0001

    # Extract relevant columns as NumPy arrays
    freq_other = df_other['signal_frequency'].values
    drift_other = df_other['signal_drift_rate'].values

    # Prepare output
    matched_indices = []

    # Process cband_k218b in chunks
    batch_size = 500  # adjust based on available RAM
    total = len(df_k218b)

    for start in tqdm(range(0, total, batch_size), desc="Processing Batches"):
    #for start in range(0, len(df_k218b), batch_size):
        end = min(start + batch_size, len(df_k218b))
        batch = df_k218b.iloc[start:end]

        freq_batch = batch['signal_frequency'].values
        drift_batch = batch['signal_drift_rate'].values

        # Broadcasting within batch
        freq_diff = np.abs(freq_batch[:, None] - freq_other[None, :]) <= freq_tol
        drift_diff = np.abs(drift_batch[:, None] - drift_other[None, :]) <= drift_tol

    #    # Find rows with NO match in other
        no_match = ~np.any(freq_diff & drift_diff, axis=1)

    #    # Collect indices
        matched_indices.extend(batch.index[no_match])

    # Final filtered DataFrame
    df_unique_rows = df_k218b.loc[matched_indices].reset_index(drop=True)
    print(len(df_unique_rows))

    return df_unique_rows


def main():
    df_k218b = pd.read_pickle('/home/cat-work/work/SETI/K2-18b/K2-18b-hits_mjd60220_K2-18b.pkl')
    df_other = pd.read_pickle('/home/cat-work/work/SETI/K2-18b/K2-18b-hits_mjd60220_Incoherent.pkl')

    start = time.perf_counter()
    claude = claudeKDTree(df_k218b, df_other)
    lap = time.perf_counter()
    chenoa = chenoaTree(df_k218b, df_other)
    end = time.perf_counter()

    print("claude:", lap-start)
    print("chenoa:", end-lap)
    
    return claude, chenoa

if __name__ == "__main__":
    claude, chenoa = main()