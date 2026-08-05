import os
import glob
import numpy as np 
import pandas as pd
import time
from pathlib import Path
from datetime import datetime, timezone, UTC
from astropy.time import Time

from seticore import hit_capnp, stamp_capnp
from seticore.viewer import read_hits, read_stamps
from cosmic_database_analysis import sarfi

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def first_pass():
    """
    This first pass was to see how well of a job Ross' code could track down my hits
    Now I think I need to do it manually...
    """
    DIR = '/srv/cosmicfs12/scratch/ctaylor/incoherentHits'
    fn_seed = "incoh_unique_signals_scan_ids.csv"
    ross_csv = "20260624incoh_unique_signals_scan_ids.csv.stamp_rels.csv"
    # ross_csv = "20260624incoh_unique_signals_scan_ids.csv.stamp_rels.v3.csv"

    # From Chenoa
    incoh = pd.read_csv(os.path.join(DIR,fn_seed))
    # From Ross - Doesn't match fch1 or fine_channel, but same coarse channel
    stamp_rels = pd.read_csv(os.path.join(DIR,ross_csv))

    # matches = incoh.join(ross,) 
    # df_matched = incoh.merge(ross, left_on='scan_id', right_on='scan_id_folder', how='inner')

    # 1. Find Stamp file in directory and pull the hit and stamp
    bad_freqs = np.zeros(len(stamp_rels))
    bad_drs = np.zeros(len(stamp_rels))
    n_channels_off = np.zeros(len(stamp_rels))

    good_hits = []
    good_stamps = []
    oneoff_hits = []
    oneoff_stamps = []
    bad_hits = []
    bad_stamps = []

    for i,row in stamp_rels.iterrows():

        # Get a hit
        hits_gen = read_hits(row.hit_file_uri)
        for index, hit in enumerate(hits_gen):
            if index == row.hit_file_local_enum:
                break

        # Get a Stamp
        stamps_gen = read_stamps(row.stamp_file_uri, find_recipe=True)
        for index, stamp in enumerate(stamps_gen):
                if index == row.stamp_file_local_enum:
                    assert(stamp != None)
                    assert(stamp.recipe != None)
                    break

        stamp_time = Time(datetime.fromtimestamp(stamp.stamp.tstart, tz=timezone.utc),  format='datetime').mjd

        
        if hit.signal.frequency != stamp.stamp.signal.frequency:
            diff = np.abs(hit.signal.frequency - stamp.stamp.signal.frequency)*1e6
            bad_freqs[i] = diff
            bad_drs[i] = hit.signal.driftRate - stamp.stamp.signal.driftRate
            n_channels_off[i] = np.abs(hit.signal.frequency - stamp.stamp.signal.frequency) / stamp.stamp.foff

            if diff < np.abs(2 * stamp.stamp.foff)*1e6:
                oneoff_hits.append(hit)
                oneoff_stamps.append(stamp)
            else:
                # Hit
                # print('-'*25)
                # print(f'Hit-in-Stamp Data: {sarfi.is_hit_in_stamp(hit, stamp)}')
                # print('-'*25)
                # print(f"Source {hit.filterbank.sourceName}")
                # print(f"Date {hit.filterbank.tstart}")
                # print(f"RA {hit.filterbank.ra}")
                # print(f"DEC {hit.filterbank.dec}")
                # print(f"Frequency {hit.signal.frequency}")
                # print(f"df: {diff} Hz")
                bad_hits.append(hit)
                bad_stamps.append(stamp)
                continue
            
        else:
            good_hits.append(hit)
            good_stamps.append(stamp)

    nz_bad_freqs = np.nonzero(bad_freqs)[0]

    print(f'Number of different frequencies: {len(nz_bad_freqs)}')
    print('bad_freqs')
    print(bad_freqs[nz_bad_freqs])
    print('bad_driftRates')
    print(bad_drs[nz_bad_freqs])

    print(f'number bad stamps: {len(bad_stamps)}')
    print(f'number good stamps: {len(good_stamps)}')
    print(f'number off_by_one stamps: {len(oneoff_stamps)}')


    hit_basis = good_hits
    stamp_basis = good_stamps
    """
    for i in range(len(hit_basis)):
        test_hit = hit_basis[i]
        test_stamp = stamp_basis[i]
        print(f'---------- {i} ----------')
        print(f"Hit    f: {test_hit.signal.frequency}")
        print(f"Hit   dr: {test_hit.signal.driftRate} [# of drift steps: {test_hit.signal.driftSteps}]")
        print(f"Stamp  f: {test_stamp.stamp.signal.frequency}")
        print(f"Stamp dr: {test_stamp.stamp.signal.driftRate} [# of drift steps: {test_stamp.stamp.signal.driftSteps}]")
        

        ### Plot
        test_stamp.show_classic_incoherent(save_to=f'test_stamp_classic_incoh{i:03}.png')
        i+=1
    # Shape Check   
    # power = np.square(stamp.real_array()).sum(axis=(2, 4)).transpose(2, 0, 1)
    """
    return

### Produced by Shat:

SIGNAL_FIELDS = [
    "frequency",
    "index",
    "driftSteps",
    "driftRate",
    "snr",
    "coarseChannel",
    "beam",
    "numTimesteps",
    "power",
    "incoherentPower",
]

STAMP_FIELDS = [
    "seticoreVersion",
    "sourceName",
    "ra",
    "dec",
    "fch1",
    "foff",
    "tstart",
    "tsamp",
    "telescopeId",
    "numTimesteps",
    "numChannels",
    "numPolarizations",
    "numAntennas",
    "coarseChannel",
    "fftSize",
    "startChannel",
    "schan",
    "obsid",
]


def stamp_metadata_dict(stamp, filepath=None, stamp_index=None):
    """
    Return scalar metadata from a seticore Stamp capnp object, excluding stamp.data.
    Includes nested signal fields as signal_<field>.
    """
    meta = {}

    if filepath is not None:
        meta["filepath"] = str(filepath)
    if stamp_index is not None:
        meta["stamp_file_local_enum"] = stamp_index

    for field in STAMP_FIELDS:
        meta[field] = getattr(stamp, field)

    # signal may be absent for CLI-extracted stamps according to the schema comments.
    # Accessing signal fields on an unpopulated struct may return defaults, so keep
    # an explicit flag if your downstream database needs to distinguish that case.
    try:
        signal = stamp.signal
        meta["has_signal"] = True
        for field in SIGNAL_FIELDS:
            meta[f"signal_{field}"] = getattr(signal, field)
    except Exception:
        meta["has_signal"] = False
        for field in SIGNAL_FIELDS:
            meta[f"signal_{field}"] = None

    return meta


def iter_stamp_metadata(stamp_file):
    stamp_file = Path(stamp_file)

    with stamp_file.open("rb") as f:
        for i, stamp in enumerate(
            stamp_capnp.Stamp.read_multiple(
                f,
                traversal_limit_in_words=2**30,
            )
        ):
            yield stamp_metadata_dict(stamp, filepath=stamp_file, stamp_index=i)


def stamp_metadata_dataframe(stamp_files):
    start = time.time()
    rows = []
    for i,stamp_file in enumerate(stamp_files):
        if i%1000 == 0:
            print(f'Time elapsed at file {i}: {time.time() - start:.3f} [{datetime.now(UTC)}]')
        rows.extend(iter_stamp_metadata(stamp_file))

    print(f'full file conv time = {time.time() - start:.3f}')
    return pd.DataFrame(rows)

def build_database():
    ### csv file stuff
    DIR = '/srv/cosmicfs12/scratch/ctaylor/incoherentHits'
    fn_seed = "incoh_unique_signals_scan_ids.csv"
    ross_csv = "20260624incoh_unique_signals_scan_ids.csv.stamp_rels.v3.csv" # most recent

    ### From Ross - Doesn't match fch1 or fine_channel, but same coarse channel
    # stamp_rels = pd.read_csv(os.path.join(DIR,ross_csv))

    ### greatest common directory path
    VLASS_TARGET = '/srv/cosmicfs8/vlass_target'

    ### Load what Chenoa gave me, add a column for an upper path for ease
    incoh = pd.read_csv(os.path.join(DIR, fn_seed))
    incoh['scan_id_dir'] = incoh.scan_id.str.rsplit('.', n=2).str[0]
    incoh['full_head_path'] = incoh.apply(lambda row: os.path.join(VLASS_TARGET, row['scan_id_dir'], row['scan_id'], row['scan_id']), axis=1)

    unique_path_inx = incoh.full_head_path.drop_duplicates().index
    unique_path_rows = incoh.loc[unique_path_inx]

    ### Get filenames for all the stamps that we need to grab
    stamp_files = []
    total_files = 0
    for i, row in unique_path_rows.iterrows():
        glob_root = row.full_head_path + '.' + row.tuning + '*.stamps'
        stamp_f_found = sorted(glob.glob(glob_root))
        if len(stamp_f_found) > 0:
            stamp_files.extend(stamp_f_found)
            total_files += len(stamp_f_found)

    print(f'total files found {total_files}')

    out_df = stamp_metadata_dataframe(stamp_files)
    
    out_df.to_pickle(os.path.join(DIR, 'stamp_metadata_v1.pkl'))
    return

if __name__ == "__main__":
    # first_pass()
    build_database()
# Step 1. Collect a matching of the hits and stamps that are available

# Step 2. Build new Table of incoherent hits info


# Step 3. Compare to the table that Chenoa gave me 


### extras ###

    #    if hit.signal.frequency != stamp.stamp.signal.frequency:
    #        diff = np.abs(hit.filterbank.tstart - stamp_time)*86400
    #        print(f"dt: {diff} sec")

    #    if hit.signal.frequency != stamp.stamp.signal.frequency:
    #        diff = np.abs(stamp.stamp.ra - hit.filterbank.ra)
    #        print(f"dRA: {diff}")

    #    if hit.signal.frequency != stamp.stamp.signal.frequency:
    #        diff = np.abs(hit.filterbank.dec - stamp.stamp.dec)
    #        print(f"dDec: {diff}")
