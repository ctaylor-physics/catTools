import glob
import time
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timezone, UTC
from astropy.time import Time
import matplotlib.pyplot as plt

from seticore import hit_capnp, stamp_capnp
from seticore.viewer import read_hits, read_stamps
from cosmic_database_analysis import sarfi

### This junk will build a pandas dataframe of the information contained in the fields
### below, excluding the large 'data' variable. 
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
            stamp_capnp.Stamp.read_multiple(f,traversal_limit_in_words=2**30,)
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

### Functions to gather similar information from hits files rather than stamps
def hit_file_metadata(hit_file):
    hitgen = read_hits(hit_file)
    rows = []
    for hit in hitgen:
        rows.append(hit.to_dict())
    return rows

def hit_metadata_dataframe(hit_files):
    rows = []
    for hit_file in hit_files:
        rows.extend(hit_file_metadata(hit_file))

    return pd.json_normalize(rows, sep='_')


def main():
    # Hardcoded for the Meerkat sample that is on the Berkeley data center
    # Would be better if this was generic enough to take some high level folder,
    # then parse for all .stamp files in subdirectories.
    
    DIR = '/datag/public/mk_sample_data'
    folders = [ 'blpn66',  'blpn68',  'blpn70',  'blpn72',  'blpn74',  'blpn76',  'blpn78'] #already did 64
    for fold in folders:
        stamp_files = sorted(glob.glob(DIR+f'/{fold}/*/*/*/seticore_search/*.stamps'))
        print(f"Number of Files blpn64: {len(stamp_files)}")
        stamp_df = stamp_metadata_dataframe(stamp_files)
        print(f"Number of Stamps: {len(stamp_df)}")
        stamp_df.to_csv(f'mkstamp_metadata_{fold}.csv')
    return

if __name__ == "__main__":
    main()




"""
# This parcel of data can be found on the berkeley cluster at:
# /datag/public/meerkat_sample_data/student_project
# and was used for testing purposes. 

DIR='/mnt/d/data1/meerkat_test_data/galacticPlane'

hit_files = sorted(glob.glob('/mnt/d/data1/meerkat_test_data/galacticPlane/*/seticore_search/*.hits'))
stamp_files = sorted(glob.glob('/mnt/d/data1/meerkat_test_data/galacticPlane/*/seticore_search/*.stamps'))

try:
    stamp_df = pd.read_csv('/mnt/d/data1/meerkat_test_data/galacticPlane/stamp_metadata_v1.csv').drop('Unnamed: 0')
    # hit_df = pd.read_csv('/mnt/d/data1/meerkat_test_data/galacticPlane/hit_metadata_v1.csv').drop('Unnamed: 0')
except:
    stamp_df = stamp_metadata_dataframe(stamp_files)
    stamp_df.to_csv('/mnt/d/data1/meerkat_test_data/galacticPlane/stamp_metadata_v1.csv')

    # hit_df = hit_metadata_dataframe(hit_files)
    # hit_df.to_csv('/mnt/d/data1/meerkat_test_data/galacticPlane/hit_metadata_v1.csv')
"""
