"""
I would like to improve my algorithm for rejecting bad stamps for the larger dataset. Here I will work on that :)
"""

import os
import pandas as pd
from seticore import viewer

STAMP_PATH = '/home/cat-work/work/SETI/cosmicStellarHosts/databaseHits/observationIds_10pc/final_10pc_candidates'


def collect_stamps(dataframe):
    ### Collect Stamps
    out_stamps = []
    for i,row in dataframe.iterrows():
        ## Fetch Stamp
        stampfn = os.path.join(STAMP_PATH, os.path.basename(row.stamp_file_uri_oi))
        stamps_gen = viewer.read_stamps(stampfn, find_recipe=True)
        for index, stamp in enumerate(stamps_gen):
            if index == row.stamp_file_local_enum:
                assert(stamp != None)
                assert(stamp.recipe != None)
                dataframe.append(stamp)
                break
    return out_stamps


### Get Voyager Stamps:
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

### Load in the classifier table
SAMPLEDIR = '/home/cat-work/work/SETI/cosmicStellarHosts/databaseHits/observationIds_10pc'
classified_stamps = pd.read_csv(os.path.join(SAMPLEDIR, 'uniqueFinalHits10pc_classified.csv'))
classes = classified_stamps.classification.unique()

### Easiest to me is broadband RFI
broadband_signals = classified_stamps[classified_stamps.classification == 'broadband transient rfi']
five_random_inx = [1,8,13,33,56]
bb_stamps = collect_stamps(broadband_signals.iloc[five_random_inx])


