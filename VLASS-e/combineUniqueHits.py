import numpy as np
import pandas as pd
import glob
import os

DIR = "/home/cat-work/work/SETI/cosmicStellarHosts/databaseHits/observationIds_10pc"
all_files = glob.glob(DIR+'/obsId*/*_unique.pkl')

for file in all_files:
    incoming = pd.read_pickle(file)
    try:
        full_set = pd.concat((full_set, incoming))
    except:
        full_set = incoming

full_set.to_csv(os.path.join(DIR, 'uniqueFinalHits10pc_v1.csv'),index=False)
