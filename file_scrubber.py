import os
import pandas as pd
import numpy as np

"""
This is a helper file to collect the data that I need from the berkeley data center. 
"""

DIR = '/home/cat-work/work/SETI/cosmicStellarHosts/databaseHits/observationIds_10pc'
STAMPS_FN1 = os.path.join(DIR, "uniqueFinalHits10pc_scan_ids_v3.storage1.stamp_rels.csv")
STAMPS_FN2 = os.path.join(DIR, "uniqueFinalHits10pc_scan_ids_v3.storage2.stamp_rels.csv")
OBS_INFO_FN = os.path.join(DIR, "uniqueFinalHits10pc_scan_ids_v3.csv")

stamps1 = pd.read_csv(STAMPS_FN1)
stamps2 = pd.read_csv(STAMPS_FN2)
observation_info = pd.read_csv(OBS_INFO_FN)

print(f"stamps1 {len(stamps1)}")
print(f"stamps2 {len(stamps2)}")
print(f"observation_info {len(observation_info)}")