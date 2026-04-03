import os
import pandas as pd
import numpy as np
from seticore import viewer
from cosmic_database_analysis import sarfi


"""
This is a helper file to collect the data that I need from the berkeley data center. 
"""

def main(DIR, storage1 = False, storage2 = False):

    if storage1:
        STAMPS_FN = os.path.join(DIR, "uniqueFinalHits10pc_scan_ids_v3.storage1.stamp_rels.csv")
        stamps = pd.read_csv(STAMPS_FN)
        print(f"stamps1 {len(stamps)}")
        storage_name = 'storage1'
    elif storage2:
        STAMPS_FN = os.path.join(DIR, "uniqueFinalHits10pc_scan_ids_v3.storage2.stamp_rels.csv")
        stamps = pd.read_csv(STAMPS_FN)
        print(f"stamps2 {len(stamps)}")
        storage_name = 'storage2'

    else:
        raise ValueError('no storage node selected')
    
    OBS_INFO_FN = os.path.join(DIR, "uniqueFinalHits10pc_scan_ids_v3.csv")
    observation_info = pd.read_csv(OBS_INFO_FN)
    print(f"observation_info {len(observation_info)}")

    new_table_rows = []

    for i,row in stamps.iterrows():
        # load stamp
        stamps_gen = viewer.read_stamps(row.stamp_file_uri, find_recipe=True)
        for index, stamp in enumerate(stamps_gen):
            if index == row.stamp_file_local_enum:
                assert(stamp != None)
                assert(stamp.recipe != None)
                break

        hits_gen = viewer.read_hits(row.hit_file_uri)
        for index,hit in enumerate(hits_gen):
            if index == row.hit_file_local_enum:
                break

      
        if sarfi.is_hit_in_stamp(hit, stamp):
            # obs_info_row = observation_info.loc[(observation_info.signal_frequency == hit.signal.frequency) & (observation_info.signal_drift_rate == hit.signal.driftRate)]
            obs_info_row = observation_info.loc[(observation_info.signal_frequency.round(5) == round(hit.signal.frequency,5))]
            if len(obs_info_row) > 0 :
                new_table_rows.append(pd.concat([obs_info_row.squeeze(), row]))
            else: 
                print(f' found nothing at sig_freq = {hit.signal.frequency}, drift_rate = { hit.signal.driftRate}')
        else:
            print('Missing Row!')
            print(row)

    combined_table = pd.DataFrame(new_table_rows)
    combined_table.sort_values(by='id')
    combined_table.to_csv(os.path.join(DIR, f"uniqueFinalHits10pc_{storage_name}.csv"))
            


if __name__ == "__main__":
    # Must change for storage node 2 I think?
    DIR = '/srv/cosmicfs0/scratch/ctaylor/'
    main(DIR, storage1 = True)
