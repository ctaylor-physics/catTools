###
# This code hosts the catalog manipulation to assemble the master dataset 
#for the COSMIC VLASS exoplanet host toy project
###

import pandas as pd
import numpy as np
import pyvo as vo
import os
import sys

### Collect data from the NASA Exoplanet Archive Stellar Hosts Table
# I've chosen an arbitrary set of columns to pull from the database
# In principle this can be slightly altered to return the anti-Table with sources missing a Gaia-DR2 ID

# This works here, but I am having trouble with the full stack in async mode... 
def getStellarHosts():
    # Start TAP Connection
    tap_service = vo.dal.TAPService("https://exoplanetarchive.ipac.caltech.edu/TAP")
    query = """
        SELECT TOP 5
            sy_name, hostname, ra, dec, sy_snum, sy_pnum, sy_dist, st_refname, st_spectype, st_teff, st_rad, st_mass, st_met, gaia_dr2_id
        FROM
            stellarhosts
        WHERE 
            ra IS NOT NULL AND gaia_dr2_id IS NOT NULL
        """
    stellarHostTable = tap_service.search(query)
    return stellarHostTable.to_table()

def removeDuplicates(fn):
    # Full Stellar Hosts Catalog for sources with a Gaia DR2 ID
    stellarHosts = pd.read_csv(fn, header=27)
    parentd = os.path.dirname(fn)
    # Drop the duplicate rows keeping the first named row for a given Gaia DR2 ID
    sH_firstname = stellarHosts.drop_duplicates(subset=['gaia_dr2_id'], keep='first')
    # Clean up any remaining missing values if any remain
    sH_noNulls = sH_firstname.dropna(subset='gaia_dr2_id')
    outpath = os.path.join(parentd, 'sH_NoDupes.csv')
    sH_noNulls.to_csv(outpath)
    return sH_noNulls


def main():
    ## First, simplify the stellar host table by removing dupes, saved. 
    # stellarHosts = removeDuplicates('/home/cat-work/work/SETI/cosmicStellarHosts/stellarHostsSeedTable.csv')

    ## New Catalog with no dupes
    fn_sh = '/home/cat-work/work/SETI/cosmicStellarHosts/sH_NoDupes.csv'
    stellarHosts = pd.read_csv(fn_sh)
    
    print(stellarHosts[:10])
    return

if __name__ == "__main__":
    main()