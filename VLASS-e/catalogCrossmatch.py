###
# This code hosts the catalog manipulation to assemble the master dataset 
#for the COSMIC VLASS exoplanet host toy project
###

import pandas as pd
import numpy as np
import pyvo as vo
import os
import sys

from astropy.table import Table, Column
from astropy.io import votable

from astroquery.xmatch import XMatch

### Collect data from the NASA Exoplanet Archive Stellar Hosts Table
# I've chosen an arbitrary set of columns to pull from the database
# In principle this can be slightly altered to return the anti-Table with sources missing a Gaia-DR2 ID

# This is not well organized. I kind of am on the edge of spinning off some of these functions into something for catTools



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

def xmatchStellarHosts():
    return


# The stellar hosts table has duplicate entries, we need em gone. 
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

# Convert to VOTable and write as xml 
def toVOTable(tabin, outname):
    outable = votable.from_table(tabin)
    return votable.writeto(outable, outname)
    
def id_to_stellar_params(gaia_dr2_id, stellarHostsTable, crossmatchTable, GaiaSrc, GaiaApsis):
    # This function would take an ID and the tables, then report the corresponding row from the Gaia Source and Apsis catalogs
    return

def main():
    ## First, simplify the stellar host table by removing dupes, saved. 
    # stellarHosts = removeDuplicates('/home/cat-work/work/SETI/cosmicStellarHosts/stellarHostsSeedTable.csv')

    ## New Catalog with no dupes
    fn_sh = '/home/cat-work/work/SETI/cosmicStellarHosts/sH_NoDupes.csv'
    ## CSV of full S.H. Table
    stellarHosts = pd.read_csv(fn_sh)
    stellarHostspd = stellarHosts.to_

    ## CSV of S.H. within 10 pc
    stellarHosts10pc = pd.read_csv('/home/cat-work/work/SETI/cosmicStellarHosts/stellarHosts_10pc.csv')

    ## DR2 to DR3 Mapping
    idCrossMatch = Table.read('/home/cat-work/work/SETI/cosmicStellarHosts/sH_GaiaDR2_DR3_crossmatch.ecsv')
    dr3Ids = idCrossMatch['dr3_source_id']
    
    # Make single list of IDs for TAP crossmatch selection
    csls = ','.join(map(str,dr3Ids))

    ## Gaia DR3 
    GaiaSource = Table.read('/home/cat-work/work/SETI/cosmicStellarHosts/sH_GaiaDR3Source_10pc.ecsv')
    GaiaApsis = Table.read('/home/cat-work/work/SETI/cosmicStellarHosts/sH_GaiaDR3Apsis_10pc.ecsv')
    
    return stellarHosts

if __name__ == "__main__":
    stellarHosts = main()