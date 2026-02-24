###
# This code hosts the catalog manipulation to assemble the master dataset 
#for the COSMIC VLASS exoplanet host toy project
###

import pandas as pd
import numpy as np
import pyvo as vo
import os
import sys
import matplotlib.pyplot as plt

from astropy.table import Table, Column
from astropy.io import votable

from astroquery.xmatch import XMatch

### Collect data from the NASA Exoplanet Archive Stellar Hosts Table
# I've chosen an arbitrary set of columns to pull from the database
# In principle this can be slightly altered to return the anti-Table with sources missing a Gaia-DR2 ID

# This is not well organized. I kind of am on the edge of spinning off some of these functions into something for catTools

## Step 1?
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

## Step 2?
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

## Step 3?
# This function would take in the table data from a TAP query and cross-match it with the Gaia 
def xmatchStellarHosts():
    return
    
def id_to_stellar_params(gaia_dr2_id, stellarHostsTable, crossmatchTable, GaiaSrc, GaiaApsis):
    # This function would take an ID and the tables, then report the corresponding row from the Gaia Source and Apsis catalogs
    return

def load_catalogs():
    """
    Load our catalog of Stellar Hosts and those within 10pc of Earth,
    a conversion table from Gaia DR2 to DR3 ids, 
    the main Gaia source table,
    and the Astrophysical solutions table. 
    """
    ## First, simplify the stellar host table by removing dupes, saved. 
    # stellarHosts = removeDuplicates('/home/cat-work/work/SETI/cosmicStellarHosts/stellarHostsSeedTable.csv')

    ## New Catalog with no dupes
    fn_sh = '/home/cat-work/work/SETI/cosmicStellarHosts/sH_NoDupes.csv'
    stellarHosts = pd.read_csv(fn_sh)
    
    ### Load catalogs: 
    # CSV of S.H. within 10 pc
    stellarHosts10pc = pd.read_csv('/home/cat-work/work/SETI/cosmicStellarHosts/stellarHosts_10pc.csv')
    # DR2 to DR3 Mapping
    idCrossMatch = Table.read('/home/cat-work/work/SETI/cosmicStellarHosts/sH_GaiaDR2_DR3_crossmatch.ecsv')
    ## Gaia DR3 
    GaiaSource = ( Table.read('/home/cat-work/work/SETI/cosmicStellarHosts/sH_GaiaDR3Source_10pc.ecsv')).to_pandas()
    GaiaApsis = ( Table.read('/home/cat-work/work/SETI/cosmicStellarHosts/sH_GaiaDR3Apsis_10pc.ecsv') ).to_pandas()

    ## Make single list of IDs for TAP crossmatch selection (just for inputting into TOPCAT)
    # dr3Ids = idCrossMatch['dr3_source_id']
    # csls = ','.join(map(str,dr3Ids))

    
    return stellarHosts, stellarHosts10pc, idCrossMatch, GaiaSource, GaiaApsis

def pie_demographics(gaia_apsis_table):
    """
    Make a pie chart of the stellar classes present in the sample
    
    As an aside, it seems like intuitively we would like this to not be M-dwarf heavy. Could also change to histogram :)
    """
    # Form here: 
    count = gaia_apsis_table["spectraltype_esphs"].value_counts()
    indices = count.index.to_list()
    vals = count.values.astype(str)
    labels = [f"{a} - {b}" for a,b in zip(indices, vals)]
    
    fig, ax = plt.subplots()
    ax.pie(count, labels=labels)
    plt.show()


    return

def main():
    stellarHosts, stellarHosts10pc, idCrossMatch, GaiaSource, GaiaApsis = load_catalogs()
    return


    
if __name__ == "__main__":
    main()



    ### MeerKAT study Gaia quality metric thresholds
    ##symbol(*GAIA Col Name*) > thresh
    # f_true ('parallax_over_error') > 20
    # sigma_G ('phot_g_mean_flux_over_error') > 50
    # sigma_rp ('phot_rp_mean_flux_over_error') > 20
    # sigma_bp ('phot_g_mean_flux_over_error') > 20 
    # I(bp-rp)/g ('phot_bp_rp_excess_factor1') < 1.3 + 0.06 * (G_bp-G_rp)**2
    # I(bp-rp)/g ('phot_bp_rp_excess_factor2') > 1.0 + 0.015 * (G_bp-G_rp)**2
    # n/a ('visibility_periods_used') >= 8 
    # Chi**2, nu ('astrometric_chi2 / (astrometric_n_good_obs_a - 5)') < 1.44*max(1,e**(-0.4(G-19.5)))

    ### Since the VLASS observations are continuous slews, you can reject signals that are present or not present in other antennas.
    ## Another method could be to compare the 'hits' between sources that are observed in the same drifting track or over some fixed duration.
    ## This is similar to the uniqueness filter that Chenoa has used for scans that use k2-18 or K2-18b at the phase center. 
    