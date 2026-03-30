"""
This is a helper function to calculate the maximum observable drift rate for a
given target observed by the VLASS mission. 

There will be some assumptions taken into consideration here and I will do my best
to list them below to keep track:

 - sin(co-latitude) = sin(inclination) = 1 -> this constrains the transmitter host rotation 
    term to be maximized. If the apparent orbit inclination is know, we could further constrain this term. 


"""

import astropy.constants as AstroConst

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import pyvo as vo
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

P_sidereal = 86164.0905 # seconds
restFrequencies = np.array([2, 3, 4])*1e9 # Hz (VLASS uses S-band which covers 2-4 GHz)
R_satellites = {"LEO": 1e6, "MEO":1.7e6, "GEO":3.5786e6} # meters

def contrib_rotEarth():
    return ( 4 * np.pi**2 * AstroConst.R_earth.value ) / ( P_sidereal**2 )

def contrib_orbEarth():
    return ( AstroConst.G.value * AstroConst.M_sun.value) / ( AstroConst.au.value**2 )

def contrib_Earth_Sat(R_type = "LEO"):
    R_sat = R_satellites[R_type]
    return ( AstroConst.G.value * AstroConst.M_earth.value) / ( R_sat**2 )

def contrib_rotExoplanet(ExoRadius, ExoMass, colat = np.pi/2, inclination = np.pi/2): # meters and seconds
    P_breakup = ( 2 * np.pi * ExoRadius**(1.5)) / np.sqrt(AstroConst.G.value * ExoMass)
    return ( 4 * np.pi**2 * ExoRadius ) / ( P_breakup**2 )

def contrib_orbExoplanet(OrbitRadius, M_hoststar): # meters and kilograms
    return ( AstroConst.G.value * M_hoststar) / ( OrbitRadius**2 )

def getExoplanetsTable():
    ExoTable = NasaExoplanetArchive.query_criteria(
    table="pscomppars", 
    select="pl_name, hostname, tic_id, ra, dec, pl_orbsmax, pl_orbper, pl_rade, pl_masse, pl_bmasse, st_spectype, st_teff, st_mass, st_rad", 
    where="hostname in ('61 Vir', 'CD Cet', 'GJ 433', 'GJ 876', 'GJ 896 A', 'Wolf 1061', 'eps Eri')")
    return ExoTable.to_pandas()

def calcMaxSystemDriftRate(ExoplanetsTable, hostname, rot = True):
    hostDF = ExoplanetsTable.loc[ExoplanetsTable.hostname == hostname]
    drift_rates = []
    for index, row in hostDF.iterrows():
        ## Correct with solar system constants
        # Mass
        ExoMass = row.pl_bmasse * AstroConst.M_earth.value

        # Radius
        ExoRadius = row.pl_rade * AstroConst.R_earth.value

        # Orbit
        OrbitRadius = row.pl_orbsmax * AstroConst.au.value

        # Host Mass
        M_Host = row.st_mass * AstroConst.M_sun.value
        if rot:
            calcDR = restFrequencies/AstroConst.c.value * ( contrib_rotEarth() + 
                                                        contrib_orbEarth() + 
                                                        contrib_rotExoplanet(ExoRadius, ExoMass) + 
                                                        contrib_orbExoplanet(OrbitRadius, M_Host))
        else: 
            calcDR = restFrequencies/AstroConst.c.value * ( contrib_rotEarth() + 
                                                        contrib_orbEarth() + 
                                                        contrib_orbExoplanet(OrbitRadius, M_Host))
        drift_rates.append(calcDR)
    
    maxDR = np.argmax(np.mean(drift_rates, axis=1))
    print(f"the maximum acceptable drift rates for {hostname} at [2, 3, 4] GHz is:\n{drift_rates[maxDR]} Hz/s\n")
    return #drift_rates[maxDR]


def main():
    ### Prerequisite stuffs
    # finalHits = pd.read_csv('/home/cat-work/work/SETI/cosmicStellarHosts/databaseHits/craig_10pc_hits.csv')
    # finalSources = "Gaia DR2 " + finalHits.source_name.unique().astype(str)

    # under10pc = pd.read_pickle("/home/cat-work/work/SETI/cosmicStellarHosts/under10pc_full.pkl")
    # finalSourcesGaia = under10pc[under10pc.gaia_dr2_id.isin(finalSources)]

    # stellarHosts = pd.read_csv('/home/cat-work/work/SETI/cosmicStellarHosts/sH_NoDupes.csv')
    # finalStellarHosts = stellarHosts[stellarHosts.gaia_dr2_id.isin(finalSources)]

    # ExoTable = getExoplanetsTable
    ExoTable = pd.read_pickle('/home/cat-work/work/SETI/cosmicStellarHosts/planetSpecs_10pc_sources.pkl')
    hosts = ExoTable.hostname.unique()

    # Earth Only Test
    maximumDR_Earth = restFrequencies/AstroConst.c.value * ( contrib_rotEarth() + contrib_orbEarth())
    maximumDR_Earth = restFrequencies/AstroConst.c.value * ( contrib_rotEarth() + contrib_orbEarth())

    for host in hosts:
        calcMaxSystemDriftRate(ExoTable, host, True)

    return

if __name__ == "__main__":
    main()