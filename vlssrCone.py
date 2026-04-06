from astroquery.vizier import Vizier
import astropy.units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.table import Table, Column

import numpy as np
import pandas as pd

# Get sources around target that are brighter than 10 Jy.
def vlssrCone(targetName, ra=None, dec=None, coneSize= 1.0 * u.degree):
    vizier = Vizier()
    try:
        sk = SkyCoord.from_name(targetName)
    except:
        sk = SkyCoord(frame='icrs',
                          ra = np.array(ra),
                          dec = np.array(dec),
                          unit=(u.hourangle, u.deg))

    result = vizier.query_region(SkyCoord.from_name(targetName),
                                radius=coneSize,
                                catalog='VIII/97/catalog',
                                column_filters={'Sp': '>10'})

    return result

### Old LoTSS Catalog
# def LoTSS_Cone(targetName, ra=None, dec=None, coneSize= 1.0 * u.degree):
#     vizier = Vizier()
#     try:
#         sk = SkyCoord.from_name(targetName)
#     except:
#         sk = SkyCoord(frame='icrs',
#                           ra = np.array(ra),
#                           dec = np.array(dec),
#                           unit=(u.hourangle, u.deg))
#     result = vizier.query_region(SkyCoord.from_name(targetName),
#                                 radius=coneSize,
#                                 catalog='J/A+A/622/A1',
#                                 column_filters={'Sint': '>10'})

#     return result

def target3C84():
    ## Define target and cone
    targetName = '3C 84'
    coneSize = 15 * u.degree #deg
    targetSource = SkyCoord.from_name(targetName)

    ## Get a cone search result
    searchResults = vlssrCone(targetName, coneSize)
    # searchResults = LoTSS_Cone(targetName, coneSize)

    ## Create SkyCoord for all found sources
    foundSources = SkyCoord(frame='icrs',
                          ra = np.array(searchResults[0]['RAJ2000']),
                          dec = np.array(searchResults[0]['DEJ2000']),
                          unit=(u.hourangle, u.deg))
    
    ## Calculate the separation in degrees from our target
    seps = targetSource.separation(foundSources)

    ## Append the separation onto our results dataframe
    searchResults[0].add_column(Column(np.around(seps.deg, 4), name='seps', unit='deg'))
    searchResults[0].sort('seps')
    print(searchResults[0])

    ## Write results out to file
    # searchResults[0].write('~/testPhaseCalTable2.csv')

    return searchResults




# This was build using TOPCAT
def main():
    searchResults = target3C84()
    return


if __name__ == "__main__":
    main()




