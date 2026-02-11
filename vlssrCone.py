from astroquery.vizier import Vizier
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, Column

import numpy as np

# Get sources around target that are brighter than 10 Jy. 
# Cone size specify units with u
def vlssrCone(targetName, coneSize): 
    vizier = Vizier()
    result = vizier.query_region(SkyCoord.from_name(targetName),
                                radius=coneSize,
                                catalog='VIII/97/catalog',
                                column_filters={'Sp': '>10'})
    
    return result

def target3C84():
    targetName = '3C 84'
    coneSize = 15 * u.degree #deg
    searchResults = vlssrCone(targetName, coneSize)

    targetSource = SkyCoord.from_name(targetName)
    foundSources = SkyCoord(frame='icrs', 
                          ra = np.array(searchResults[0]['RAJ2000']), 
                          dec = np.array(searchResults[0]['DEJ2000']), 
                          unit=(u.hourangle, u.deg))
    seps = targetSource.separation(foundSources)
    searchResults[0].add_column(Column(np.around(seps.deg, 4), name='seps', unit='deg'))
    searchResults[0].sort('seps')
    print(searchResults[0])
    searchResults[0].write('~/testPhaseCalTable.csv')
    return searchResults

def main(): 
    exoVLSSr = Table.read('/home/cat-work/work/SETI/swarmSETI/exoVLSSr_1.5deg.ecsv')
    exo10pc = exoVLSSr[exoVLSSr['sy_dist']<10]
    exo10pc.sort('Sp', reverse=True)
    print(exo10pc['ra_1', 'dec_1', 'Sp', 'sy_name', 'sy_snum', 'sy_pnum', 'sy_dist'][:10])
    return


if __name__ == "__main__":
    main()



