import numpy as np
import pandas as pd
from utils import calcFigSize, STYLE_PATH

from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.path as mpath
star = mpath.Path.unit_regular_star(5)

from PIL import Image

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

import astropy.units as u
from astropy.coordinates import Angle,Longitude,SkyCoord
from astropy.wcs import WCS
from astropy.visualization.wcsaxes import WCSAxes

def calcSeps(targets):
    fullseps = targets[:,None].separation(targets[None,:])
    uppert = np.triu_indices(len(targets), k=1)
    seps = fullseps[uppert]

    print('Mean separation between Calibrators:', seps.mean())
    print('STD of separation between Calibrators:', seps.std())
    return

def main():
    ## READ TARGETS 
    all_targets = pd.read_pickle("/home/cat-work/work/SETI/cosmicStellarHosts/under40pc_sample/under40pc_full.pkl")
    targets = pd.read_pickle("/home/cat-work/work/SETI/cosmicStellarHosts/under10pc_full.pkl")
    hits_10pc = pd.read_csv('/home/cat-work/work/SETI/cosmicStellarHosts/databaseHits/craig_10pc_hits.csv')
    unique_sources = hits_10pc.source_name.unique()
    hits_targets = targets[targets["Gaia_ID"].isin(unique_sources.astype(str))]

    # flux = np.array(targets['Sp (Jy)'], dtype=float)


    ## Sort Targets between obs and un-obs
    all_targs = SkyCoord(ra=all_targets.rastr, dec=all_targets.decstr, 
                    frame='icrs',unit=(u.hourangle, u.deg))
    main_targs = SkyCoord(ra=targets.rastr, dec=targets.decstr, 
                    frame='icrs',unit=(u.hourangle, u.deg))
    extra_targs = SkyCoord(ra=hits_targets.rastr, dec=hits_targets.decstr, 
                    frame='icrs',unit=(u.hourangle, u.deg))

    print(f"all {len(all_targs)}")
    print(f"main {len(main_targs)}")
    print(f"extra {len(extra_targs)}")
    ## Separation Statistics
    # print('Full survey')
    # calcSeps(all_targs)
    # print('Observed Sources')
    # calcSeps(main_targs)

    ### Plot Style for Publication
    figs = calcFigSize(name="CQG",columns='onecol')
    plt.style.use(STYLE_PATH)
    title = 'LWA Swarm Sources'

    ### Plotting Now!
    ## Aitoff projection (works)
    mtra = np.radians(main_targs.ra.value)
    mtra[mtra>np.pi] -= 2*np.pi
    mtdec = np.radians(main_targs.dec.value)

    etra = np.radians(extra_targs.ra.value)
    etra[etra>np.pi] -= 2*np.pi
    etdec = np.radians(extra_targs.dec.value)

    atra = np.radians(all_targs.ra.value)
    atra[atra>np.pi] -= 2*np.pi
    atdec = np.radians(all_targs.dec.value)

    fig = plt.figure()
    plt.title('Exoplanets visible to VLASS', y=0.9)
    plt.axis('off')
    ax = fig.add_subplot(1,1,1,projection='aitoff')
    ax.scatter(atra, atdec, c='grey', label='dist $<40$ pc',s=1.5, alpha = 0.75, marker='.')
    ax.scatter(mtra, mtdec, c='cyan', label='dist $<10$ pc',s=1.5, alpha = 0.75, marker='p')
    ax.scatter(etra, etdec, c='darkorange', label='This Sample',s=10, marker='*', linewidths=0.5)
    ax.set_xlabel('Right Ascension')
    ax.set_ylabel('Declination')
    ax.grid(True)
    plt.legend(loc=(0.9,0.9), fontsize=8)
    plt.savefig("/home/cat-work/work/SETI/cosmicStellarHosts/databaseHits/observationIds_10pc/paper_plots/exoVLASS_target_map3.png")
    plt.show()


    return main_targs, extra_targs

if __name__ == "__main__":
    main_targs, extra_targs = main()
    

########################################################### JUNKYARD
# ## Aitoff projection (works)
# mtra = np.radians(main_targs.ra.value)
# mtra[mtra>np.pi] -= 2*np.pi
# mtdec = np.radians(main_targs.dec.value)

# etra = np.radians(extra_targs.ra.value)
# etra[etra>np.pi] -= 2*np.pi
# etdec = np.radians(extra_targs.dec.value)

# fig = plt.figure()
# ax.set_title('MPL only')
# ax = fig.add_subplot(1,1,1,projection='aitoff')
# ax.scatter(mtra, mtdec) #, c='k', label='observed',s=0.3*flux[:24])
# ax.scatter(etra, etdec) #, c='r', label='extras >40Jy',s=0.3*flux[24:])

# ax.grid(True)
# plt.show()

# ## TEST MAKING WCS
# wcs_input_dict = {
#     'CTYPE1': 'RA---MER', 
#     'CUNIT1': 'deg', 
#     'CDELT1': 0.5, 
#     'CRPIX1': 1.0, 
#     'CRVAL1': 0.0, 
#     'NAXIS1': 720,
#     'CTYPE2': 'DEC--MER', 
#     'CUNIT2': 'deg', 
#     'CDELT2': -1, 
#     'CRPIX2': 1, 
#     'CRVAL2': 0.0, 
#     'NAXIS2': 300
# }
# wcs_testdict = WCS(wcs_input_dict)

#Test1

# fig = plt.figure()
# ax = plt.subplot(projection=wcs_testdict)
# ax.scatter(main_targs.ra, main_targs.dec, c='k')
# ax.scatter(extra_targs.ra, extra_targs.dec, c='r')
# plt.xlabel(r'RA')
# plt.ylabel(r'DEC')
# overlay = ax.get_coords_overlay('icrs')
# overlay.grid(color='red', ls='dotted')


## PLOTTING (square)
# test_cm = {}
# test_cm['name'] = 'ra', 'dec'
# test_cm['type'] = 'longitude', 'latitude'
# test_cm['wrap'] = '360' * u.deg, None
# test_cm['unit'] = u.hourangle,u.deg
# test_cm['format_unit'] = u.hourangle,None
# Linear scaled scatter of target Flux
# fig = plt.figure(figsize=(10,10))
# ax = WCSAxes(fig,[0.1,0.1,0.8,0.8], aspect='auto',coord_meta=test_cm)
# fig.add_axes(ax)
# ax.scatter(main_targs.ra, main_targs.dec, c='k', label='Observed',s=0.3*flux[:24])
# ax.scatter(extra_targs.ra, extra_targs.dec, c='r', label='Unobserved (>40Jy)',s=0.3*flux[24:])
# ax.set_xlim(0,360.0)
# ax.set_ylim(-10,90)
# ax.grid()
# plt.title(title)
# plt.xlabel(r'RA')
# plt.ylabel(r'DEC')
# plt.legend()
# plt.show()

## Blocky Robinson projection
# # Cartopy
# trnsf = ccrs.Robinson()
# fmt = FuncFormatter(lambda x,pos: round(Longitude(x, unit=u.deg).hour))

# fig2,ax = plt.subplots(figsize=figs, subplot_kw={'projection':trnsf}) #layout='constrained')

# ax.set_extent([-180, 180, -40,90],crs=ccrs.PlateCarree())
# gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False, linewidth=0.4,alpha=0.7, linestyle='--')
# gl.xlocator = mticker.FixedLocator(np.arange(-180,180,15))
# gl.top_labels = False
# gl.xformatter=fmt
# gl.yformatter = LatitudeFormatter(direction_label=False)
# gl.xlabel_style = {'fontsize': 8}
# gl.ylabel_style = {'fontsize': 8}


# ax.xaxis.set_major_formatter(lambda x,pos: Angle(x).hour)
# #ax.set_title(title)
# ax.scatter(main_targs.ra.value,main_targs.dec.value,c='k',label='Observed',s=1, transform=ccrs.PlateCarree())
# ax.scatter(extra_targs.ra.value, extra_targs.dec.value, c='r', label='Unobserved',s=1, transform=ccrs.PlateCarree())
# ax.set_title('LWA Swarm Calibrator Survey Targets')

# plt.gcf().text(0.465, 0.22, 'RA (J2000)', fontsize=8)
# plt.gcf().text(0.055, 0.42,'DEC (J2000)', rotation='vertical', fontsize=8)
# plt.legend(loc=(0.9,0.9))
# # plt.savefig('/home/cat-work/pictures/swarmsources_robinson_legend5.png')
# plt.show()
