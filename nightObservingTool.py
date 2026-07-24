"""
My vision for this tool is that I can plug in an observatory location on the ground (Lat/Lon)
then get back a plot that shows the LST ranges that are possible to observe when the Sun is down
as a quick reference rather. 

This should save me some time when considering what sources will have optimal low frequency
performance without having to make an SDF for every idea that I have. 

The output is an image based plot showing colored cells, x = RA, y = month, with: 
    green cell if the sun at -12 degrees elevation,
    yellow cell if below 0 degrees elevation, 
    red cell if the sun is above 0 degrees elevation
"""

import numpy as np
import ephem
from astropy.coordinates import EarthLocation, SkyCoord, get_sun
from astropy.time import Time
import astropy.units as u
import matplotlib.pyplot as plt
from lsl.common import stations
from utils import STYLE_PATH, calcFigSize, get_colors

def arbObserver(year, latitude, longitude, elev, hor = "-12"): #use float degrees and meters, string date
    obs = ephem.Observer()
    obs.lat = str(latitude)
    obs.lon = str(longitude)
    obs.date = f"{year}/01/01 00:00:00.00"
    obs.elevation = elev
    obs.horizon = hor
    return obs

def dayAlts(observer, telescope_loc, mjd_date, sun): #Earth Location of telescope
    # Correct the starting UTC to be in hours LST rather than hours UTC
    mjd_date = Time(mjd_date, format='mjd', location=telescope_loc)
    start_lst = mjd_date.sidereal_time('apparent')
    hour_offset = (0.0 - start_lst.value) % 24
    corr_utc = mjd_date + ( hour_offset * u.hour * 0.997 )
    # These are normalized to LST hours of 0-23 (roughly)
    lst_hours = Time(corr_utc + (np.arange(0,24,1) * u.hour * 0.99726957), location = telescope_loc)
    dayAlts = []
    for t in lst_hours:
        observer.date = t.to_datetime()
        sun.compute(observer)
        dayAlts.append(np.degrees(sun.alt))
    return np.array(dayAlts)


def main(year, station):
    sun = ephem.Sun()
    station_loc = station.earth_location
    obs = station.get_observer()
    obs.horizon = '-12'

    tstart = Time(f"{year}-01-01T00:00:00.00", format="isot")
    mjd_dates = np.arange(tstart.mjd, tstart.mjd+365, 1)

    # Year Compute, approximately monthly dates
    stack_days = []
    for day in mjd_dates:
        day_alts = dayAlts(obs,station_loc, day, sun)
        stack_days.append(day_alts)
    stack_days = np.array(stack_days)
    # %%
    # Format plot
    figs = calcFigSize(name="CQG",columns='onecol')
    plt.style.use(STYLE_PATH)

    fig, ax = plt.subplots(1,1,figsize=figs, constrained_layout=True)
    plt.title(f'{year} Annual Observing Plot for LWA1')
    im = ax.imshow(stack_days, origin='lower', cmap = 'afmhot', aspect='auto',alpha=0.8)

    ax.set_xlabel('LST Hour (R.A.)')
    ax.set_xticks(np.arange(0,24,1))

    ax.set_ylabel('Month')
    ax.set_yticks(np.array([1,32,60,91,121,152,182,213,244,274,305,335]), 
                labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    cbar = plt.colorbar(im, fraction = 0.046 * (stack_days.shape[0]/stack_days.shape[1]), pad = 0.04 )
    cbar.ax.set_ylabel('Solar Elevation')

    # LWA Interferometry Primary Calibrators

    for name in ['3C 48', '3C 147', '3C 196', '3C 295', '3C 380', '3C 409']:
        RA = SkyCoord.from_name(name).ra.value / 15.0
        obsday = np.argmin(stack_days[:,int(round(RA))])
        ax.scatter(RA, obsday, color='cyan', s=10, marker = "s")
        ax.annotate(name, xy = (RA,obsday), xycoords='data',
                    xytext=(4,-3), textcoords='offset points', color='cyan', fontsize=10)

    ax.grid(False)
    plt.savefig(f'/home/cat-work/pictures/{year}_{station.name}_observing_plot.png')
    plt.show()

    print(f"For each RA on this plot:\n  - If yellow at date and target RA, the transit will follow the sun. \n\n So to observe at night, select a target RA that is dark and use the local UTC for nighttime :)")

if __name__ == '__main__':
    year = 2026
    station = stations.lwasv
    main(year, station)

    # Probably would be better to input the observer and earth location, but thats ok.
    # I'm only using the LWA/VLA at this point, so they are equivalent. 





