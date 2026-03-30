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
import matplotlib.pyplot as plt
from lsl.common import stations

def arbObserver(year, latitude, longitude, elev, hor = "-12"): #use float degrees and meters, string date
    obs = ephem.Observer()
    obs.lat = str(latitude)
    obs.lon = str(longitude)
    obs.date = f"{year}/01/01 00:00:00.00"
    obs.elevation = elev
    obs.horizon = hor
    return obs

def dayAlts(observer, mjd_date):
    dayAlts = []
    mjd_hours = np.linspace(mjd_date, mjd_date+1, 25)
    mjd_Times = Time(mjd_hours[:-1], format='mjd')
    for t in mjd_Times:
        observer.date = t.to_datetime()
        sun.compute(observer)
        dayAlts.append(np.degrees(sun.alt))
    return np.array(dayAlts)


# LWA1 trial location:

sun = ephem.Sun()

lwa1 = stations.lwa1
obs = lwa1.get_observer()
obs.horizon = '-12'

year = 2026
tstart = Time(f"{year}-01-01T00:00:00.00", format="isot")
mjd_dates = np.arange(tstart.mjd, tstart.mjd+390, 30)
# hours = np.arange(0,24, 1)

today_alts = dayAlts(obs, 61119)
print(today_alts)
fig, axs = plt.subplots(1,1)
axs.set_xlabel('UTC Hour')
axs.set_ylabel('Solar Elevation')
axs.scatter(np.linspace(61119, 61119+1, 25)[:-1],
            today_alts)
plt.show()

# # Year Compute, approximately monthly dates
# solarAlts = []
# for day in mjd_dates:
#         solarAlts.append(dayAlts(day))






