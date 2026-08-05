import numpy as np
import ephem
from astropy.coordinates import EarthLocation, SkyCoord, get_sun, AltAz
from astropy.time import Time
import astropy.units as u
import matplotlib.pyplot as plt
from lsl.common import stations
from utils import STYLE_PATH, calcFigSize, get_colors

# Dates of VLA configuration 27A
date1 = "2027-06-25"
date2 = "2027-08-25"

# Configure observer
lwa1 = stations.lwa1
station_loc = lwa1.earth_location
obs = lwa1.get_observer()
obs.horizon='-12'
target = SkyCoord.from_name('SagA')

# Start Location
tstart = Time(f"{date1}T03:54:30", format='isot', location=station_loc, scale='utc')
tstart_aa = target.transform_to(AltAz(obstime=tstart, location=station_loc))
print(f"Start elevation: {tstart_aa.alt:.4f}")
tstart_lst = tstart.sidereal_time('apparent')
print(f"Start LST: {tstart_lst.to_string(unit=u.hour, sep='hms')}")
tstart_lst = tstart.sidereal_time('apparent')

# Stop Location 
tstop = Time(f"{date1}T09:36:00", format='isot', location=station_loc, scale='utc')
tstop_aa = target.transform_to(AltAz(obstime=tstop, location=station_loc))
print(f"Stop elevation: {tstop_aa.alt:.4f}")
tstop_lst = tstop.sidereal_time('apparent')
print(f"Stop LST: {tstop_lst.to_string(unit=u.hour, sep='hms')}")
tstop_lst = tstop.sidereal_time('apparent')

