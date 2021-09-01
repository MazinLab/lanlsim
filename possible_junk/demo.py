"""
20% reflective Lambertian sphere located in Geosynchronous Earth Orbit (GEO).
-Gives spectrum (solar + earth) and flux for point source
-Background? (zodiacal+galacic)

Convert to incident spectrum
-atmospheric spectrum

photometric and spectroscopic R&D model: observe incident spectrum
- convolve with PSF
- "observe" with detector (medis vs mirisim)

"""

import warnings

warnings.filterwarnings('ignore')
import astropy
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.signal import convolve
from astropy.convolution import convolve

# import mirisim components
import numpy  as np
from mirisim import skysim, obssim
# import the scene configuration parser
from mirisim.config_parser import SceneConfig

# import astropy components
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
from astropy import units as u

# import MEDIS components
from inspect import getframeinfo, stack
import proper

from medis.params import sp, ap, tp
from medis.utils import dprint
import medis.optics as opx
import medis.aberrations as aber
import medis.adaptive as ao
import medis.coronagraphy as cg

# define a center RA and DEC for the image
CENTER_RA,CENTER_Dec = np.array([0,0]) * u.deg
width = np.ar0.01 * u.deg
height = 0.01 * u.deg
SpatialSampling = .1  # spatial sampling (in arcsec)
WavelengthRange = [0.4, 0.7]  # wavelength range to process (in microns)
WavelengthSampling = 0.01
zp = 25.7934
wref = 640.50 * u.nm

coord = SkyCoord(ra=CENTER_RA, dec=CENTER_Dec, frame='icrs')
r = Gaia.query_object_async(coordinate=coord, width=width, height=height)

# create Background emission object
xlim = (width / 2).to(u.arcsec).value
ylim = (height / 2).to(u.arcsec).value
FOV = np.array([[-xlim, xlim],
                [-ylim, ylim]])  # field of view [xmin,xmax],[ymin,ymax] (in arcsec)

Background = skysim.Background(level='low')
# initialise the galaxy with a center at (-1,-1) arcsec,
# an axial ratio of 2, an effective radius of 0.5 arcsec, etc
Stars = []
for star in r:
    # initialise the point source with a position
    # convert Gaia G mag to AB mag using the zeropoint provided:
    # https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/chap_cu5pho/sec_cu5pho_calibr/ssec_cu5pho_calibr_extern.html
    g_flux = star['phot_g_mean_flux']
    ab_mag = (0.9965 * (-2.5) * np.log10(g_flux) + zp) * u.ABmag

    offset_ra = (star['ra'] * u.degree - CENTER_RA).to(u.arcsec).value
    offset_dec = (star['dec'] * u.degree - CENTER_Dec).to(u.arcsec).value
    Star = skysim.Point(Cen=(offset_ra, offset_dec))

    # set properties of the SED, temp in K, wavelength in microns, flux in uJy
    Blackbody = skysim.sed.BBSed(Temp=float(star['teff_val']), wref=wref.to(u.micron).value,
                                 flux=ab_mag.to(u.uJy).value)

    # add the SED to the point source
    Star.set_SED(Blackbody)
    Stars.append(Star)

scene = Background
for i in Stars:
    scene += i
# overwrite = True enables overwriting of any previous version of the fits file
# with the same name as that given in the writecube command
scene.writecube(cubefits='shashank_example_scene.fits',
                FOV=FOV, time=0.0,
                spatsampling=SpatialSampling,
                wrange=WavelengthRange,
                wsampling=WavelengthSampling,
                overwrite=True)


#see mirisim.obssim.skycube.get_points_from_scene for convolution with PSF
