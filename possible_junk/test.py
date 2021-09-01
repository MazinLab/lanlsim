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
import pyximport; pyximport.install()

import warnings
import scipy.integrate
warnings.filterwarnings('ignore')
from filterphot import deadtime_mask
import astropy
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.convolution import convolve
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
import astropy.units as u

# import mirisim components
from mirisim import skysim, obssim
from mirisim.config_parser import SceneConfig



# define a center RA and DEC for the image
CENTER_RA = 83.825 * u.deg
CENTER_Dec = -5.4 * u.deg
coord = SkyCoord(CENTER_RA, CENTER_Dec)
# coord = SkyCoord.from_name('HIP48331')

array_shape = (146, 140)

ssampling = 10.4*u.uas  # spatial sampling
wrange = (0.4, 0.7)  # wavelength range to process (in microns)
R400 = 12
zp = 25.7934
wref = 640.50 * u.nm
exp_time=10

array_shape = np.array(array_shape)
width, height = (array_shape*ssampling).to('deg')
wsamp = wrange[0]/R400

FOV = np.array([[-width.to('arcsec').value, width.to('arcsec').value],
                [-height.to('arcsec').value, height.to('arcsec').value]])/2  # field of view [xmin,xmax],[ymin,ymax] (in arcsec)


scene = skysim.Background(level='low')

d = Gaia.query_object(coordinate=coord, width=30*u.arcsec, height=30*u.arcsec)[0]

# initialise the point source with a position
# convert Gaia G mag to AB mag using the zeropoint provided:
# https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/chap_cu5pho/sec_cu5pho_calibr/ssec_cu5pho_calibr_extern.html
ab_mag = (0.9965 * 2.5 * np.log10(d['phot_g_mean_flux']) + zp) * u.ABmag

# offset_ra = (d['ra'] * u.degree - CENTER_RA).to(u.arcsec).value
# offset_dec = (d['dec'] * u.degree - CENTER_Dec).to(u.arcsec).value
star = skysim.Point(Cen=(0, 0))

# set properties of the SED, temp in K, wavelength in microns, flux in uJy add the SED to the point source
star.set_SED(skysim.sed.BBSed(5500 if d['teff_val'].mask else d['teff_val'], wref=wref.to(u.micron).value,
                              flux=ab_mag.to(u.uJy).value))
scene += star

# overwrite = True enables overwriting of any previous version of the fits file
# with the same name as that given in the writecube command
scene.writecube(cubefits='example_scene.fits',
                FOV=FOV, spatsampling=ssampling.to('arcsec').value,
                wrange=wrange, wsampling=wsamp, overwrite=True)
hdu = fits.open('example_scene.fits')[0]  #.data [wave, nx, ny]

atmosphere = 1.0
telescope = 1.0
psf = 1.0
#Cube is in spectral flux density
scene*=atmosphere
scene*=telescope
scene.apply_psf(psf)

scene.resample(pixels, spatsampling)
photon_rate = scene.integrate(wrange, out='photons', fov, sampling)

n_photons = np.random.poisson(photon_rate)

for pixel,n in np.ndenumerate(n_photons):

    raw_arrival_times = np.random.uniform(0,exp_time, n)

    raw_energies = scene.sed.sample(n)
    import mirisim.obssim.skycube as skcube
    skcube.get_points_from_scene


    #Times within 1us merge
    bins, energies = np.histogram(raw_arrival_times, bins=exp_time*1e6, weights=raw_energies)

    #Times within 10us are dropped
    # nonzero 1us bin indices where ndx >= ndx-9 is  in array are dead

    measured_energy = np.random.normal(loc=energies, sigma=sigma_from(pixel_res[pixel], energies))

#Need to slice up arrival times per pixel
    deadtime_mask(raw_arrival_times)


#Option 2 Total Energy over wavelength band -> total number of photons (how, average energy?) - > draw from SED

# a normal distribution with a mean of 0 and standard deviation of 1
n = stats.norm(loc=0, scale=1)
# draw some random samples from it
sample = n.rvs(100)
# compute the ECDF of the samples
def ecdf(sample):
    # convert sample to a numpy array, if it isn't already
    sample = np.atleast_1d(sample)
    # find the unique values and their corresponding counts
    quantiles, counts = np.unique(sample, return_counts=True)
    # take the cumulative sum of the counts and divide by the sample size to
    # get the cumulative probabilities between 0 and 1
    cumprob = np.cumsum(counts).astype(np.double) / sample.size
    return quantiles, cumprob
qe, pe = ecdf(sample)
# evaluate the theoretical CDF over the same range
q = np.linspace(qe[0], qe[-1], 1000)
p = n.cdf(q)

# Generate a KDE from the empirical sample
sample_pdf = scipy.stats.gaussian_kde(orig_sample_data)
# Sample new datapoints from the KDE
new_sample_data = sample_pdf.resample(10000).T[:,0]
