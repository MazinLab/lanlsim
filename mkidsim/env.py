import warnings
import scipy.integrate
warnings.filterwarnings('ignore')

import astropy
import numpy as np
import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})
from .filterphot import mask_deadtime
from logging import getLogger
log = getLogger('photonsim')
import scipy.interpolate

import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.convolution import convolve
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia

import astropy.constants as c
from astropy.modeling.functional_models import AiryDisk2D
import astropy.units as u

# import mirisim components
from mirisim import skysim, obssim
from mirisim.config_parser import SceneConfig
from mkidcore.binfile.mkidbin import PhotonNumpyType
from statsmodels.distributions.empirical_distribution import ECDF, StepFunction
from scipy import stats

from .psf import *
import pysynphot as S

from .spectra import AtmosphericTransmission, SubaruTransmission, PhoenixModel, SatModel

DEFAULT_ND=2.5
splitter = .1
filter = .9
MIN_TRIGGER_ENERGY = 1 / (1.5 * u.um)
SATURATION_WAVELENGTH_NM = 400
#Some basic definitions
deadtime = 10 * u.us
detector_shape = np.array([146, 140])
detector_sampling = 10.4 * u.mas
detector_wavelength_range = 950*u.nm, 1375*u.nm
detector_wavelength_center = (detector_wavelength_range[1]+detector_wavelength_range[0])*.5
detector_wavelength_width = detector_wavelength_range[1]-detector_wavelength_range[0]
primary_area = (53*u.m**2).to('cm^2')
R_ref = 10
R_wave = 800 * u.nm
R_variance = .2
# from collections import defaultdict
# defaultdict(lambda x)
R_0 = np.random.normal(scale=R_ref * R_variance, size=detector_shape)
R_at = lambda w: R_ref * R_wave / w


def R_at(w, inverse=False):
    if inverse:
        w=1/w
    v = R_ref * R_wave / w
    return v.value


def R(w, pixel=None, inverse=False):
    if inverse:
        w=1/w
    if pixel is not None:
        v = R_ref * R_wave / w + R_0[pixel]
    else:
        v = R_ref * R_wave / w[:, None, None] + R_0
    return v.clip(2).value

from specutils import Spectrum1D
from specutils.manipulation import FluxConservingResampler
from specutils.manipulation import gaussian_smooth

fcrzf = FluxConservingResampler(extrapolation_treatment='zero_fill')
flam = u.erg / u.s / u.cm ** 2 / u.angstrom


def Spectrum1D_to_counts(s, primary_area):
    """compute counts in Spectrum1D"""
    counts = primary_area * u.cm ** 2 * s.flux * np.diff(s.spectral_axis)[0]
    counts /= astropy.constants.h * astropy.constants.c / s.spectral_axis
    return counts.si


def ObsFluxCheck(incident_sp, wavelengths):
    """Compute the from an incident pysynphot spectrum using specutils"""
    bp = Spectrum1D(flux=incident_sp.bandpass.throughput * u.dimensionless_unscaled,
                    spectral_axis=incident_sp.bandpass.wave * u.angstrom)
    sp = Spectrum1D(flux=incident_sp.spectrum.flux * flam, spectral_axis=incident_sp.spectrum.wave * u.angstrom)
    # sp = Spectrum1D(flux=sim.spec(t, pose).rawspec * flam, spectral_axis=sim.spec(t, pose).wavelengths)
    bp = fcrzf(bp, sp.wavelength)
    incident = bp * sp
    stdev = detector_wavelength_center / R_ref / 2.355 / np.diff(incident.spectral_axis)[0]  # Rref is fwhm
    smoothed = gaussian_smooth(incident, stddev=stdev.si.value)
    smoothedbin = fcrzf(smoothed, wavelengths * 10 * u.angstrom)  # f['CUBE_EDGES'].data.edges * 10 * u.angstrom)
    return Spectrum1D_to_counts(smoothedbin, incident_sp.bandpass.primary_area)


class SpecgenInverse:
    def __init__(self, spectrum, wavedom=None):
        if wavedom is not None:
            assert wavedom[0]>=spectrum.binwave.min() and wavedom[1]<=spectrum.binwave.max()
            min_ndx = np.argwhere(spectrum.binwave <= wavedom[0]).max()
            max_ndx = np.argwhere(spectrum.binwave >= wavedom[1]).min()
            sl = slice(min_ndx, max_ndx+1)
            self._dom = wavedom
            flux = spectrum.binflux[sl]
            wave = spectrum.binwave[sl]
        else:
            self._dom = spectrum.binwave.min(), spectrum.binwave.max()
            wave = spectrum.binwave
            flux = spectrum.binflux
        self._wave = np.zeros(wave.size+1)
        self._wave[1:] = wave
        self._wave[0] = 2*self._wave[1]-self._wave[2]
        self._tab_pdf = flux / flux.sum()
        # self._edf = StepFunction(spectrum._PYSPsed.wave, self._tab_pdf.cumsum(), side='right', sorted=True)
        cum_values = np.zeros(self._tab_pdf.size+1)
        cum_values[1:] = self._tab_pdf.cumsum() #prob that photon has lambda [i]<= _wave[i]
        self._edf = scipy.interpolate.interp1d(cum_values, self._wave)
        self.spectrum = spectrum

    def __call__(self, size=None):
        return self._edf(np.random.uniform(size=size))


class Specgen(stats.rv_continuous):
    def __init__(self, spectrum):
        super(Specgen, self).__init__()
        self._spectrum = spectrum._PYSPsed
        fprob = spectrum.flux / spectrum.flux.sum()
        self._edf = StepFunction(spectrum.wave, fprob.cumsum(), side='right', sorted=True)
        self._spectrum_norm = spectrum.integrate() * 1e4

    def _cdf(self, x):
        ret = np.zeros_like(x)
        hi = x > self._spectrum.wave[-1]
        use = (x >= self._spectrum.wave[0]) & (~hi)
        ret[use] = self._edf(x[use])
        ret[hi] = 1
        return ret

    def _pdf(self, x):
        ret = np.zeros_like(x)
        use = (x >= self._spectrum.wave[0]) & (x <= self._spectrum.wave[-1])
        ret[use] = self._spectrum(x[use]*1e4)/self._spectrum_norm
        return ret

    def _stats(self):
        return 0., 0., 0., 0.


def simulate(pixel_count_image, specgen, exp_time, residmap):
    print(f'Simulated dataset may take up to {pixel_count_image.sum() * 16 / 1024 ** 3:.2} GB of RAM')

    photons = np.recarray(pixel_count_image.sum(), dtype=PhotonNumpyType)
    photons[:] = 0
    photons.weight[:] = 1.0
    observed = 0
    total_merged = 0
    total_missed = []
    # Compute photon arrival times and wavelengths for each photon
    for pixel, n in np.ndenumerate(pixel_count_image):
        if not n:
            continue

        # Generate arrival times for the photons
        arrival_times = np.random.uniform(0, exp_time, size=n)
        arrival_times.sort()

        # Generate wavelengths for the photons
        wavelengths = (specgen(n) * u.Angstrom).to(u.micron)
        energies = 1 / wavelengths

        # merge photon energies within 1us
        to_merge = (np.diff(arrival_times) < 1e-6).nonzero()[0]
        if to_merge.size:
            cluster_starts = to_merge[np.concatenate(([0], (np.diff(to_merge) > 1).nonzero()[0] + 1))]
            cluser_last = to_merge[(np.diff(to_merge) > 1).nonzero()[0]] + 1
            cluser_last = np.append(cluser_last, to_merge[-1] + 1)  # inclusive
            for start, stop in zip(cluster_starts, cluser_last):
                merge = slice(start + 1, stop + 1)
                energies[start] += energies[merge].sum()
                energies[merge] = np.nan
                total_merged += energies[merge].size

        # Determine measured energies
        #  Use the relevant R value for each photon pixel combo
        energy_width = energies / R(energies, pixel=pixel, inverse=True)
        measured_energies = np.random.normal(loc=energies, scale=energy_width/2.35482)

        # Filter those that wouldn't trigger
        will_trigger = measured_energies / u.um > MIN_TRIGGER_ENERGY
        if not will_trigger.any():
            continue

        # Drop photons that arrive within the deadtime
        detected = mask_deadtime(arrival_times[will_trigger], deadtime.to(u.s).value)

        missed = will_trigger.sum() - detected.sum()
        total_missed.append(missed)

        arrival_times = arrival_times[will_trigger][detected]
        measured_wavelengths = 1000 / measured_energies[will_trigger][detected]
        measured_wavelengths.clip(SATURATION_WAVELENGTH_NM, out=measured_wavelengths)

        # Add photons to the pot
        sl = slice(observed, observed + arrival_times.size)
        photons.wavelength[sl] = measured_wavelengths
        photons.time[sl] = (arrival_times * 1e6)  # in microseconds
        photons.resID[sl] = residmap[pixel[::-1]]
        observed += arrival_times.size
    print(f'A total of {total_merged} photons had their energies '
          f'merged and {np.sum(total_missed)} were missed due to deadtime, {observed} observed.')
    return photons, observed


def incident_spectrum(sp, nd=DEFAULT_ND, desired_avg_countrate=None):
    """
    Takes a pysenphot spectrum

    returns a tuple of
    - SpecgenInverse,
    - pysynphot.Observation of the spectrum using a bandpass the is the product of
        the instrument, telescope, atmosphere, and dc_throughput bandpasses
    - the dc throughput use
    :param sp:
    :param nd:
    :param desired_avg_countrate:
    :return:
    """
    angstrom_domain = (detector_wavelength_range[0].to(u.Angstrom).value,
                       detector_wavelength_range[1].to(u.Angstrom).value)
    # Normalize the spectrum to desired flux
    atmosphere = AtmosphericTransmission()
    telescope = SubaruTransmission()
    telescope.convert('angstrom')
    atmosphere.convert('angstrom')

    # Multiply spectrum by earth's atmosphere anfd telescope transmission then renorm in bandpass
    bp = S.Box(detector_wavelength_center.to(u.Angstrom).value,
               detector_wavelength_width.to(u.Angstrom).value)  # angstroms, sigh

    spec = sp * telescope * atmosphere

    if desired_avg_countrate:
        spec = spec.renorm(desired_avg_countrate / (R_wave / R_ref).to('AA').value, 'photlam', bp)

    dc_throughput = splitter * 10 ** -nd * filter
    bp.primary_area = primary_area.value

    full_bp = bp * telescope * atmosphere * dc_throughput

    binset = np.arange(angstrom_domain[0], angstrom_domain[1]+1, 1)
    a = S.Observation(spec, bp, binset=binset)  # binset needed?  expect a.countrate() to be 446.9e6 for a 7.59 mag satillite
    observation = S.Observation(sp, full_bp, binset=binset)  # binset needed?

    a.convert('counts')
    observation.convert('counts')

    return SpecgenInverse(a, wavedom=angstrom_domain), observation, dc_throughput


def simulate_observation(sp, psf_radius, exp_time, nd=DEFAULT_ND, desired_avg_countrate=None):
    field_extent = np.array([(-detector_shape / 2),
                             (+detector_shape / 2)]) * detector_sampling
    fov = np.diff(field_extent, axis=0).ravel()

    specgen, a2, dc_throughput = incident_spectrum(sp, nd=nd, desired_avg_countrate=desired_avg_countrate)

    # Convolve scene with PSF and integrate to get a detector count image. Include throughput effects
    grid, sampled_psf = get_mec_psf(fov, detector_sampling, psf_radius)

    try:
        from mkidpipeline.pipeline import generate_default_config
        cfg = generate_default_config(instrument='MEC')
        cfg.update('paths.out', './')
        from mkidpipeline.steps.flatcal import FlatSolution
        ff = FlatSolution('flatcal0_b4a953dd_0b159b84.flatcal.npz')
        flat_field = np.median(ff.flat_weights, 2).clip(0, 1).T
    except:
        log.info('Skipping flat field', exc_info=True)
        flat_field = None

    photon_rate_image = a2.countrate() * sampled_psf * flat_field  # photons/s

    # Add in background
    pass

    # Compute a total number of emitted photons per pixel pos on sky
    pixel_count_image = np.random.poisson(photon_rate_image * exp_time)

    # We've computed the expected count rate image for our observing bandpass but
    #  photons outside of that region could make it through and would combine as below.
    #  If so the total number we need to deal with for random draws needs to be increased.
    #  Detector cant receive photons below 950 but may register photons as below 950.

    photons, observed = simulate(pixel_count_image, specgen, exp_time, cfg.beammap.residmap)
    return pixel_count_image * flat_field, specgen, a2, dc_throughput, flat_field, photons[:observed], photons.size


from goeslib import SatLib
lib = SatLib('./data/goeslib')
silversim = lib['MKID_20201013_Silver']
goldsim = lib['MKID_20201013_Kapton']
