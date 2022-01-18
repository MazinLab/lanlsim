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
R_0 = np.random.normal(scale=R_ref * R_variance, size=detector_shape)


def R(w, pixel=None, inverse=False):
    if inverse:
        w=1/w
    if pixel is not None:
        v = R_ref * R_wave / w + R_0[pixel]
    else:
        v = R_ref * R_wave / w[:, None, None] + R_0
    return v.clip(1, 25).value


class SpecgenInverse:
    def __init__(self, spectrum, wavedom=None):
        if wavedom is not None:
            assert wavedom[0]>=spectrum.wave.min() and wavedom[1]<=spectrum.wave.max()
            min_ndx = np.argwhere(spectrum.wave < wavedom[0]).max()
            max_ndx = np.argwhere(spectrum.wave > wavedom[1]).min()
            sl = slice(min_ndx, max_ndx)
            self._dom = wavedom
            flux = spectrum.flux[sl]
            wave = spectrum.wave[sl]
        else:
            self._dom = spectrum.wave.min(), spectrum.wave.max()
            wave = spectrum.wave
            flux = spectrum.flux
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


def simulate(pixel_count_image, specgen, flat_field, exp_time, residmap):
    print(f'Simulated dataset may take up to {pixel_count_image.sum() * 16 / 1024 ** 3:.2} GB of RAM')

    photons = np.recarray(pixel_count_image.sum(), dtype=PhotonNumpyType)
    photons[:] = 0
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

        # Drop photons from the flatfield
        if flat_field is not None:
            seen = np.random.uniform(size=n) < flat_field[pixel]
        else:
            seen = np.ones_like(arrival_times, dtype=bool)

        arrival_times = arrival_times[seen]

        # Generate wavelengths for the photons
        # ultimately the spectrum may vary with field position, but for now we'll use the global
        # specgen = SpecgenInverse(scene.sed_at_pixel(pixel))
        wavelengths = (specgen(n)[seen] * u.Angstrom).to(u.micron)
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
        measured_energies = np.random.normal(loc=energies, scale=energy_width)

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
        measured_wavelengths.clip(SATURATION_WAVELENGTH_NM)

        # Add photons to the pot
        sl = slice(observed, observed + arrival_times.size)
        photons.wavelength[sl] = measured_wavelengths
        photons.time[sl] = (arrival_times * 1e6)  # in microseconds
        photons.resID[sl] = residmap[pixel[::-1]]
        observed += arrival_times.size
    print(f'A total of {total_merged} photons had their energies '
          f'merged and {np.sum(total_missed)} were missed due to deadtime')
    return photons, observed


def simulate_observation(sp, psf_radius, exp_time, nd=DEFAULT_ND, desired_avg_countrate=None):
    field_extent = np.array([(-detector_shape / 2),
                             (+detector_shape / 2)]) * detector_sampling
    square_fov = np.diff(field_extent, axis=0).ravel().max()
    fov = np.diff(field_extent, axis=0).ravel()

    # Normalize the spectrum to desired flux
    atmosphere = AtmosphericTransmission()
    telescope = SubaruTransmission()
    telescope.convert('angstrom')
    atmosphere.convert('angstrom')

    # Multiply spectrum by earth's atmosphere anfd telescope transmission then renorm in bandpass
    bp = S.Box(detector_wavelength_center.to(u.Angstrom).value,
               detector_wavelength_width.to(u.Angstrom).value)  # angstroms, sigh


    spec = sp * telescope * atmosphere
    # spec.primary_area = primary_area.value  #does not appear needed

    if desired_avg_countrate:
        spec = spec.renorm(desired_avg_countrate / (R_wave / R_ref).to('AA').value, 'photlam', bp)

    specgen = SpecgenInverse(spec, wavedom=(detector_wavelength_range[0].to(u.Angstrom).value,
                                            detector_wavelength_range[1].to(u.Angstrom).value))

    bp.primary_area = primary_area.value
    full_bp = bp * telescope * atmosphere
    binset = np.arange(detector_wavelength_range[0].to(u.Angstrom).value,
                       detector_wavelength_range[1].to(u.Angstrom).value, 10)
    a = S.Observation(spec, bp, binset=binset)  # binset needed?
    a2 = S.Observation(sp, full_bp, binset=binset)  # binset needed?

    dc_throughput = splitter * 10 ** -nd * filter

    # plt.figure()
    # plt.plot(a2.wave, a2.flux, label='native')
    a2.convert('counts')
    plt.plot(a2.binwave/10, a2.binflux*dc_throughput, drawstyle='steps-mid', label='binned')
    # plt.xlim(5030, 5050)
    # plt.xlabel(a2.waveunits)
    # plt.ylabel(a2.fluxunits)
    # a.integrate()
    # a.countrate()

    # Patch in to MIRISIM
    # a.convert('um')
    # tab_sed = a.tabulate().resample(binset)
    # spectrum._PYSPsed = tab_sed
    # spectrum.goodwrange = tab_sed.wave[[0, -1]]

    # Create a Scene
    # source_pos = SkyCoord(0*u.deg, 0*u.deg)
    #
    # scene = skysim.Point()
    # scene.set_SED(spectrum)

    # Convolve scene with PSF and integrate to get a detector count image. Include throughput effects
    grip, sampled_psf = get_mec_psf(fov, detector_sampling, psf_radius)
    # scene.convolve_with(sampled_psf)
    photon_rate_image = a2.countrate() * sampled_psf * dc_throughput  # photons/s
    # photon_rate_cube = scene.buildcube(wavelengths=detector_wavelength_range, fov=fov, units='photon/s',
    #                                    spatial_sampling=detector_sampling)

    # Add in background
    pass

    # Compute a total number of emitted photons per pixel pos on sky
    pixel_count_image = np.random.poisson(photon_rate_image * exp_time)

    # v, b = np.histogram(x(1000000), bins=x._wave)
    # plt.plot(b[:-1], v / v.sum())

    # TODO We've computed the expected count rate image for our observing bandpass but photons outside of that region could
    # make it through and would combine as below. If so the total number we need to deal with for random draws needs to be
    # increased. Detector cant get photons below 950 but may register photons as below 950.

    from mkidpipeline.pipeline import generate_default_config

    cfg = generate_default_config(instrument='MEC')
    cfg.update('paths.out', './')

    try:
        from mkidpipeline.steps.flatcal import FlatSolution
        ff = FlatSolution('flatcal0_b4a953dd_0b159b84.flatcal.npz')
        flat_field = np.median(ff.flat_weights, 2).clip(0, 1).T
    except:
        log.info('Skipping flat field', exc_info=True)
        flat_field = None

    photons, observed = simulate(pixel_count_image, specgen, flat_field, exp_time, cfg.beammap.residmap)

    return pixel_count_image * flat_field, specgen, a, photons, observed
