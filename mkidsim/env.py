import warnings
import scipy.integrate
warnings.filterwarnings('ignore')

import astropy
import numpy as np
import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})
from .filterphot import mask_deadtime

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

from statsmodels.distributions.empirical_distribution import ECDF, StepFunction
from scipy import stats

from .psf import *
import pysynphot as S

from .spectra import AtmosphericTransmission, SubaruTransmission

MIN_TRIGGER_ENERGY = 1 / (1.5 * u.um)
SATURATION_WAVELENGTH_NM = 400
#Some basic definitions
deadtime = 10 * u.us
detector_shape = np.array([146, 140])
detector_sampling = 10.4 * u.mas
detector_wavelength_range = 950*u.nm, 1300*u.nm
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
    def __init__(self, spectrum):
        self._dom = spectrum._PYSPsed.wave.min(), spectrum._PYSPsed.wave.max()
        self._wave = np.zeros(spectrum._PYSPsed.wave.size+1)
        self._wave[1:] = spectrum._PYSPsed.wave
        self._wave[0] = 2*self._wave[1]-self._wave[2]
        self._tab_pdf = spectrum._PYSPsed.flux / spectrum._PYSPsed.flux.sum()
        # self._edf = StepFunction(spectrum._PYSPsed.wave, self._tab_pdf.cumsum(), side='right', sorted=True)
        cum_values = np.zeros(self._tab_pdf.size+1)
        cum_values[1:] = self._tab_pdf.cumsum() #prob that photon has lambda [i]<= _wave[i]
        self._edf = scipy.interpolate.interp1d(cum_values, self._wave)

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
