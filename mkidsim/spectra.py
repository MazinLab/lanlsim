import pysynphot as S
import specutils
import astropy.units as u
import numpy as np
from astropy.io import fits

_atm = None


def _get_atm():
    global _atm
    if _atm is not None:
        return _atm
    x = np.genfromtxt('data/transdata_0.5_1_mic')
    y = np.genfromtxt('data/transdata_1_5_mic')
    x = x[x[:, 1] > 0]
    x[:, 0] = 1e4 / x[:, 0]
    y = y[y[:, 1] > 0]
    y[:, 0] = 1e4 / y[:, 0]
    x = x[::-1]
    y = y[::-1]
    trans = np.vstack((x[x[:, 0] < 1.111], y[y[:, 0] >= 1.111]))
    atmosphere = trans[(trans[:, 0] > .8) & (trans[:, 0] < 1.4)]
    _atm = atmosphere[:, 0], atmosphere[:, 1]
    return _atm


def AtmosphericTransmission():
    w, t = _get_atm()
    return S.ArrayBandpass(w, t, name='AtmosphericTrans', waveunits='um')


def SubaruTransmission():
    w = np.linspace(500, 1500, 10000)
    t = np.linspace(1, .95, 10000)
    return S.ArrayBandpass(w, t, name='SubaruTrans', waveunits='nm')

# sp = S.Icat('k93models', 6440, 0, 4.3)
