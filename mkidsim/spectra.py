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


def SubaruTransmission(reflectivity=.75):
    w = np.linspace(500, 1500, 10000)
    t = np.linspace(1, .95, 10000)*reflectivity
    return S.ArrayBandpass(w, t, name='SubaruTrans', waveunits='nm')


def PhoenixModel(teff, feh, logg, desired_magnitude=None):
    sp = S.Icat('phoenix', teff, feh, logg)
    if desired_magnitude is not None:
        sp = sp.renorm(desired_magnitude, 'vegamag', S.ObsBandpass('johnson,v'))
    return sp


def SatModel(file='data/Time0000_Pose0000.lc', materials='all', exclude=tuple(), swap=tuple()):
    with open(file) as f:
        header = f.readline()
    material_labels = header.strip(',\n').split(',')
    radiance_vector = np.loadtxt(file, delimiter=',', skiprows=1)
    wavelengths = radiance_vector[:, 0]*1e4  # um to AA
    radiance = radiance_vector[:, 1:]  # W/cm^2/sr/um

    #1 m satillite at GEO ~36e6 m ~1/36e6 rad-> pi*(1/36e6/2)**2 ~6.06e-16 sr
    radiance *= np.pi*(1/36e6/2)**2  #W/cm^2/um
    # 1e4 AA/um
    radiance /= 1e4  # #W/cm^2/AA
    # 1 W = e7 erg/s
    radiance *= 1e7  #erg/s/cm^2/AA
    #boils down to ~ 3.283e-2
    #flam =erg/s/cm^2/AA

    if materials is 'all':
        materials = material_labels

    use = [material_labels.index(m) for m in materials if m not in exclude]

    if swap:
        renorm = 1+radiance[:, material_labels.index(swap[1])].mean()/radiance[:, material_labels.index(swap[0])].mean()
        print(f'Swapping {swap[0]} for {swap[1]} by renormalizing to {renorm}')
        radiance[:, material_labels.index(swap[0])] *= renorm

    from pysynphot import ArraySpectrum
    sp = ArraySpectrum(wavelengths, radiance[:, use].sum(1), fluxunits='flam')
    return sp
