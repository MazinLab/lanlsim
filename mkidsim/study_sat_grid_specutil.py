import matplotlib.pyplot as plt
import numpy as np
from logging import getLogger
import logging
from mkidsim.env import *
from mkidpipeline.steps.buildhdf import buildfromarray
from mkidpipeline.pipeline import generate_default_config
from mkidcore.corelog import getLogger
import mkidpipeline.pipeline as pipe
import mkidpipeline.config as config
from mkidpipeline.definitions import MKIDObservation, MKIDWCSCal, MKIDOutput, MKIDOutputCollection
import mkidpipeline.steps as steps
from datetime import datetime
from mkidpipeline.photontable import Photontable as P
from mkidsim.env import R_ref, R_wave


from specutils import Spectrum1D
from specutils.manipulation import FluxConservingResampler, LinearInterpolatedResampler, \
    SplineInterpolatedResampler
from specutils.manipulation import gaussian_smooth

fcrzf = FluxConservingResampler(extrapolation_treatment='zero_fill')

flam = u.erg / u.s / u.cm ** 2 / u.angstrom


logging.basicConfig()
log = getLogger('photonsim')
getLogger('mkidcore', setup=True, logfile=f'mkidpipe_{datetime.now().strftime("%Y-%m-%d_%H%M")}.log')
getLogger('mkidpipe').setLevel('DEBUG')
getLogger('mkidpipeline').setLevel('DEBUG')
getLogger('mkidcore').setLevel('DEBUG')

exp_time = 10
psf_radius = .3 * u.arcsec / 2

from goeslib import SatLib
lib = SatLib('./data/goeslib')
silversim = lib['MKID_20201013_Silver']
goldsim = lib['MKID_20201013_Kapton']

times = range(0,71,8)
poses = (0,4,5,8)


cfg = pipe.generate_default_config(instrument='MEC')
cfg.update('paths.out', './out/')
cfg.update('paths.tmp', './scratch/')
cfg.update('paths.database', './db/')
with open('pipe.yaml', 'w') as f:
    config.yaml.dump(cfg, f)
config.configure_pipeline(cfg)
outputs = MKIDOutputCollection('simout.yaml', datafile='simdata.yaml')
dataset = outputs.dataset

data=np.zeros((2,len(times),len(poses), 5, 140,146))

i,j,k=0,0,0
sim=(silversim, goldsim)[i]
t=times[j]
pose=poses[k]


for i, sim in enumerate((silversim, goldsim)):
    fig, axes = plt.subplots(len(times), len(poses), sharex=True, figsize=(15, 10),
                             gridspec_kw=dict(wspace=0.0, hspace=0.0, top=0.95, bottom=0.05, left=0.08, right=0.95))
    ax = None
    for j, t in enumerate(times):
        for k, pose in enumerate(poses):
            # if k+j>0:
            #     continue
            plt.sca(axes[j, k])
            fitsfile=f'./out/{sim.name}_{t}_{pose}/{sim.name}_{t}_{pose}_scube.fits'
            h5file=f'./out/{sim.name}_{t}_{pose}.h5'
            pt=P(h5file)
            f=fits.open(fitsfile)
            data[i, j, k]= f['science'].data

            wave = np.diff(f['CUBE_EDGES'].data.edges) / 2 + f['CUBE_EDGES'].data.edges[:-1]
            specgen, incident_sp, dc_throughput = incident_spectrum(sim.spec(t, pose).pysynphot, nd=3.5)
#================

            sp = sim.spec(t, pose).pysynphot
            result = simulate_observation(sp, psf_radius, exp_time, nd=3.5)
            pixel_count_image_ff, specgen, a2, dc_throughput, flat_field, photons, photonssize = result
            phots=pt.query(startw=950, stopw=1375)
            bp = Spectrum1D(flux=incident_sp.bandpass.throughput * u.dimensionless_unscaled,
                           spectral_axis=incident_sp.bandpass.wave * u.angstrom)
            sp = Spectrum1D(flux=incident_sp.spectrum.flux * flam, spectral_axis=incident_sp.spectrum.wave * u.angstrom)
            # same as
            # sp = Spectrum1D(flux=sim.spec(t, pose).rawspec * flam, spectral_axis=sim.spec(t, pose).wavelengths)
            bp = fcrzf(bp, sp.wavelength)
            incident = bp * sp

            stdev = detector_wavelength_center / R_ref / 2.355 / np.diff(incident.spectral_axis)[0]  #Rref is fwhm
            smoothed = gaussian_smooth(incident, stddev=stdev.si.value)
            smoothedbin = fcrzf(smoothed, wave*10*u.angstrom) # f['CUBE_EDGES'].data.edges * 10 * u.angstrom)

            def S1D2counts(s):
                counts = incident_sp.bandpass.primary_area * u.cm ** 2 * s.flux * np.diff(s.spectral_axis)[0]
                counts /= astropy.constants.h * astropy.constants.c / s.spectral_axis
                return counts.si

            def ObsFluxCheck(incident_sp):
                bp = Spectrum1D(flux=incident_sp.bandpass.throughput * u.dimensionless_unscaled,
                                spectral_axis=incident_sp.bandpass.wave * u.angstrom)
                sp = Spectrum1D(flux=incident_sp.spectrum.flux*flam, spectral_axis=incident_sp.spectrum.wave*u.angstrom)
                # sp = Spectrum1D(flux=sim.spec(t, pose).rawspec * flam, spectral_axis=sim.spec(t, pose).wavelengths)
                bp = fcrzf(bp, sp.wavelength)
                incident = bp * sp
                stdev = detector_wavelength_center / R_ref / 2.355 / np.diff(incident.spectral_axis)[0]  # Rref is fwhm
                smoothed = gaussian_smooth(incident, stddev=stdev.si.value)
                smoothedbin = fcrzf(smoothed, wave * 10 * u.angstrom)  # f['CUBE_EDGES'].data.edges * 10 * u.angstrom)
                return S1D2counts(smoothedbin)

            counts = S1D2counts(smoothedbin)

            plt.figure(figsize=(16,9))
            ax = plt.subplot(131)
            norm = np.nanmax(sp.flux)
            plt.step(sp.spectral_axis, sp.flux/norm, label=f'GOES, norm {norm:.1g}')

            norm = np.nanmax(bp.flux)
            plt.step(bp.spectral_axis, bp.flux/norm, label=f'Bandpass {norm:.1g}')

            norm=np.nanmax(incident.flux)
            plt.step(incident.spectral_axis, incident.flux/norm, label=f'Incident {norm:.1g}')
            plt.ylabel('Normalized Flux')
            plt.xlabel('$\AA$')
            plt.title('erg/s/AA/cm2')
            plt.legend()
            plt.xlim(9000,14000)

            plt.subplot(132)
            plt.step(smoothedbin.spectral_axis, counts.si.value*.7*2/np.pi, label='Conv., bin., w/ff')
            plt.step(wave*10, data[i, j, k].sum(1).sum(1), label=f'MEC Sim')
            plt.ylabel('Counts')
            plt.xlabel('$\AA$')
            plt.ylim(0, None)
            plt.xlim(9000,14000)
            plt.legend()

            plt.subplot(133)
            plt.hist(phots['wavelength']*10, incident.spectral_axis.value, histtype='step', label='Extracted (10s)')
            plt.hist(photons['wavelength']*10, incident.spectral_axis.value, histtype='step', label='Extracted Hr (10s)')
            plt.step(smoothed.spectral_axis, S1D2counts(smoothed)*.7*10*2/np.pi, label='Conv., w/ff, pi/2 norm (10s)')
            plt.ylabel('Counts')
            plt.xlabel('$\AA$')
            plt.xlim(9000,14000)
            plt.legend()
            plt.tight_layout()
# ================

            simple_obs = S.Observation(incident_sp.spectrum, incident_sp.bandpass,
                                       binset=f['CUBE_EDGES'].data.edges * 10)
            simple_obs.convert('counts')

            anorm = simple_obs.binflux.max() / incident_sp.binflux.max()
            #.7=median ff

            plt.plot(incident_sp.binwave / 10, incident_sp.binflux * anorm*0.70/1e3, drawstyle='steps-mid',
                     label='Raw (arb. norm)')
            plt.plot(simple_obs.binwave / 10, simple_obs.binflux*0.70/1e3, drawstyle='steps-mid', label='Binned Incident')
            plt.plot(wave, data[i, j, k].sum(1).sum(1)/1e3, drawstyle='steps-mid', label=f'MEC')

            plt.xlim(wave.min(), wave.max())
            if k == 0:
                plt.ylabel(f'Timestep {t}')
            if j == len(times)-1:
                plt.xlabel(f'Pose {pose} (nm)')

    fig.text(0.02, 0.5, 'Photons (thousands)', va='center', rotation='vertical')
    plt.legend()
    plt.suptitle(f'{sim.name}', y=.97)
    plt.savefig(f'{sim.name}.pdf')
    plt.close()

fig, axes = plt.subplots(len(times), len(poses), sharex=True, sharey=False, figsize=(15, 10),
                         gridspec_kw=dict(top=0.92, bottom=0.058, left=0.08, right=0.98))

for j, t in enumerate(times):
    for k, pose in enumerate(poses):
        plt.sca(axes[j,k])
        difference = np.diff(data[:, j, k].sum(2).sum(2), axis=0).squeeze()
        plt.plot(wave, 100*difference/data[0, j, k].sum(1).sum(1), drawstyle='steps-mid', label=f'MEC')

        plt.xlim(wave.min(), wave.max())
        if k == 0:
            plt.ylabel(f'Timestep {t}')
        if j == len(times)-1:
            plt.xlabel(f'Pose {k} (nm)')
        # plt.ylim(-1, 1)
# plt.ylim(-1,1)
plt.suptitle(f'Silver - Kapton / Silver', y=.96)
fig.text(0.02, 0.5, 'Percent Difference', va='center', rotation='vertical')
plt.savefig(f'difference.pdf')
plt.close()

plt.figure(figsize=(14,10))
for j, t in enumerate(times):
    for k, pose in enumerate(poses):
        difference = np.diff(data[:, j, k].sum(2).sum(2), axis=0).squeeze()
        plt.plot(wave, difference/data[0, j, k].sum(1).sum(1), drawstyle='steps-mid', label=f'{t}-{pose}')

plt.xlim(wave.min(), wave.max())
plt.ylabel(f'Percent Difference')
plt.xlabel(f'Wavelength (nm)')
plt.legend()
# plt.ylim(-1,1)
plt.suptitle(f'Silver - Kapton / Silver')
