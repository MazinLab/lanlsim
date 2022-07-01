from mkidsim.env import *

from mkidcore.corelog import getLogger
import mkidpipeline.pipeline as pipe
import mkidpipeline.config as config
from mkidpipeline.definitions import MKIDOutputCollection
from datetime import datetime
from mkidpipeline.photontable import Photontable as P
from mkidsim.env import R_ref, R_wave

from specutils.manipulation import gaussian_smooth

logging.basicConfig()
log = getLogger('photonsim')
getLogger('mkidcore', setup=True, logfile=f'mkidpipe_{datetime.now().strftime("%Y-%m-%d_%H%M")}.log')
getLogger('mkidpipe').setLevel('DEBUG')
getLogger('mkidpipeline').setLevel('DEBUG')
getLogger('mkidcore').setLevel('DEBUG')
getLogger('mkidcore.config').setLevel('INFO')

exp_time = 10
psf_radius = .3 * u.arcsec / 2

times = range(0,71,8)
poses = (0,4,5,8)


cfg = pipe.generate_default_config(instrument='MEC')
cfg.update('paths.out', './out2/')
cfg.update('paths.tmp', './scratch/')
cfg.update('paths.database', './db/')
config.configure_pipeline('pipe.yaml')
outputs = MKIDOutputCollection('simout.yaml', datafile='simdata.yaml')
dataset = outputs.dataset

data = np.zeros((2, len(times), len(poses), 5, 140, 146))  #140,146 is detector size

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
            fitsfile=f'{cfg.paths.out}{sim.name}_{t}_{pose}/{sim.name}_{t}_{pose}_scube.fits'
            h5file=f'{cfg.paths.out}{sim.name}_{t}_{pose}.h5'
            pt = P(h5file)
            f = fits.open(fitsfile)
            data[i, j, k] = f['science'].data
            sp = sim.spec(t, pose).pysynphot

            wave = np.diff(f['CUBE_EDGES'].data.edges) / 2 + f['CUBE_EDGES'].data.edges[:-1]
            specgen, incident_sp, dc_throughput = incident_spectrum(sim.spec(t, pose).pysynphot, nd=3.5)
#================

            phots = pt.query(startw=detector_wavelength_range[0].value, stopw=detector_wavelength_range[1].value)
            bp = Spectrum1D(flux=incident_sp.bandpass.throughput * u.dimensionless_unscaled,
                            spectral_axis=incident_sp.bandpass.wave * u.angstrom)
            sp = Spectrum1D(flux=incident_sp.spectrum.flux * flam, spectral_axis=incident_sp.spectrum.wave * u.angstrom)
            # is the same as  Spectrum1D(flux=sim.spec(t, pose).rawspec * flam,
            #   spectral_axis=sim.spec(t, pose).wavelengths)
            bp = fcrzf(bp, sp.wavelength)
            incident = bp * sp

# ================
            # norm=np.nanmax(incident.flux)
            stdev = detector_wavelength_center / R_at(detector_wavelength_range[1]) / 2.35482 / np.diff(incident.spectral_axis)[0]  #Rref is fwhm
            smoothed2 = gaussian_smooth(incident*.7, stddev=stdev.si.value)
            plt.step(incident.spectral_axis/10, Spectrum1D_to_counts(incident, incident_sp.bandpass.primary_area)*10*.7,
                     label=f'Incident')
            plt.hist(phots['wavelength'], incident.spectral_axis.value/10, histtype='step', label='MEC')
            plt.step(smoothed2.spectral_axis/10, Spectrum1D_to_counts(smoothed2, incident_sp.bandpass.primary_area)*10,
                     label='Simple Sim')
            # plt.ylabel('Photon Flux')
            # plt.xlabel('$\AA$')
            # plt.legend()
            # plt.xlim(detector_wavelength_range[0].value,detector_wavelength_range[1].value)

            simple_obs = S.Observation(incident_sp.spectrum, incident_sp.bandpass,
                                       binset=f['CUBE_EDGES'].data.edges * 10)
            # simple_obs.convert('counts')
            #
            # anorm = simple_obs.binflux.max() / incident_sp.binflux.max()
            # #.7=median ff
            #
            # plt.plot(incident_sp.binwave / 10, incident_sp.binflux * anorm*0.70/1e3, drawstyle='steps-mid',
            #          label='Raw (arb. norm)')
            # plt.plot(simple_obs.binwave / 10, simple_obs.binflux*0.70/1e3, drawstyle='steps-mid', label='Binned Incident')
            # plt.plot(wave, data[i, j, k].sum(1).sum(1)/1e3, drawstyle='steps-mid', label=f'MEC')

            plt.xlim(wave.min(), wave.max())
            if k == 0:
                plt.ylabel(f'Timestep {t}')
            if j == len(times)-1:
                plt.xlabel(f'Pose {pose} (nm)')

    fig.text(0.02, 0.5, 'Photon Flux', va='center', rotation='vertical')
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
            plt.xlabel(f'Pose {pose} (nm)')
        # plt.ylim(-1, 1)
# plt.ylim(-1,1)
plt.suptitle(f'Silver - Kapton / Silver', y=.96)
fig.text(0.02, 0.5, 'Percent Difference', va='center', rotation='vertical')
plt.savefig(f'difference2.pdf')
plt.close()
