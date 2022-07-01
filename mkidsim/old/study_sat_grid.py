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

for i, sim in enumerate((silversim, goldsim)):
    fig, axes = plt.subplots(len(times), len(poses), sharex=True, figsize=(15, 10),
                             gridspec_kw=dict(wspace=0.0, hspace=0.0, top=0.95, bottom=0.05, left=0.08, right=0.95))
    ax = None
    for j, t in enumerate(times):
        for k, pose in enumerate(poses):
            # if k+j>0:
            #     continue
            plt.sca(axes[j, k])

            f=fits.open(f'./out/{sim.name}_{t}_{pose}/{sim.name}_{t}_{pose}_scube.fits')
            data[i, j, k]= f['science'].data

            wave = np.diff(f['CUBE_EDGES'].data.edges) / 2 + f['CUBE_EDGES'].data.edges[:-1]
            sp = sim.spec(t, pose).pysynphot
            specgen, incident_sp, dc_throughput = incident_spectrum(sim.spec(t, pose).pysynphot, nd=3.5)


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
