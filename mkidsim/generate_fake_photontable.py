import numpy as np
from logging import getLogger
log = getLogger('photonsim')
from mkidsim.env import *
from mkidpipeline.steps.buildhdf import buildfromarray
from mkidpipeline.pipeline import generate_default_config

exp_time = 10
psf_radius = .3 * u.arcsec / 2
desired_avg_countrate = None  #5000
desired_magnitude = 3.52


#HIP 109427 A
# sp = PhoenixModel(9040, -0.38, 4.27, desired_magnitude=desired_magnitude if not desired_avg_countrate else None)

sp = SatModel(materials='all')
sp2 = SatModel(exclude=('mli_black',))


def make_plot(image, photons, title):
    plt.clf()
    plt.subplot(121)
    plt.imshow(image)
    plt.colorbar().set_label('photons')
    plt.xlabel('Pixel')
    plt.ylabel('Pixel')
    plt.title(f'{title} MEC focal plane input')

    plt.subplot(122)
    plt.title(f'{title} sampled spectrum')
    plt.hist(photons.wavelength, bins=np.linspace(950, 1300, 1000), histtype='step', density=True, label='data')
    plt.hist(photons.wavelength, bins=np.linspace(950, 1300, 5), histtype='step', density=True, label='MEC')
    plt.ylabel('Photon Flux Density')
    plt.xlabel('Wavelength (nm)')
    plt.tight_layout()#top=0.905, bottom=0.152, left=0.028, right=0.984, hspace=0.481, wspace=0.259)



image, gen, observatio, photons, observed = simulate_observation(sp, psf_radius, exp_time, nd=3.5)
plt.figure()
make_plot(image, photons, 'Satellite')
cfg = generate_default_config(instrument='MEC')
cfg.update('paths.out', './')
print(f'Detected {observed} of {photons.size} ({observed/photons.size*100:.1f}%)')
buildfromarray(photons[:observed], config=cfg, user_h5file='./satellite.h5')

image, gen, photons, observed = simulate_observation(sp2, psf_radius, exp_time, nd=3.5)
print(f'Detected {observed} of {photons.size} ({observed/photons.size*100:.1f}%)')
plt.figure()
make_plot(image, photons, 'Satellite (No Black)')
buildfromarray(photons[:observed], config=cfg, user_h5file='./satellite_noblack.h5')
