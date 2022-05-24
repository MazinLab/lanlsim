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
sp2 = SatModel(exclude=('mli_black',), swap=(('mli_silver', 'mli_black')))


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



image, gen, observation, dc_throughput, flat_field, photons, total_launched = simulate_observation(sp, psf_radius, exp_time, nd=3.5)


# Dummy broaden to get in the ballpark as a test
c, w = np.histogram(gen(int(observation.countrate())) / 10, bins=observation.binset / 10)
import scipy.ndimage as ndi
broad = ndi.gaussian_filter(c, 1000/10 / np.diff(w)[0], mode='nearest')

plt.subplot(121)
plt.plot(w[:-1], c*flat_field.mean(),  label='Generated')
plt.plot(w[:-1], broad*flat_field.mean(), label='Generated (est. convol)')
plt.plot(observation.binwave / 10, observation.binflux*flat_field.mean(), drawstyle='steps-mid', label='binned synphot')
c,w = np.histogram(photons.wavelength, bins=observation.binset/10)
plt.plot(w[:-1], c/exp_time, label='photon rate')
plt.legend()
print(f'Detected {photons.size} of {total_launched} ({photons.size/total_launched*100:.1f}%)')
plt.title('Satellite')
cfg = generate_default_config(instrument='MEC')
cfg.update('paths.out', './')
buildfromarray(photons, config=cfg, user_h5file='./satellite.h5')


image, gen, observation, dc_throughput, flat_field, photons, observed= simulate_observation(sp2, psf_radius, exp_time, nd=3.5)
plt.subplot(122)
c, w = np.histogram(gen(int(observation.countrate())) / 10, bins=observation.binset / 10)
import scipy.ndimage as ndi
broad = ndi.gaussian_filter(c, 1000/10 / np.diff(w)[0], mode='nearest')
plt.plot(w[:-1], c*flat_field.mean(),  label='Generated')
plt.plot(w[:-1], broad*flat_field.mean(), label='Generated (est. convol)')
plt.plot(observation.binwave / 10, observation.binflux*flat_field.mean(), drawstyle='steps-mid', label='binned synphot')
c,w = np.histogram(photons.wavelength, bins=observation.binset/10)
plt.plot(w[:-1], c/exp_time, label='photon rate')
plt.legend()
plt.title('Satellite (mli_sliver)')
print(f'Detected {photons.size} of {total_launched} ({photons.size/total_launched*100:.1f}%)')
buildfromarray(photons[:observed], config=cfg, user_h5file='./satellite_noblack.h5')
