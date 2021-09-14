import numpy as np

from mkidsim.env import *
from mkidcore.binfile.mkidbin import PhotonNumpyType

exp_time = 10
psf_radius = .3 * u.arcsec / 2
desired_avg_countrate = 5000

# plt.plot(_spectrum.wave, _spectrum.flux/_spectrum.integrate(), label='f/int')
# plt.plot(_spectrum.wave, _spectrum(_spectrum.wave*1e4)/_spectrum.integrate(), label='f(w)/int')
# plt.plot(_spectrum.wave, _spectrum.flux/_spectrum.flux.sum(), label='f/f.sum')


# x = pipe.wavecal.fetch(o.wavecals).values()[0]
# y, _ = x.find_resolving_powers()
# w = np.array(x.cfg.wavelengths)
# r=np.nanmedian(y, 0)
# re=np.nanstd(y, 0)
#
# rfunc = lambda x,w: x[0]/w**x[1] + x[2]
# err = lambda x: (rfunc(x,w) - r)/re
# import scipy.optimize as opt
# z= opt.least_squares(err, (5*w[0], 1, 0), bounds=((0, 0, -np.inf), (50000, 1.0, np.inf)))
# plt.plot(np.linspace(800,1400,1000), rfunc(z.x,np.linspace(800,1400,1000)))
# plt.plot(w, r,'o')
# err = lambda x: (rfunc(x,w) - r)/re


field_extent = np.array([(-detector_shape / 2),
                         (+detector_shape / 2)]) * detector_sampling
square_fov = np.diff(field_extent, axis=0).ravel().max()
fov = np.diff(field_extent, axis=0).ravel()

# Normalize the spectrum to desired flux
flux = 6 * u.ABmag
spectrum = skysim.PYSPSed(family='k93models', sedpars=(6440, 0, 4.3))

# Multiply spectrum by earths atmosphere and telescope transmission
atmosphere = AtmosphericTransmission()
telescope = SubaruTransmission()

bp = S.Box(1.150, .400, 'um')
sp = S.Icat('k93models', 6440, 0, 4.3)
atmosphere.convert('um')
telescope.convert('um')
sp.convert('um')
x = sp * telescope * atmosphere
x = x.renorm(desired_avg_countrate / (R_wave / R_ref).to('AA').value, 'photlam', bp)
# x.primary_area = ((8*u.m)**2).to('cm^2').value
# x.integrate()
x = x.tabulate().resample(np.arange(7000, 15000, 1))
spectrum._PYSPsed = x  # x.tabulate().taper().resample(np.arange(7000,15000, 1))
spectrum.goodwrange = [x.wave[0], x.wave[-1]]
spectrum._PYSPsed.integrate()

# Create a Scene
# source_pos = SkyCoord(0*u.deg, 0*u.deg)

scene = skysim.Point()
scene.set_SED(spectrum)

# Convolve scene with PSF
grip, sampled_psf = get_mec_psf(fov, detector_sampling, psf_radius)

# scene.convolve_with(sampled_psf)

# Integrate scene onto the detector

photon_rate_image = spectrum._PYSPsed.integrate() * sampled_psf  # photons/s/cm^2* cm^2 ??
# photon_rate_cube = scene.buildcube(wavelengths=detector_wavelength_range, fov=fov, units='photon/s',
#                                    spatial_sampling=detector_sampling)


# Compute a total number of detected photons per pixel
pixel_count_image = np.random.poisson(photon_rate_image * exp_time)

specgen = SpecgenInverse(spectrum)

# v, b = np.histogram(x(1000000), bins=x._wave)
# plt.plot(b[:-1], v / v.sum())



from mkidpipeline.pipeline import generate_default_config
from mkidpipeline.steps.buildhdf import buildfromarray

cfg = generate_default_config(instrument='MEC')
cfg.update('paths.out', './')
photons = np.recarray(pixel_count_image.sum(), dtype=PhotonNumpyType)
photons[:] = 0
observed = 0




# Compute photon arrival times and wavelengths for each photon
for pixel, n in np.ndenumerate(pixel_count_image):
    if not n:
        continue

    # Generate arrival times for the photons
    arrival_times = np.random.uniform(0, exp_time, size=n)
    arrival_times.sort()

    # Generate wavelengths for the photons
    # ultimately the spectrum may vary with field position, but for now we'll use the global
    # specgen = SpecgenInverse(scene.sed_at_pixel(pixel))
    wavelengths = specgen(n) * u.micron
    energies = 1/wavelengths
    # merge photon energies within 1us
    to_merge = (np.diff(arrival_times) < 1e-6).nonzero()[0]
    if to_merge.size:
        cluster_starts = to_merge[np.concatenate(([0], (np.diff(to_merge) > 1).nonzero()[0] + 1))]
        cluser_last = to_merge[(np.diff(to_merge) > 1).nonzero()[0]]+1
        cluser_last = np.append(cluser_last, to_merge[-1]+1)    # inclusive
        for start, stop in zip(cluster_starts, cluser_last):
            merge = slice(start+1, stop+1)

            energies[start] += energies[merge].sum()
            energies[merge] = np.nan

            # e = (1 / wavelengths[merge]).sum()+1/wavelengths[start]
            # wavelengths[start] = (1 / e).to('um')
            # wavelengths[merge] = np.nan * u.nm

    # Determine measured energies
    energy_width = energies / R(energies, pixel=pixel, inverse=True)  # The relevant R value for each photon pixel combo
    measured_energies = np.random.normal(loc=energies, scale=energy_width)

    # Filter those that wouldn't trigger
    will_trigger = measured_energies/u.um > MIN_TRIGGER_ENERGY
    if not will_trigger.any():
        continue
    # Drop photons that arrive within the deadtime
    detected = mask_deadtime(arrival_times[will_trigger], deadtime.to(u.s).value)

    arrival_times = arrival_times[will_trigger][detected]
    measured_wavelengths = 1000/measured_energies[will_trigger][detected]
    measured_wavelengths.clip(SATURATION_WAVELENGTH_NM)


    # Add photons to the pot
    sl = slice(observed, observed + arrival_times.size)
    photons.wavelength[sl] = measured_wavelengths
    photons.time[sl] = (arrival_times*1e6)  #in microseconds
    photons.resID[sl] = cfg.beammap.residmap[pixel[::-1]]
    observed += arrival_times.size

#plt.
# plt.imshow(pixel_count_image)
# plt.colorbar().set_label('photons')
# plt.xlabel('Pixel')
# plt.ylabel('Pixel')
# plt.title('F7 (6th mag) MEC focal plane input')
#
# plt.title('F7 (6th mag) sampled spectrum')
# plt.hist(photons.wavelength*1000, bins=np.linspace(950, 1300, 1000), histtype='step', density=True, label='data')
# plt.hist(photons.wavelength*1000, bins=np.linspace(950, 1300, 5), histtype='step', density=True, label='MEC')
# plt.ylabel('Photon Flux Density')
# plt.xlabel('Wavelength (nm)')


print(f'Detected {observed} of {photons.size} ({observed/photons.size*100:.1f}%)')
# Store the photons into an h5 file
buildfromarray(photons[:observed], config=cfg)
