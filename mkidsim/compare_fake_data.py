from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

hdul_sim = fits.open('out/simulated/out2_scube.fits')
hdul_sci = fits.open('out/science/out0_scube.fits')


mask = hdul_sci['SCIENCE'].data==0
hdul_sim['SCIENCE'].data[mask]=0



plt.subplot(212)
hdul, lbl=hdul_sci, 'HIP 109427'
plt.plot(hdul['CUBE_EDGES'].data[:-1].edges, hdul['SCIENCE'].data.sum(1).sum(1)/10, label=lbl)
hdul, lbl=hdul_sim, 'HIP 109427 (Simulated)'
plt.plot(hdul['CUBE_EDGES'].data[:-1].edges, hdul['SCIENCE'].data.sum(1).sum(1)/10, label=lbl)
hdul, lbl=hdul_sim, 'HIP 109427 (Sim., 40% Strehl Corr.)'
plt.plot(hdul['CUBE_EDGES'].data[:-1].edges, hdul['SCIENCE'].data.sum(1).sum(1)/10*.4, label=lbl)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Count Rate (photons/s)')
plt.legend()

plt.subplot(221)
hdul, lbl=hdul_sci, 'HIP 109427 (coronagraph in)'
plt.imshow(hdul['SCIENCE'].data.sum(0), origin='lower')
plt.colorbar().set_label('Photons')
plt.xlabel('Pixel')
plt.ylabel('Pixel')
plt.title(lbl)

plt.subplot(222)
hdul, lbl=hdul_sim, 'HIP 109427 (Simulated)'
plt.imshow(hdul['SCIENCE'].data.sum(0), origin='lower')
plt.colorbar().set_label('Photons')
plt.xlabel('Pixel')
plt.ylabel('Pixel')
plt.title(lbl)
