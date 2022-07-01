import matplotlib.pyplot as plt
import numpy as np

from mkidsim.env import *

from mkidcore.corelog import getLogger
import mkidpipeline.pipeline as pipe
import mkidpipeline.config as config
from mkidpipeline.definitions import MKIDOutputCollection
from datetime import datetime
from mkidpipeline.photontable import Photontable as P
from mkidsim.env import R_ref, R_wave

from specutils.manipulation import gaussian_smooth
import mkidsim.env
KEEP_KEYS = ("CROP_EN1,CROP_EN2,CROP_OR1,CROP_OR2,CROPPED ,DETECTOR,DETGAIN ,EQUINOX ,EXPTIME ,GAIN    ,HST-END ,"
            "HST-STR ,INSTRUME,E_BMAP  ,E_CFGHSH,E_DARK  ,E_FLTCAL,E_H5FILE,E_PLTSCL,E_PREFX ,E_PREFY ,E_SPECAL,"
            "E_WAVCAL,E_WCSCAL,OBJECT  ,TELESCOP,TELFOCUS,UNIXEND ,UNIXSTR ,UT-END  ,UT-STR  ,CRAYCAL ,PIXCAL  ,"
            "LINCAL  ,SPECCAL ,WAVECAL ,FLATCAL ,H5MINWAV,H5MAXWAV,MINWAVE ,MAXWAVE ,ERESOL  ,DEADTIME,UNIT    ,"
            "EXFLAG")
KEEP_KEYS = tuple([k.strip() for k in KEEP_KEYS.split(',')])


logging.basicConfig()
log = getLogger('photonsim')
getLogger('mkidcore', setup=True, logfile=f'mkidpipe_{datetime.now().strftime("%Y-%m-%d_%H%M")}.log')
getLogger('mkidpipe').setLevel('DEBUG')
getLogger('mkidpipeline').setLevel('DEBUG')
getLogger('mkidcore').setLevel('DEBUG')
getLogger('mkidcore.config').setLevel('INFO')

exp_time = 10
psf_radius = 0.01828 * u.arcsec

from goeslib import SatLib

lib = SatLib('./data/goeslib')
silversim = lib['MKID_20201013_Silver']
goldsim = lib['MKID_20201013_Kapton']

config.configure_pipeline('pipe.yaml')
outputs = MKIDOutputCollection('simout.yaml', datafile='simdata.yaml')
dataset = outputs.dataset

posetimes = ((0, 6), (32, 4))
magnitudes = np.array([14, 18, 22])
R0 = np.array([4, 16, 64])

mkidsim.env.R_wave = detector_wavelength_range[1]

# Gather the data
hcubes = []
waves = []
for r in R0:
    mkidsim.env.R_ref = r
    headers = []
    data = []
    for sim in (silversim, goldsim):
        d1 = []
        for t, pose in posetimes:
            d2 = []
            for m in magnitudes:
                spec = sim.spec(t, pose)
                sp = spec.pysynphot
                nd = (m - spec.mag_j) / 2.5
                name = f'{sim.name}_{t}_{pose}_{nd:.3f}_{m}_{r}'
                hf = f'./grid/{name}.h5'
                f = fits.open(f'./grid/{name}/{name}_out_scube.fits')
                headers.append(f['SCIENCE'].header)
                whdu=f['CUBE_EDGES']
                w = f['CUBE_EDGES'].data['edges']
                d2.append(f['SCIENCE'].data)
            d1.append(d2)
        data.append(d1)
    waves.append(w)
    hcube=np.array(data)
    header=headers[0]
    for k in (k for k in tuple(header.keys()) if k not in KEEP_KEYS):
        header.pop(k)
    hcf=fits.HDUList([fits.PrimaryHDU(header=header), fits.ImageHDU(hcube, header), whdu])
    hcf.writeto(f'./grid/MEC_GOESR_R{r}.fits', overwrite=True)
    hcubes.append(hcube)

#Example image
plt.close('all')
fig, axes = plt.subplots(1, 3, figsize=(10, 3))

plt.sca(axes[0])
x=hcubes[1][0,0,1,1]
plt.imshow(x, interpolation='nearest', origin='lower', norm=mplc.LogNorm(vmin=.01, vmax=x.max()))
plt.ylabel('Pixel')
plt.xlabel('Pixel')
m=magnitudes[1]
r=R0[1]
w=waves[1][1]
plt.title(f'$\lambda$={w:.0f} nm')
plt.colorbar(shrink=0.7, aspect=20*0.7)

plt.sca(axes[2])
x=hcubes[1][0,0,1,-3]
plt.imshow(x, interpolation='nearest', origin='lower', norm=mplc.LogNorm(vmin=.01, vmax=x.max()))
plt.xlabel('Pixel')
m=magnitudes[1]
r=R0[1]
w=waves[1][-3]
plt.title(f'$\lambda$={w:.0f} nm')
plt.colorbar(shrink=0.7, aspect=20*0.7).set_label('$\gamma s^{-1}$')

plt.sca(axes[1])
x=hcubes[1][0,0,1].mean(0)
plt.imshow(x, interpolation='nearest', origin='lower', norm=mplc.LogNorm(vmin=.01, vmax=x.max()))
plt.xlabel('Pixel')
m=magnitudes[1]
r=R0[1]
w=waves[1].mean()
plt.title(f'Mean')
plt.colorbar(shrink=0.7, aspect=20*0.7)
plt.suptitle(f'Silver MLI J={m} R={r} @ {detector_wavelength_range[1].value:.0f} nm')
plt.subplots_adjust(top=0.817,bottom=0.15,left=0.059,right=0.958,hspace=0.22,wspace=0.186)
plt.savefig('./grid/example_image.pdf')


# Create a S/N estimate cube  (NB hcubes[R][material, orientation, magnitude, wavelength, nx, ny])
plt.close('all')
fig, axes = plt.subplots(1, 3, figsize=(10, 4))
for r, hc, w, ax in zip(R0, hcubes, waves, axes):
    hc = hc.sum(-1).sum(-1) # / np.diff(w)  normalize by wavelength bin size?
    ddsim = np.diff(hc, axis=0).squeeze()
    ddpos = np.diff(hc, axis=1).squeeze()
    ddsimspos = np.diff(ddsim, axis=0).squeeze()
    sn = np.abs(ddsimspos) / np.sqrt(hc[0].mean(0))
    l = np.diff(w) / 2 + w[:-1]
    plt.sca(ax)
    plt.step(l, sn.T, label=list(map(str, magnitudes)))
    plt.title(f'R={r} @ {detector_wavelength_range[1].value:.0f} nm')
    plt.xlabel('nm')
    plt.xlim(detector_wavelength_range[0].value, detector_wavelength_range[1].value)
plt.legend()
plt.sca(axes[0])
plt.ylabel('S/N ($d\gamma/d_{spec}/d_{pose}/\sqrt{s}$)')
plt.tight_layout()
plt.savefig('./grid/sn_estimate, unnorm.pdf')


# Create an example image
ls = ['solid', 'dashed','dashdot', 'dotted']
j = 0
k = 0
median_ff = .7
for i, (r, hc, ax) in enumerate(zip(R0, hcubes, axes)):
    plt.close('all')
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    plt.sca(ax)
    mkidsim.env.R_ref = r
    dw = np.diff(waves[i])
    w = dw / 2 + waves[i][:-1]

    sim=(silversim, goldsim)[j]
    t, pose = posetimes[k]
    m=magnitudes[i]

    spec = sim.spec(t, pose)

    nd = (m - spec.mag_j) / 2.5
    phot_per_nm = hc[j, k, l].sum(-1).sum(-1) / dw  #photons/nm/s, includes ND, splitter, filter, median FF

    plt.step(w, phot_per_nm, color=f'C0', label='Captured')

    # The incident spectrum, resampled
    _, incident_sp, dc_throughput = incident_spectrum(spec.pysynphot, nd=nd)
    # Recall
    #   dc_throughput = splitter * 10 ** -nd * filter
    #   incident bandpass = bp * telescope * atmosphere * dc_throughput
    bp = Spectrum1D(flux=incident_sp.bandpass.throughput * u.dimensionless_unscaled,
                    spectral_axis=incident_sp.bandpass.wave * u.angstrom)
    sp = Spectrum1D(flux=incident_sp.spectrum.flux * flam,
                    spectral_axis=incident_sp.spectrum.wave * u.angstrom)

    bp = fcrzf(bp, sp.wavelength)
    incident = bp * sp * median_ff  # NB 10s A->nm
    dw2 = np.diff(incident.spectral_axis)[0]/10

    plt.step(incident.spectral_axis / 10, Spectrum1D_to_counts(incident, incident_sp.bandpass.primary_area) /dw2,
             label='Incident', color=f'C0', linestyle='dotted', linewidth=.7)
    plt.xlim(detector_wavelength_range[0].value, detector_wavelength_range[1].value)
    plt.xlabel('nm')
    plt.ylabel('Flux Density ($\gamma$/s)')
    plt.title(f'R={r} @ {detector_wavelength_range[1].value:.0f} nm')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./grid/specexample_J={m}.pdf')


plt.close('all')
fig, axes = plt.subplots(3, 1, figsize=(10, 8))
ls = ['solid', 'dashed','dashdot', 'dotted']
for i, (r, hc, ax) in enumerate(zip(R0, hcubes, axes)):
    mkidsim.env.R_ref = r
    plt.sca(ax)
    dw = np.diff(waves[i])
    w = dw / 2 + waves[i][:-1]
    for j, sim in enumerate((silversim, goldsim)):
        for k, (t, pose) in enumerate(posetimes):
            spec = sim.spec(t, pose)

            # for l, m in enumerate(magnitudes):
            l=2
            m=magnitudes[l]

            median_ff = .7
            nd = (m - spec.mag_j) / 2.5

            phot_per_nm = hc[j, k, l].sum(-1).sum(-1) / dw
            lab = f'S{j} P{k}' if l == 0 else None
            plt.plot(w, phot_per_nm, label=lab, color=f'C{l}', linestyle=ls[j*2+k])

            #The incident spectrum, resampled
            specgen, incident_sp, dc_throughput = incident_spectrum(spec.pysynphot, nd=nd)
            bp = Spectrum1D(flux=incident_sp.bandpass.throughput * u.dimensionless_unscaled,
                            spectral_axis=incident_sp.bandpass.wave * u.angstrom)
            sp = Spectrum1D(flux=incident_sp.spectrum.flux * flam,
                            spectral_axis=incident_sp.spectrum.wave * u.angstrom)

            bp = fcrzf(bp, sp.wavelength)
            incident = bp * sp * median_ff
            # NB 10s A->nm

            dw2 = np.diff(incident.spectral_axis / 10)[0]

            plt.step(incident.spectral_axis / 10,
                     Spectrum1D_to_counts(incident, incident_sp.bandpass.primary_area) /dw2,
                     color=f'C{l}', linestyle=ls[j*2+k], linewidth=.7)
            plt.xlim(detector_wavelength_range[0].value, detector_wavelength_range[1].value)
            plt.xlabel('nm')

            # # Convolution is in array samples
            # stdev = detector_wavelength_center / R_at(detector_wavelength_center)  / dw2 / 2.35482 / 10
            # smoothed2 = gaussian_smooth(incident, stddev=stdev.si.value)
            # plt.step(incident.spectral_axis / 10,
            #          Spectrum1D_to_counts(smoothed2, incident_sp.bandpass.primary_area)/dw2,
            #          label='Simple', color=f'C{l}', linestyle=ls[j*2+k], linewidth=1.2)

plt.legend()



pt = P(hf)
phots = pt.query(startw=detector_wavelength_range[0].value, stopw=detector_wavelength_range[1].value)

for i, sim in enumerate((silversim, goldsim)):
    fig, axes = plt.subplots(len(times), len(poses), sharex=True, figsize=(15, 10),
                             gridspec_kw=dict(wspace=0.0, hspace=0.0, top=0.95, bottom=0.05, left=0.08, right=0.95))
    ax = None
    for j, t in enumerate(times):
        for k, pose in enumerate(poses):
            # if k+j>0:
            #     continue
            plt.sca(axes[j, k])
            fitsfile = f'{cfg.paths.out}{sim.name}_{t}_{pose}/{sim.name}_{t}_{pose}_scube.fits'
            h5file = f'{cfg.paths.out}{sim.name}_{t}_{pose}.h5'
            pt = P(h5file)
            f = fits.open(fitsfile)
            data[i, j, k] = f['science'].data
            sp = sim.spec(t, pose).pysynphot

            wave = np.diff(f['CUBE_EDGES'].data.edges) / 2 + f['CUBE_EDGES'].data.edges[:-1]
            specgen, incident_sp, dc_throughput = incident_spectrum(sim.spec(t, pose).pysynphot, nd=3.5)
            # ================

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
            stdev = detector_wavelength_center / R_at(detector_wavelength_range[1]) / 2.35482 / \
                    np.diff(incident.spectral_axis)[0]  # Rref is fwhm
            smoothed2 = gaussian_smooth(incident * .7, stddev=stdev.si.value)
            plt.step(incident.spectral_axis / 10,
                     Spectrum1D_to_counts(incident, incident_sp.bandpass.primary_area) * 10 * .7,
                     label=f'Incident')
            plt.hist(phots['wavelength'], incident.spectral_axis.value / 10, histtype='step', label='MEC')
            plt.step(smoothed2.spectral_axis / 10,
                     Spectrum1D_to_counts(smoothed2, incident_sp.bandpass.primary_area) * 10,
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
            if j == len(times) - 1:
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
        plt.sca(axes[j, k])
        difference = np.diff(data[:, j, k].sum(2).sum(2), axis=0).squeeze()
        plt.plot(wave, 100 * difference / data[0, j, k].sum(1).sum(1), drawstyle='steps-mid', label=f'MEC')

        plt.xlim(wave.min(), wave.max())
        if k == 0:
            plt.ylabel(f'Timestep {t}')
        if j == len(times) - 1:
            plt.xlabel(f'Pose {pose} (nm)')
        # plt.ylim(-1, 1)
# plt.ylim(-1,1)
plt.suptitle(f'Silver - Kapton / Silver', y=.96)
fig.text(0.02, 0.5, 'Percent Difference', va='center', rotation='vertical')
plt.savefig(f'difference2.pdf')
plt.close()
