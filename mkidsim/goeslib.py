from glob import glob
import os.path
import numpy as np
import astropy.units as u
import astropy.io
from logging import getLogger

# goes solid angle np.pi*(3/35.786e6)**2


class SatSpec:
    def __init__(self, file, materials='all', solid_angle=(5e-6/5000)**2, exclude=tuple(), swap=tuple(), mag_j=None):
        """
        GOES-R is ~29 m^2 or about a 6m circle's worth of area
        3 m radius satillite at GEO ~36e6 m ~1/36e6 rad-> pi*(3/35.786e6/2)**2 ~6.06e-16 sr
        """
        self.mag_j = mag_j
        with open(file) as f:
            header = f.readline()
        material_labels = header.strip(',\n').split(',')
        radiance_vector = np.loadtxt(file, delimiter=',', skiprows=1)
        self.wavelengths = radiance_vector[:, 0]*1e4*u.angstrom  # um to AA
        #radiance = radiance_vector[:, 1:]*solid_angle*1e3  #W/cm^2/sr/um -> erg/s/cm^2/AA
        radiance = radiance_vector[:, 1:] * solid_angle * 1e3 * np.sqrt(2)  #rt2 is arbitrary
        # W/cm^2/sr/um == J/s/cm^2/um == 10^7 erg/s/cm^2/(AA * 10^4) = 10^3 erg/s/cm^2/AA
        #um = 10^4 AA
        #J = 10^7 erg
        #flam =erg/s/cm^2/AA
        self.ref_solid_angle = solid_angle
        if materials is 'all':
            materials = material_labels

        self.use = [material_labels.index(m) for m in materials if m not in exclude]

        if swap:
            renorm = 1+radiance[:, material_labels.index(swap[1])].mean()/radiance[:, material_labels.index(swap[0])].mean()
            getLogger(__name__).info(f'Swapping {swap[0]} for {swap[1]} by renormalizing to {renorm}')
            radiance[:, material_labels.index(swap[0])] *= renorm

        self.file=file
        fn = os.path.basename(file)
        self.time = int(fn[4:8])
        self.pose = int(fn[13:17])
        self.radiance = radiance

    @property
    def rawspec(self):
        return self.radiance[:, self.use].sum(1)

    @property
    def pysynphot(self):
        from pysynphot import ArraySpectrum
        sp = ArraySpectrum(self.wavelengths.value, self.rawspec, fluxunits='flam')
        return sp


class SatSim:
    def __init__(self, d):
        self.mags_j = np.loadtxt(os.path.join(d, '20201013_MR01_NewModel_j_CalMags.txt'))  # mag_j[pose, time]
        self.data = astropy.io.ascii.read(d+'/AccessData.csv')
        lightcurves = glob(os.path.join(d, 'Lightcurves','*'))
        self.times = sorted(set([int(os.path.basename(lc)[4:8]) for lc in lightcurves]))
        self.poses = sorted(set([int(os.path.basename(lc)[13:17]) for lc in lightcurves]))
        self._lightcurve_file = os.path.join(d, 'Lightcurves', 'Time{time:04}_Pose{pose:04}.lc')
        self.name = os.path.basename(d)

    def spec(self, time, pose):
        return SatSpec(self._lightcurve_file.format(time=time, pose=pose), mag_j=self.mags_j[pose, time])

    def cube(self):
        wave = self.spec(0, 0).wavelengths
        ret = np.zeros((len(self.times), len(self.poses), len(wave)))
        for i in range(len(self.times)):
            for j in range(len(self.poses)):
                ret[i, j, :] = self.spec(i, j).rawspec
        return wave, ret

    # def plot(self, ax=None, dc_throughput=1):
    #     if ax is not None:
    #         plt.sca(ax)
    #     plt.plot(a.binwave / 10, a.binflux*dc_throughput, drawstyle='steps-mid', label='binned')
    #     plt.hist(specgen(int(a2.countrate())) / 10, bins=binset / 10, histtype=u'step', label='Generated')
    #     plt.xlabel('nm')
    #     plt.ylabel('Photons')
    #     plt.legend()
    #
    #     plt.figure()
    #     plt.plot(a2.wave, a2.flux, label='native')
    #     a2.convert('counts')
    #     plt.plot(a2.binwave/10, a2.binflux*dc_throughput, drawstyle='steps-mid', label='binned')
    #     plt.xlim(5030, 5050)
    #     plt.xlabel(a2.waveunits)
    #     plt.ylabel(a2.fluxunits)
    #     a.integrate()
    #     a.countrate()


class SatLib:
    def __init__(self, dir):
        datasets = [d for d in glob(os.path.join(dir, '*')) if os.path.isdir(d)]
        self.datasets = {}
        for d in datasets:
            self.datasets[os.path.basename(d)] = SatSim(d)

    def __getitem__(self, item):
        return self.datasets[item]
