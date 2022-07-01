
#=======
import os
import pysynphot as S
import astropy.units as u
from mkidpipeline.photontable import Photontable as PT
detector_wavelength_center = 1162.5 * u.nm
detector_wavelength_width = 425 * u.nm
nd=2.5
filename = os.path.join(os.environ['PYSYN_CDBS'], 'grid', 'pickles', 'dat_uvk', 'pickles_uk_27.fits')  #G5V
sp = S.FileSpectrum(filename)
sp_norm = sp.renorm(6.23, 'vegamag', S.ObsBandpass('johnson,j'))
filterbp=S.Box(detector_wavelength_center.to(u.Angstrom).value, detector_wavelength_width.to(u.Angstrom).value)
filterbp.primary_area=530000  #cm^2
atmosphere = AtmosphericTransmission()
telescope = SubaruTransmission()
telescope.convert('angstrom')
atmosphere.convert('angstrom')
bandp=filterbp*telescope*atmosphere
phot_per_nm_raw = S.Observation(sp_norm, bandp).countrate()/425
flatfield_factor=.7
p = PT('./data/1642681900.h5')
phot=p.get_fits(startw=950, stopw=1375, rate=False, weight=True)
x=phot['SCIENCE'].data
xm=np.ma.MaskedArray(x, x==0)
sim=xm[62-20:62+20, 103-20:103+20]
photrate_obs_nm = sim.size*sim.mean()/p.duration/detector_wavelength_width.value # this includes 2.5ND splitter and filter (latter two estimated at 9% total tput)
print(f'Crude throughput {photrate_obs_nm*10**nd/phot_per_nm_raw*100:.1f}%')  # finding about 11.8% slightly better than 9% used in sims
print(f'Expected raw rate for {6.23}Mj {phot_per_nm_raw*flatfield_factor/1e6:.2f} Mphot/nm/s')

from goeslib import SatLib

lib = SatLib('./data/goeslib')
silversim = lib['MKID_20201013_Silver']
goldsim = lib['MKID_20201013_Kapton']

t,pose= (0, 6)
spec = silversim.spec(t, pose)
sat_spec_rate = S.Observation(spec.pysynphot, bandp).countrate()/425*flatfield_factor
expected_rate=flatfield_factor*phot_per_nm_raw*10**-((spec.mag_j-6.23)/2.5)
print(f'Expected raw rate for {spec.mag_j:.2f} Mj satellite {expected_rate/1e6:.2f} Mphot/nm/s')
print(f'Raw rate anticipated from {spec.mag_j:.2f} Mj satellite spec flux table '
      f'{sat_spec_rate/1e6:.2f} Mphot/nm/s')

t,pose= (32, 4)
spec = silversim.spec(t, pose)
sat_spec_rate = S.Observation(spec.pysynphot, bandp).countrate()/425*flatfield_factor
expected_rate=flatfield_factor*phot_per_nm_raw*10**-((spec.mag_j-6.23)/2.5)
print(f'Expected raw rate for {spec.mag_j:.2f} Mj satellite {expected_rate/1e6:.2f} Mphot/nm/s')
print(f'Raw rate anticipated from {spec.mag_j:.2f} Mj satellite spec flux table '
      f'{sat_spec_rate/1e6:.2f} Mphot/nm/s')
