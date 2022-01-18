import mkidcore
import mkidcore.config
from mkidpipeline.photontable import Photontable as Pt
from mkidcore.corelog import getLogger
import mkidpipeline.pipeline as pipe
import mkidpipeline.config as config
from mkidpipeline.definitions import MKIDObservation, MKIDWCSCalDescription, MKIDOutput, MKIDOutputCollection
import mkidpipeline.steps as steps
from datetime import datetime
import numpy as np

#Build a data object and outputs
getLogger('mkidcore', setup=True, logfile=f'mkidpipe_{datetime.now().strftime("%Y-%m-%d_%H%M")}.log')
getLogger('mkidpipe').setLevel('DEBUG')
getLogger('mkidpipeline').setLevel('DEBUG')
getLogger('mkidcore').setLevel('DEBUG')

startt = 0
duration = 10

header = {'DEC': '0:0:0.0', 'INSTRUME':'MEC', 'M_BASELI':True, 'M_BMAP':'MEC_default', 'M_CFGHSH':'FOO3123798AFE',
          'M_CONEXX':0, 'M_CONEXY':0,'EQUINOX':'J2000',
          #'M_FLTCAL', 'M_GITHSH', 'M_H5FILE', 'M_SPECAL', 'M_WAVCAL', 'M_WCSCAL', 'X_GRDAMP', 'X_GRDMOD', 'X_GRDSEP', 'X_GRDST'
          'OBJECT':'HIP 109427', 'RA':'0:0:0.0', 'TELESCOP':'Subaru'}
# , 'X_GRDAMP', 'X_GRDMOD', 'X_GRDSEP', 'X_GRDST'

dataset = (MKIDObservation(name='simulated', start=startt, duration=duration, wcscal='wcscal0', header=header),
           MKIDWCSCalDescription(name='wcscal0', pixel_ref=[107, 46], conex_ref=[-0.16, -0.4],
                                        data='10.40 mas'))
config.dump_dataconfig(dataset, 'simdata.yaml')

from mkidpipeline.samples import _namer
outputs = [MKIDOutput(name=_namer('out'), data='simulated', min_wave='950 nm', max_wave='1375 nm', kind=k,
                             flatcal=False, use_weights=False, cosmical=False, wavestep='85 nm', timestep=1,
                             units='photons', duration=10.0) for k in ('image', 'tcube', 'scube')]
with open('simout.yaml', 'w') as f:
    config.yaml.dump(outputs, f)

cfg = pipe.generate_default_config(instrument='MEC')
cfg.update('paths.out', './out/')
cfg.update('paths.tmp', './scratch/')
cfg.update('paths.database', './db/')
config.configure_pipeline(cfg)
outputs = MKIDOutputCollection('simout.yaml', datafile='simdata.yaml')
dataset = outputs.dataset
o = list(outputs)[0]

pipe.batch_applier('metadata', outputs)


#Patch the data object with key info
wavelengths = np.array([950, 1050, 1150, 1250, 1300.0])
from mkidsim.env  import R_ref, R_wave
powers = R_ref * R_wave / wavelengths
resdata = np.zeros(len(wavelengths),
                   dtype=np.dtype([('r', np.float32), ('r_err', np.float32), ('wave', np.float32)],
                                  align=True))
resdata['r'] = powers
resdata['wave'] = wavelengths
for obs in dataset.all_observations:
    pt = obs.photontable
    pt.enablewrite()
    pt.update_header('wavecal', 'EXACT')
    pt.update_header('wavecal.resolution', resdata)
    pt.update_header('M_WAVCAL', 'EXACT')
    pt.bad_pixel_mask
    pt.flag('beammap.noDacTone', p)
    pt.disablewrite()

pipe.batch_applier(steps.pixcal.apply, outputs.to_pixcal)
# pipe.batch_applier(steps.lincal.apply, outputs.to_lincal)
# pipe.batch_applier(steps.cosmiccal.apply, list(outputs.to_cosmiccal)[:1])
#Skip flatcal for now

steps.output.generate(outputs, remake=True)

from astropy.io import fits
hdul = fits.open('./out/simulated/out2_scube.fits')
yv,x=np.histogram(hdul['CUBE_EDGES'].data[:-1].edges, 5, weights=hdul['VARIANCE'].data.sum(1).sum(1))
y,x=np.histogram(hdul['CUBE_EDGES'].data[:-1].edges, 5, weights=hdul['SCIENCE'].data.sum(1).sum(1))
plt.errorbar(x[:-1]+np.diff(x)/2,y,yerr=np.sqrt(yv))
plt.errorbar(x[:-1]+np.diff(x)/2,y,yerr=np.sqrt(yv),fmt='o')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Photons')
plt.title('Extracted Spectrum')
