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
from goeslib import SatLib
import time

logging.basicConfig()
log = getLogger('photonsim')
getLogger('mkidcore', setup=True, logfile=f'mkidpipe_{datetime.now().strftime("%Y-%m-%d_%H%M")}.log')
getLogger('mkidpipe').setLevel('DEBUG')
getLogger('mkidpipeline').setLevel('DEBUG')
getLogger('mkidcore').setLevel('DEBUG')

exp_time = 10
psf_radius = .3 * u.arcsec / 2


lib = SatLib('./data/goeslib')
silversim = lib['MKID_20201013_Silver']
goldsim = lib['MKID_20201013_Kapton']

times = range(0,71,8)
poses = (0,4,5,8)
cfg = generate_default_config(instrument='MEC')
cfg.update('paths.out', '../out/')

tic=time.time()
for sim in (silversim, goldsim):
    for t in times:
        for pose in poses:
            sp = sim.spec(t, pose).pysynphot
            result = simulate_observation(sp, psf_radius, exp_time, nd=2.0)
            image, gen, observation, dc_throughput, flat_field, photons, total_launched = result
            buildfromarray(photons, config=cfg, user_h5file=f'./out2/{sim.name}_{t}_{pose}.h5')
toc=time.time()


# Build a data object and outputs
duration = exp_time

header = {'DEC': '0:0:0.0', 'INSTRUME': 'MEC', 'M_BASELI': True, 'M_BMAP': 'MEC_default',
          'M_CFGHSH': 'FOO3123798AFE', 'M_CONEXX': 0, 'M_CONEXY': 0, 'EQUINOX': 'J2000',
          'M_FLTCAL': 'flatcal0_b4a953dd_0b159b84.flatcal.npz',
          'OBJECT': 'GOES', 'RA': '0:0:0.0', 'TELESCOP': 'Subaru'}

outputs = []
dataset = []
for sim in (silversim, goldsim):
    for t in times:
        for pose in poses:
            dataset.append(MKIDObservation(name=f'{sim.name}_{t}_{pose}', start=0, duration=duration,
                                     wcscal='wcscal0', header=header, h5_file=f'{sim.name}_{t}_{pose}.h5'))
            outputs.append(MKIDOutput(name=f'{sim.name}_{t}_{pose}', data=f'{sim.name}_{t}_{pose}',
                                      min_wave='950 nm', max_wave='1375 nm', kind='scube', flatcal=False,
                                      use_weights=False, cosmical=False, wavestep='85 nm', timestep=1,
                                      units='photons/s', duration=duration))
dataset += [MKIDWCSCal(name='wcscal0', pixel_ref=[107, 46], conex_ref=[-0.16, -0.4], data='10.40 mas',
                       dp_dcx=-63.09, dp_dcy=67.61)]

config.dump_dataconfig(dataset, 'simdata.yaml')

with open('simout.yaml', 'w') as f:
    config.yaml.dump(outputs, f)

cfg = pipe.generate_default_config(instrument='MEC')
cfg.update('paths.out', './out2/')
cfg.update('paths.tmp', './scratch/')
cfg.update('paths.database', './db/')
with open('pipe.yaml', 'w') as f:
    config.yaml.dump(cfg, f)
config.configure_pipeline(cfg)
outputs = MKIDOutputCollection('simout.yaml', datafile='simdata.yaml')
dataset = outputs.dataset

pipe.batch_applier('attachmeta', outputs)
#
# Patch the data object with key info
wavelengths = np.array([950, 1050, 1150, 1250, 1300.0, 1375])
powers = R_ref * R_wave / wavelengths
resdata = np.zeros(len(wavelengths),
                   dtype=np.dtype([('r', np.float32), ('r_err', np.float32), ('wave', np.float32)], align=True))
resdata['r'] = powers
resdata['wave'] = wavelengths
for obs in dataset.all_observations:
    pt = P(obs.h5, mode='write')
    pt.update_header('wavecal', 'EXACT')
    pt.update_header('wavecal.resolution', resdata)
    pt.update_header('M_WAVCAL', 'EXACT')
    del pt

pipe.batch_applier('pixcal', outputs.to_pixcal)

config.make_paths(output_collection=outputs)
steps.output.generate(outputs, remake=False)
