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
import mkidsim.env
import time

logging.basicConfig()
log = getLogger('photonsim')
getLogger('mkidcore', setup=True, logfile=f'mkidpipe_{datetime.now().strftime("%Y-%m-%d_%H%M")}.log')
getLogger('mkidpipe').setLevel('DEBUG')
getLogger('mkidpipeline').setLevel('DEBUG')
getLogger('mkidcore').setLevel('DEBUG')
getLogger('mkidcore.config').setLevel('INFO')

exp_time = 10
psf_radius = .3 * u.arcsec / 2

from goeslib import SatLib
lib = SatLib('./data/goeslib')
silversim = lib['MKID_20201013_Silver']
goldsim = lib['MKID_20201013_Kapton']


# posetimes = ((0,0), (32,4))
posetimes = ((0, 6), (32, 4))
magnitudes = np.array([14, 18, 22])  #14, 15, 18, 20, 22])  Was simulating at ~16-17
R0 = np.array([4, 16, 64])  #2, 4, 8, 16, 32, 64])

mkidsim.env.R_wave = detector_wavelength_range[1]

cfg = generate_default_config(instrument='MEC')
cfg.update('paths.out', '../grid/')
tic = time.time()
for m in magnitudes:
    for sim in (silversim, goldsim):
        for t, pose in posetimes:
            spec = sim.spec(t, pose)
            sp = spec.pysynphot
            for r in R0:
                mkidsim.env.R_ref = r
                nd = (m-spec.mag_j)/2.5
                result = simulate_observation(sp, psf_radius, exp_time, nd=nd)
                image, gen, observation, dc_throughput, flat_field, photons, total_launched = result
                buildfromarray(photons, config=cfg, user_h5file=f'./grid/{sim.name}_{t}_{pose}_{nd}_{r}.h5')
toc = time.time()




# Build a data object and outputs


duration = exp_time

header = {'DEC': '0:0:0.0', 'INSTRUME': 'MEC', 'M_BASELI': True, 'M_BMAP': 'MEC_default',
          'M_CFGHSH': 'FOO3123798AFE', 'M_CONEXX': 0, 'M_CONEXY': 0, 'EQUINOX': 'J2000',
          'M_FLTCAL': 'flatcal0_b4a953dd_0b159b84.flatcal.npz',
          #'M_GITHSH', 'M_H5FILE', 'M_SPECAL', 'M_WAVCAL', 'M_WCSCAL', 'X_GRDAMP', 'X_GRDMOD', 'X_GRDSEP', 'X_GRDST'
          'OBJECT': 'GOES', 'RA': '0:0:0.0', 'TELESCOP': 'Subaru'}

outputs = []
dataset = []
for m in magnitudes:
    for sim in (silversim, goldsim):
        for t, pose in posetimes:
            spec = sim.spec(t, pose)
            sp = spec.pysynphot
            for r in R0:
                nd = (m - spec.mag_j) / 2.5
                dataset.append(MKIDObservation(name=f'{sim.name}_{t}_{pose}_{nd}_{r}', start=0, duration=duration,
                                               # flatcal='flatcal0_b4a953dd_0b159b84.flatcal.npz',
                                               wcscal='wcscal0', header=header, h5_file=f'{sim.name}_{t}_{pose}_{nd}_{r}.h5'))
                outputs.append(MKIDOutput(name=f'{sim.name}_{t}_{pose}_{nd}_{m}_{r}', data=f'{sim.name}_{t}_{pose}_{nd}_{r}',
                                          min_wave='950 nm', max_wave='1375 nm', kind='scube', flatcal=False,
                                          use_weights=False, cosmical=False, wavestep='85 nm', timestep=1,
                                          units='photons/s', duration=duration))
dataset += [MKIDWCSCal(name='wcscal0', pixel_ref=[107, 46], conex_ref=[-0.16, -0.4], data='10.40 mas',
                       dp_dcx=-63.09, dp_dcy=67.61)]
config.dump_dataconfig(dataset, 'simdata.yaml')
with open('simout.yaml', 'w') as f:
    config.yaml.dump(outputs, f)

cfg = pipe.generate_default_config(instrument='MEC')
cfg.update('paths.out', './grid/')
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
wavelengths = np.array([950, 1050, 1150, 1250, 1300.0])
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

# config.config.update('ncpu', 5)
# config.n_cpus_available()
#TODO config not getting carried across processes???

# o=list(outputs)[0]
# pt=P(o.data.obs[0].h5)
# config.config.update('pixcal.remake', True)

# pc, pcmeta = steps.pixcal.fetch(o.photontable, o.start, o.stop, config=config.config)
#


pipe.batch_applier('pixcal', outputs.to_pixcal)

# pipe.batch_applier('lincal', outputs.to_lincal)
# for obs in dataset.all_observations:
#     obs.flatcal = './flatcal0_b4a953dd_0b159b84.flatcal.npz'
# for o in outputs:
#     o.flatcal=True
# pipe.batch_applier('flatcal', outputs.to_flatcal)

config.make_paths(output_collection=outputs)
steps.output.generate(outputs, remake=False)
