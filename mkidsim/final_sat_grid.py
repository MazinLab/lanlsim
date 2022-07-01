import os.path

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
import mkidsim.env

logging.basicConfig()
log = getLogger('photonsim')
getLogger('mkidcore', setup=True, logfile=f'mkidpipe_{datetime.now().strftime("%Y-%m-%d_%H%M")}.log')
getLogger('mkidpipe').setLevel('DEBUG')
getLogger('mkidpipeline').setLevel('DEBUG')
getLogger('mkidcore').setLevel('DEBUG')
getLogger('mkidcore.config').setLevel('INFO')

exp_time = 10
psf_radius = 0.01828 * u.arcsec  # radius of first Airy zero 1.22*(950+1375)/2*1e-9/8*206265*1e3/2

from goeslib import SatLib
lib = SatLib('./data/goeslib')
silversim = lib['MKID_20201013_Silver']
goldsim = lib['MKID_20201013_Kapton']


posetimes = ((0, 6), (32, 4))
magnitudes = np.array([14, 18, 22])
R0 = np.array([4, 16, 64])

mkidsim.env.R_wave = detector_wavelength_range[1]

cfg = generate_default_config(instrument='MEC')
cfg.update('paths.out', '../grid/')
for m in magnitudes:
    for sim in (silversim, goldsim):
        for t, pose in posetimes:
            spec = sim.spec(t, pose)
            sp = spec.pysynphot
            for r in R0:
                mkidsim.env.R_ref = r
                nd = (m - spec.mag_j) / 2.5
                name = f'{sim.name}_{t}_{pose}_{nd:.3f}_{m}_{r}'
                hf = f'./grid/{name}.h5'
                if os.path.exists(hf):
                    continue
                result = simulate_observation(sp, psf_radius, exp_time, nd=nd)
                image, gen, observation, dc_throughput, flat_field, photons, total_launched = result
                buildfromarray(photons, config=cfg, user_h5file=hf)


# Build a data object and outputs
if True or not os.path.exists('simout.yaml'):
    outputs, dataset = [], []
    header = {'DEC': '0:0:0.0', 'INSTRUME': 'MEC', 'M_BASELI': True, 'M_BMAP': 'MEC_default',
              'M_CFGHSH': 'FOO3123798AFE', 'M_CONEXX': 0, 'M_CONEXY': 0, 'EQUINOX': 'J2000',
              'M_FLTCAL': 'flatcal0_b4a953dd_0b159b84.flatcal.npz',
              'OBJECT': 'GOES', 'RA': '0:0:0.0', 'TELESCOP': 'Subaru'}
    for m in magnitudes:
        for sim in (silversim, goldsim):
            for t, pose in posetimes:
                spec = sim.spec(t, pose)
                sp = spec.pysynphot
                for r in R0:
                    nd = (m - spec.mag_j) / 2.5
                    name = f'{sim.name}_{t}_{pose}_{nd:.3f}_{m}_{r}'
                    output_name = f'{name}_out'
                    dataset.append(MKIDObservation(name=name, start=0, duration=exp_time,
                                                   wcscal='wcscal0', header=header, h5_file=f'{name}.h5'))
                    outputs.append(MKIDOutput(name=output_name, data=name, min_wave='950 nm', max_wave='1375 nm',
                                              kind='scube', flatcal=False, use_weights=False, cosmical=False,
                                              timestep=1, units='photons/s', duration=exp_time,
                                              exclude_flags='none'))

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

config.configure_pipeline('pipe.yaml')
outputs = MKIDOutputCollection('simout.yaml', datafile='simdata.yaml')
config.make_paths(output_collection=outputs)

# ============ Run the pipeline ============

pipe.batch_applier('attachmeta', outputs)

#Patch the data object with wavelength solution info
wavelengths = np.array([950, 1050, 1150, 1250, 1300.0, 1375.0]) * u.nm
resdata_type = np.dtype([('r', np.float32), ('r_err', np.float32), ('wave', np.float32)], align=True)
resdata = np.zeros(len(wavelengths), dtype=resdata_type)
resdata['wave'] = wavelengths
resdata['r_err'] = 1.0

for m in magnitudes:
    for sim in (silversim, goldsim):
        for t, pose in posetimes:
            spec = sim.spec(t, pose)
            for r in R0:
                nd = (m - spec.mag_j) / 2.5
                name = f'{sim.name}_{t}_{pose}_{nd:.3f}_{m}_{r}'
                resdata['r'] = (r * mkidsim.env.R_wave / wavelengths).value
                pt = P(f'./grid/{name}.h5', mode='write')
                pt.update_header('min_wavelength', detector_wavelength_range[0])
                pt.update_header('max_wavelength', detector_wavelength_range[1])
                pt.update_header('wavecal', 'EXACT')
                pt.update_header('M_WAVCAL', 'EXACT')
                pt.update_header('wavecal.resolution', resdata)
                del pt

pipe.batch_applier('pixcal', outputs.to_pixcal)

steps.output.generate(outputs, remake=True)
