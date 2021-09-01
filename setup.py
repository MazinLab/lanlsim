from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='filterphotpython',
    ext_modules=cythonize("filterphot.pyx"),
    zip_safe=False,include_dirs=[numpy.get_include()]
)
