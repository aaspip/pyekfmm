#!/usr/bin/env python
# -*- encoding: utf8 -*-

from setuptools import setup
from distutils.core import Extension
import numpy

long_description = """
Source code: https://github.com/aaspip/pyekfmm""".strip() 

eikonalc_module = Extension('eikonalc', sources=['pyekfmm/src/eikonal.c'], 
										include_dirs=[numpy.get_include()])

eikonalvtic_module = Extension('eikonalvtic', sources=['pyekfmm/src/eikonalvti.c'], 
										include_dirs=[numpy.get_include()])

setup(
    name="pyekfmm",
    version="0.0.8.5",
    license='GNU General Public License, Version 3 (GPLv3)',
    description="Fast Marching Method for Traveltime Calculation",
    long_description=long_description,
    author="pyekfmm developing team",
    author_email="chenyk2016@gmail.com",
    url="https://github.com/aaspip/pyekfmm",
    ext_modules=[eikonalc_module,eikonalvtic_module],
    packages=['pyekfmm'],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"
    ],
    keywords=[
        "seismology", "exploration seismology", "array seismology", "traveltime", "ray tracing", "earthquake location", "earthquake relocation", "surface wave tomography", "body wave tomography"
    ],
    install_requires=[
        "numpy", "scipy", "matplotlib"
    ],
    extras_require={
        "docs": ["sphinx", "ipython", "runipy"]
    }
    
)
