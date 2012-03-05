#!/usr/bin/env python
from distutils.core import setup, Extension

setup(name='pyfact',
      version='0.1.0',
      description='Python and fits based analysis for Cherenkov Telescopes',
      requires=['numpy', 'scipy', 'pyfits'],
      provides=['pyfact'],
      author='Martin Raue',
      author_email='martin.raue@desy.de',
      license='BSD',
      url='https://github.com/mraue/pyfact'
    )
