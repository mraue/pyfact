#===========================================================================
# Copyright (c) 2011, Martin Raue
# All rights reserved.
#
# LICENSE
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the PyFACT developers nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL MARTIN RAUE BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

import logging

import numpy as np
import pyfits

"""
===========
DESCRIPTION

This script demonstrates how to write a response matrix (energy resolution/redistribution matrix)
from a numpy 2d array to a FITS file in the RMF standard.

For more info on the RMF FITS file format see:
http://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/summary/cal_gen_92_002_summary.html

"""

#----------------------------------------------------------------------
# INIT

# Setup fancy logging for output
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

#----------------------------------------------------------------------
# INPUT

# Create dummy energy distribution matrix (response matrix, RM)
# with diagonal elements 1 and all other elements 0.
# The matrix should be bins of E_true vs E_reco and contains the probability
# that an event with energy E_true is reconstructed as E_reco.
rm = np.zeros([16, 16])
for i in range(16) :
    rm[i][i] = 1.
    if i > 1 :
        rm[i][i-1] = .5
        
# Set one row to something a bit more elaborated to demonstrate that the code works
rm[0] = np.array([1., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1., 1., 1., 0., 1., 0.])

# Create dummy arrays with the energy ranges covered by the matrix.
# Here we use energies from 0.01 to 100 TeV with four log10 steps per decade.
erange = 10. ** np.linspace(-2., 2., 17) # E_true [TeV]
ebounds = 10. ** np.linspace(-2., 2., 17) # E_reco [TeV]

# Minimal probability to be stored in the RMF
minprob = 1E-7

#----------------------------------------------------------------------
# CREATE FITS OUTPUT

# Intialize the arrays to be used to construct the RM extension
n_rows = len(rm)
energy_lo = np.zeros(n_rows) # Low energy bounds
energy_hi = np.zeros(n_rows) # High energy bounds
n_grp = [] # Number of channel subsets
f_chan = [] # First channels in each subset
n_chan = [] # Number of channels in each subset
matrix = [] # Matrix elements

# Loop over the matrix and fill the arrays
for i, r in enumerate(rm) :
    energy_lo[i] = erange[i]
    energy_hi[i] = erange[i + 1]
    # Create mask for all matrix row values above the minimal probability
    m = r > minprob
    # Intialize variables & arrays for the row
    n_grp_row, n_chan_row_c = 0, 0
    f_chan_row, n_chan_row, matrix_row = [], [], []
    new_subset = True
    # Loop over row entries and fill arrays appropriately
    for j, v in enumerate(r) :
        if m[j] :
            if new_subset :
                n_grp_row += 1
                f_chan_row.append(j + 1)
                new_subset = False
            matrix_row.append(v)
            n_chan_row_c += 1
        else :
            if not new_subset :
                n_chan_row.append(n_chan_row_c)
                n_chan_row_c = 0
                new_subset = True
    if not new_subset :
        n_chan_row.append(n_chan_row_c)
    n_grp.append(n_grp_row)
    f_chan.append(f_chan_row)
    n_chan.append(n_chan_row)
    matrix.append(matrix_row)

# Create RMF FITS table extension from data
tbhdu = pyfits.new_table(
    [pyfits.Column(name='ENERGY_LO',
                  format='1E',
                  array=energy_lo,
                  unit='TeV'),
     pyfits.Column(name='ENERGY_HI',
                  format='1E',
                  array=energy_hi,
                  unit='TeV'),
     pyfits.Column(name='N_GRP',
                   format='1I',
                   array=n_grp),
     pyfits.Column(name='F_CHAN',
                   format='PI()',
                   array=f_chan),
     pyfits.Column(name='N_CHAN',
                   format='PI()',
                   array=n_chan),
     pyfits.Column(name='MATRIX',
                   format='PE(()',
                   array=matrix)
     ]
    )

# Write FITS extension header

chan_min, chan_max, chan_n = 0, rm.shape[1] - 1, rm.shape[1]

tbhdu.header.update('EXTNAME ', 'MATRIX', 'name of this binary table extension')
tbhdu.header.update('TLMIN4  ', chan_min, 'First legal channel number')
tbhdu.header.update('TLMAX4  ', chan_max, 'Highest legal channel number')
tbhdu.header.update('TELESCOP', 'DUMMY', 'mission/satellite name')
tbhdu.header.update('INSTRUME', 'DUMMY', 'instrument/detector')
tbhdu.header.update('FILTER  ', 'NONE    ', 'filter information')
tbhdu.header.update('CHANTYPE', 'PI      ', 'Type of channels (PHA, PI etc)')
tbhdu.header.update('DETCHANS', chan_n, 'Total number of detector PHA channels')
tbhdu.header.update('LO_THRES', minprob, 'Lower probability density threshold for matrix')
tbhdu.header.update('HDUCLASS', 'OGIP', 'Organisation devising file format')
tbhdu.header.update('HDUCLAS1', 'RESPONSE', 'File relates to response of instrument')
tbhdu.header.update('HDUCLAS2', 'RSP_MATRIX', 'Keyword information for Caltools Software.')
tbhdu.header.update('HDUVERS ', '1.3.0', 'Version of file format')
tbhdu.header.update('HDUCLAS3', 'DETECTOR', 'Keyword information for Caltools Software.')
tbhdu.header.update('CCNM0001', 'MATRIX', 'Keyword information for Caltools Software.')
tbhdu.header.update('CCLS0001', 'CPF', 'Keyword information for Caltools Software.')
tbhdu.header.update('CDTP0001', 'DATA', 'Keyword information for Caltools Software.')

# UTC date when this calibration should be first used (yyy-mm-dd)
tbhdu.header.update('CVSD0001', '2011-01-01 ', 'Keyword information for Caltools Software.')

# UTC time on the dat when this calibration should be first used (hh:mm:ss)
tbhdu.header.update('CVST0001', '00:00:00', 'Keyword information for Caltools Software.')

# String giving a brief summary of this data set
tbhdu.header.update('CDES0001', r'dummy data - do not use', 'Keyword information for Caltools Software.')

# Optional, but maybe useful (taken from the example in the RMF/ARF document)
tbhdu.header.update('CBD10001', 'CHAN({0}- {1})'.format(chan_min, chan_max), 'Keyword information for Caltools Software.')
tbhdu.header.update('CBD20001', 'ENER({0}-{1})TeV'.format(erange[0], erange[-1]), 'Keyword information for Caltools Software.')

# Obsolet RMF headers, included for the benefit of old software
tbhdu.header.update('RMFVERSN', '1992a', 'Obsolete')
tbhdu.header.update('HDUVERS1', '1.1.0', 'Obsolete')
tbhdu.header.update('HDUVERS2', '1.2.0', 'Obsolete')

# Create EBOUNDS FITS table extension from data
tbhdu2 = pyfits.new_table(
    [pyfits.Column(name='CHANNEL',
                   format='1I',
                   array=np.arange(len(ebounds) - 1)),
     pyfits.Column(name='E_MIN',
                  format='1E',
                  array=ebounds[:-1],
                  unit='TeV'),
     pyfits.Column(name='E_MAX',
                  format='1E',
                  array=ebounds[1:],
                  unit='TeV')
     ]
    )

tbhdu2.header.update('EXTNAME ', 'EBOUNDS', 'Name of this binary table extension')
tbhdu2.header.update('TELESCOP', 'DUMMY', 'Mission/satellite name')
tbhdu2.header.update('INSTRUME', 'DUMMY', 'Instrument/detector')
tbhdu2.header.update('FILTER  ', 'NONE    ', 'Filter information')
tbhdu2.header.update('CHANTYPE', 'PI      ', 'Type of channels (PHA, PI etc)')
tbhdu2.header.update('DETCHANS', chan_n, 'Total number of detector PHA channels')
tbhdu2.header.update('HDUCLASS', 'OGIP', 'Organisation devising file format')
tbhdu2.header.update('HDUCLAS1', 'RESPONSE', 'File relates to response of instrument')
tbhdu2.header.update('HDUCLAS2', 'EBOUNDS', 'This is an EBOUNDS extension')
tbhdu2.header.update('HDUVERS ', '1.2.0', 'Version of file format')
tbhdu2.header.update('HDUCLAS3', 'DETECTOR', 'Keyword information for Caltools Software.')
tbhdu2.header.update('CCNM0001', 'EBOUNDS', 'Keyword information for Caltools Software.')
tbhdu2.header.update('CCLS0001', 'CPF', 'Keyword information for Caltools Software.')
tbhdu2.header.update('CDTP0001', 'DATA', 'Keyword information for Caltools Software.')

# UTC date when this calibration should be first used (yyy-mm-dd)
tbhdu2.header.update('CVSD0001', '2011-01-01 ', 'Keyword information for Caltools Software.')

# UTC time on the dat when this calibration should be first used (hh:mm:ss)
tbhdu2.header.update('CVST0001', '00:00:00', 'Keyword information for Caltools Software.')

# Optional - name of the PHA file for which this file was produced
#tbhdu2.header.update('PHAFILE', '', 'Keyword information for Caltools Software.')

# String giving a brief summary of this data set
tbhdu2.header.update('CDES0001', r'dummy data - do not use', 'Keyword information for Caltools Software.')

# Obsolet EBOUNDS headers, included for the benefit of old software
tbhdu2.header.update('RMFVERSN', '1992a', 'Obsolete')
tbhdu2.header.update('HDUVERS1', '1.0.0', 'Obsolete')
tbhdu2.header.update('HDUVERS2', '1.1.0', 'Obsolete')

# Some output for debug
logging.info(tbhdu.header)
logging.info(tbhdu2.header)

# Create primary HDU and HDU list to be stored in the output file
hdu = pyfits.PrimaryHDU()
hdulist = pyfits.HDUList([hdu, tbhdu, tbhdu2])

#----------------------------------------------------------------------
# WRITE FITS OUTPUT

# Write RM & EBOUNDS FITS table to file
hdulist.writeto('rmf_example.fits')

#----------------------------------------------------------------------
#----------------------------------------------------------------------
