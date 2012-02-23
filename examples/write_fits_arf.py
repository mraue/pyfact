#===========================================================================
# Copyright (c) 2011-2012, Martin Raue
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

#===========================================================================
# Imports

import logging

import numpy as np
import pyfits

#===========================================================================
# Main

#----------------------------------------------------------------------
# SETUP

# Setup fancy logging for output
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

#----------------------------------------------------------------------
# INPUT

# Create dummy effective area (EA; or auxiliary response; AR)
ea = np.array([0., 1E2, 1E4, 1E6, 1E6, 1E6, 1E6, 1E6,
               1E6,1E6, 1E6, 1E6, 1E6, 1E6, 1E5, 1E4]) # [m^2]

# Create dummy arrays with the energy ranges covered by the EA
# Here we use energies from 0.01 to 100 TeV with four log10 steps per decade.
erange = 10. ** np.linspace(-2., 2., 17) # E_true [TeV]

#----------------------------------------------------------------------
# CREATE FITS OUTPUT

# Create EA/AR FITS table extension from data
# Recommended units are keV and cm^2, but TeV and m^2 are chosen
# as the more natural units for IACTs
tbhdu = pyfits.new_table(
    [pyfits.Column(name='ENERG_LO',
                  format='1E',
                  array=erange[:-1],
                  unit='TeV'),
     pyfits.Column(name='ENERG_HI',
                  format='1E',
                  array=erange[1:],
                  unit='TeV'),
     pyfits.Column(name='SPECRESP',
                  format='1E',
                  array=ea,
                  unit='m^2')
     ]
    )

# Write FITS extension header
tbhdu.header.update('EXTNAME ', 'SPECRESP', 'Name of this binary table extension')
tbhdu.header.update('TELESCOP', 'DUMMY', 'Mission/satellite name')
tbhdu.header.update('INSTRUME', 'DUMMY', 'Instrument/detector')
tbhdu.header.update('FILTER  ', 'NONE    ', 'Filter information')
tbhdu.header.update('HDUCLASS', 'OGIP', 'Organisation devising file format')
tbhdu.header.update('HDUCLAS1', 'RESPONSE', 'File relates to response of instrument')
tbhdu.header.update('HDUCLAS2', 'SPECRESP', 'Effective area data is stored')
tbhdu.header.update('HDUVERS ', '1.1.0', 'Version of file format')

# Optional
#tbhdu.header.update('PHAFILE', '', 'PHA file for which ARF was produced')

# Obsolet ARF headers, included for the benefit of old software
tbhdu.header.update('ARFVERSN', '1992a', 'Obsolete')
tbhdu.header.update('HDUVERS1', '1.0.0', 'Obsolete')
tbhdu.header.update('HDUVERS2', '1.1.0', 'Obsolete')

#----------------------------------------------------------------------
# WRITE FITS OUTPUT

# Write EA/AR to FITS file
tbhdu.writeto('arf_example.fits')

#----------------------------------------------------------------------
#----------------------------------------------------------------------
