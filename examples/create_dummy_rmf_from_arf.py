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

#===========================================================================
# Imports

import sys
import os
import logging

import numpy as np
import pyfits
import scipy.special
import scipy.interpolate
#import ROOT
import matplotlib.pyplot as plt

# Add script parent directory to python search path to get access to the pyfact package
sys.path.append(os.path.abspath(sys.path[0].rsplit('/', 1)[0]))
import pyfact as pf

#===========================================================================
# Functions

#===========================================================================
# Main

"""
DESCRIPTION MISSING
"""

#---------------------------------------------------------------------------
# Setup

# Setup fancy logging for output
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Read input file from command line
arf, rmf = '', ''
if len(sys.argv) != 3 :
    logging.info('You need to specifify the ARF input file and the output file name.')
    sys.exit(0)
else :
    arf = sys.argv[1]
    rmf = sys.argv[2]
    
#---------------------------------------------------------------------------
# Open ARF

f = pyfits.open(arf)
ea, ea_erange = pf.arf_to_np(f[1])
nbins = len(ea)
instrument = f[1].header['INSTRUME']
telescope = f[1].header['TELESCOP']

#---------------------------------------------------------------------------
# Create MRF

#rm = np.zeros([nbins, nbins])
#for i in range(nbins) :
#    rm[i][i] = 1.

sigma = .2

logerange = np.log10(ea_erange)
logemingrid = logerange[:-1] * np.ones([nbins, nbins])
logemaxgrid = logerange[1:] * np.ones([nbins, nbins])
logecentergrid = np.transpose(((logerange[:-1] + logerange[1:]) / 2.) * np.ones([nbins, nbins]))

#gauss = lambda p, x: p[0] / np.sqrt(2. * np.pi * p[2] ** 2.) * np.exp(- (x - p[1]) ** 2. / 2. / p[2] ** 2.)
gauss_int = lambda p, x_min, x_max: .5 * (scipy.special.erf((x_max - p[1]) / np.sqrt(2. * p[2] ** 2.)) - scipy.special.erf((x_min - p[1]) / np.sqrt(2. * p[2] ** 2.)))

rm = gauss_int([1., 10. ** logecentergrid, sigma], 10. ** logemingrid, 10. ** logemaxgrid)

logging.info('Sanity check, integrated rows should be 1.: {0}'.format(np.sum(rm, axis=1)))

# Create RM hdulist
hdulist = pf.np_to_rmf(rm, ea_erange, ea_erange, 1E-5,
                       telescope=telescope, instrument=instrument)
# Write RM to file
hdulist.writeto(rmf)

# DEBUG plots
#plt.subplot(221)
#plt.imshow(np.log10(rm[::-1]),  interpolation='nearest')
#cb = plt.colorbar()
#plt.subplot(222)
#plt.imshow(logecentergrid,  interpolation='nearest')
#cb = plt.colorbar()
#plt.subplot(223)
#plt.imshow(logemingrid,  interpolation='nearest')
#cb = plt.colorbar()
#plt.subplot(224)
#plt.imshow(logemaxgrid,  interpolation='nearest')
#cb = plt.colorbar()
#plt.show()

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
