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
import ROOT
import matplotlib.pyplot as plt

# Add script parent directory to python search path to get access to the pyfact package
sys.path.append(os.path.abspath(sys.path[0].rsplit('/', 1)[0]))
import pyfact as pf

#===========================================================================
# Functions

#---------------------------------------------------------------------------
def root_axis_to_array(ax) :
    a = np.zeros(ax.GetNbins() + 1)
    for i in range(ax.GetNbins()) :
        a[i] = ax.GetBinLowEdge(i + 1)
    a[-1] = ax.GetBinUpEdge(ax.GetNbins())
    return a

#---------------------------------------------------------------------------
def root_1dhist_to_array(hist) :
    nbins = hist.GetXaxis().GetNbins()
    a = np.zeros(nbins)
    for i in range(nbins) :
        a[i] = hist.GetBinContent(i + 1)
    return a

#---------------------------------------------------------------------------
def root_2dhist_to_array(hist2d) :
    nbinsx = hist2d.GetXaxis().GetNbins()
    nbinsy = hist2d.GetYaxis().GetNbins()
    a = np.zeros([nbinsx, nbinsy])
    for x in range(nbinsx) :
        for y in range(nbinsy) :
            a[x, y] = hist2d.GetBinContent(x + 1, y + 1)
    return a

#---------------------------------------------------------------------------
def root_th1_to_fitstable() :
    pass

#---------------------------------------------------------------------------
def root_th2_to_fitsimage() :
    pass

#===========================================================================
# Main

"""
This script converts a CTA response stored in a root file into a set of FITS
files, namely ARF, RMF, and one auxiliary file, which stores all information
from the response file in simple fits tables.
"""

#---------------------------------------------------------------------------
# Setup

# Setup fancy logging for output
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# Read input file from command line
#irf_root_file_name = '/Users/mraue/Stuff/work/cta/2011/fits/irf/cta/SubarrayE_IFAE_50hours_20101102.root'
irf_root_file_name = ''
if len(sys.argv) != 2 :
    logging.info('You need to specifify the CTA IRF root file as a command line option')
    sys.exit(0)
else :
    irf_root_file_name = sys.argv[1]

#---------------------------------------------------------------------------
# Open CTA response file in root format

irf_file_name_base = irf_root_file_name.rsplit('.',1)[0].rsplit('/',1)[1]

irf_root_file = ROOT.TFile(irf_root_file_name)
irf_root_file.ls()

eff_area_root_hist = irf_root_file.Get('EffectiveArea;1')

ea_loge = np.round(root_axis_to_array(eff_area_root_hist.GetXaxis()), decimals=5)
ea_val = root_1dhist_to_array(eff_area_root_hist)

#---------------------------------------------------------------------------
# Write ARF & MRF

# Read RM
h = irf_root_file.Get('MigMatrix;1')
# Transpose and normalize RM
rm = np.transpose(root_2dhist_to_array(h))
n = np.transpose(np.sum(rm, axis=1) * np.ones(rm.shape[::-1]))
rm[rm > 0.] /= n[rm > 0.]
# Read bin enery ranges
rm_erange_log = root_axis_to_array(h.GetYaxis())
rm_ebounds_log = root_axis_to_array(h.GetXaxis())
# Create RM hdulist
hdulist = pf.np_to_rmf(rm,
                       (10. ** rm_erange_log).round(decimals=6),
                       (10. ** rm_ebounds_log).round(decimals=6),
                       1E-5,
                       telescope='CTASIM')
# Write RM to file
hdulist.writeto(irf_file_name_base + '.rmf.fits')

# Read EA
h = irf_root_file.Get('EffectiveArea;1')
ea = root_1dhist_to_array(h)
# Read EA bin energy ranges
ea_erange_log = root_axis_to_array(h.GetXaxis())
# Re-sample EA to match RM
resample_ea_to_mrf = True
#resample_ea_to_mrf = False
if resample_ea_to_mrf :
        logging.info('Resampling effective area in log10(EA) vs log10(E) to match RM.')
        ea_spl = scipy.interpolate.UnivariateSpline(10. ** ((ea_erange_log[1:] + ea_erange_log[:-1]) / 2.), np.log10(ea), s=0, k=1)
        e = (rm_erange_log[1:] + rm_erange_log[:-1]) / 2.
        ea = 10. ** ea_spl(10. ** e)
        ea_erange_log = rm_erange_log
tbhdu = pf.np_to_arf(ea,
                     (10. ** ea_erange_log).round(decimals=6),
                     telescope='CTASIM')
# Write AR to file
tbhdu.writeto(irf_file_name_base + '.arf.fits')


# DEBUG
#t = np.transpose(root_2dhist_to_array(h))
#print 'XX', np.sum(rm, axis=0)
#print 'YY', np.sum(rm, axis=1)

print 'XXX', np.sum(np.dot(np.ones(len(rm)), rm)), rm.shape, np.sum(np.sum(rm, axis=1))

#---------------------------------------------------------------------------
# Close CTA IRF root file

irf_root_file.Close()
sys.exit(0)

#---------------------------------------------------------------------------
# DEBUG plots

f = pyfits.open('SubarrayE_IFAE_50hours_20101102.rmf.fits')

plt.set_cmap(plt.cm.Purples)

plt.subplot(221)
a,b,c,d = pf.rmf_to_np(f)
#print np.log10(b[1:]) - np.log10(b[:-1])
#print np.log10(c[1:]) - np.log10(c[:-1])
extent=np.log10(np.array([c[0], c[-1], b[-1], b[0]]))
plt.imshow((a - rm)[::-1], extent=extent)
plt.colorbar()

plt.subplot(222)

plt.imshow(rm[::-1], extent=extent)
plt.colorbar()

plt.subplot(223)

plt.imshow(a[::-1], extent=extent)
plt.colorbar()

plt.subplot(224)
plt.hist(a - rm, bins=31)

#---------------------------------------------------------------------------
# Plot

#plt.semilogy((ea_loge[:-1] + ea_loge[1:]) / 2., ea_val)
#
#plt.title(r'I come from a root file \o/')
#plt.xlabel(r'log10(Energy / 1 TeV)')
#plt.ylabel(r'Effective area (m$^2$)')

plt.show()

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
