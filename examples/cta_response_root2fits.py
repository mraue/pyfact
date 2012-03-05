#===========================================================================
# Copyright (c) 2011, the PyFACT developers
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
# DISCLAIMED. IN NO EVENT SHALL THE PYFACT DEVELOPERS BE LIABLE FOR ANY
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
    a, e = np.zeros(nbins), np.zeros(nbins)
    for i in range(nbins) :
        a[i] = hist.GetBinContent(i + 1)
        e[i] = hist.GetBinError(i + 1)
    return (a, e)

#---------------------------------------------------------------------------
def root_2dhist_to_array(hist2d) :
    nbinsx = hist2d.GetXaxis().GetNbins()
    nbinsy = hist2d.GetYaxis().GetNbins()
    a = np.zeros([nbinsx, nbinsy])
    e = np.zeros([nbinsx, nbinsy])
    for x in range(nbinsx) :
        for y in range(nbinsy) :
            a[x, y] = hist2d.GetBinContent(x + 1, y + 1)
            e[x, y] = hist2d.GetBinError(x + 1, y + 1)            
    return (a, e)

#---------------------------------------------------------------------------
def root_th1_to_fitstable(hist, xunit='', yunit='') :
    d, e = root_1dhist_to_array(hist)
    ax = root_axis_to_array(hist.GetXaxis())
    tbhdu = pyfits.new_table(
        [pyfits.Column(name='BIN_LO',
                       format='1E',
                       array=ax[:-1],
                       unit=xunit),
         pyfits.Column(name='BIN_HI',
                       format='1E',
                       array=ax[1:],
                       unit=xunit),
         pyfits.Column(name='VAL',
                       format='1E',
                       array=d,
                       unit=yunit),
         pyfits.Column(name='ERR',
                       format='1E',
                       array=e,
                       unit=yunit)
         ]
        )
    tbhdu.header.update('ROOTTI', hist.GetTitle(), 'ROOT hist. title')
    tbhdu.header.update('ROOTXTI', hist.GetXaxis().GetTitle(), 'ROOT X-axis title')
    tbhdu.header.update('ROOTYTI', hist.GetYaxis().GetTitle(), 'ROOT Y-axis title')
    tbhdu.header.update('ROOTUN', hist.GetBinContent(0), 'ROOT n underflow')
    tbhdu.header.update('ROOTOV', hist.GetBinContent(hist.GetXaxis().GetNbins() + 1), 'ROOT n overflow')
    return tbhdu

#---------------------------------------------------------------------------
def plot_th1(hist, logy=False) :
    d, e = root_1dhist_to_array(hist)
    ax = root_axis_to_array(hist.GetXaxis())
    if logy :
        plt.semilogy((ax[:-1] + ax[1:]) / 2., d)        
    else :
        plt.plot((ax[:-1] + ax[1:]) / 2., d)
    plt.xlabel(hist.GetXaxis().GetTitle(), fontsize='small')
    plt.ylabel(hist.GetYaxis().GetTitle(), fontsize='small')
    plt.title(hist.GetTitle(), fontsize='small')
    fontsize='small'
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)    
    return

#---------------------------------------------------------------------------
def fit_th1(fitter, p0, hist, errscale=None, range_=None, xaxlog=True) :
    y, yerr = root_1dhist_to_array(hist)
    if errscale :
        yerr = errscale * y
    ax = root_axis_to_array(hist.GetXaxis())
    x = ((ax[:-1] + ax[1:]) / 2.)
    if xaxlog :
        x = 10 ** x
    if range_ is not None :
        m = (x >= range_[0]) * (x <= range_[1])
        x = x[m]
        y = y[m]
        yerr = yerr[m]
    fitter.fit_data(p0, x, y, yerr)
    return (fitter, x, y, yerr)

#---------------------------------------------------------------------------
#def root_th2_to_fitsimage() :
#    pass

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

write_output = False

#---------------------------------------------------------------------------
# Open CTA response file in root format

irf_file_name_base = irf_root_file_name.rsplit('.',1)[0].rsplit('/',1)[1]

logging.info('Reading IRF data from file {0}'.format(irf_root_file_name))
irf_root_file = ROOT.TFile(irf_root_file_name)
logging.info('File content (f.ls()) :')
irf_root_file.ls()

#---------------------------------------------------------------------------
# Write ARF & MRF

#----------------------------------------------
# Read RM
h = irf_root_file.Get('MigMatrix')
rm_erange_log, rm_ebounds_log = None, None
if h != None :
    # Transpose and normalize RM
    rm = np.transpose(root_2dhist_to_array(h)[0])
    n = np.transpose(np.sum(rm, axis=1) * np.ones(rm.shape[::-1]))
    rm[rm > 0.] /= n[rm > 0.]
    # Read bin enery ranges
    rm_erange_log = root_axis_to_array(h.GetYaxis())
    rm_ebounds_log = root_axis_to_array(h.GetXaxis())
else :
    logging.info('ROOT file does not contain MigMatrix.')
    logging.info('Will produce RMF from ERes histogram.')

    # Read energy resolution
    h = irf_root_file.Get('ERes')
    d = root_1dhist_to_array(h)[0]
    ax = root_axis_to_array(h.GetXaxis())

    # Resample to higher resolution in energy
    nbins = int((ax[-1] - ax[0]) * 20) # 20 bins per decade
    rm_erange_log = np.linspace(ax[0], ax[-1], nbins + 1)
    rm_ebounds_log = rm_erange_log

    sigma = scipy.interpolate.UnivariateSpline((ax[:-1] + ax[1:]) / 2., d, s=0, k=1)

    logerange = rm_erange_log
    logemingrid = logerange[:-1] * np.ones([nbins, nbins])
    logemaxgrid = logerange[1:] * np.ones([nbins, nbins])
    logecentergrid = np.transpose(((logerange[:-1] + logerange[1:]) / 2.) * np.ones([nbins, nbins]))

    gauss_int = lambda p, x_min, x_max: .5 * (scipy.special.erf((x_max - p[1]) / np.sqrt(2. * p[2] ** 2.)) - scipy.special.erf((x_min - p[1]) / np.sqrt(2. * p[2] ** 2.)))

    rm = gauss_int([1., 10. ** logecentergrid, sigma(logecentergrid).reshape(logecentergrid.shape) * 10. ** logecentergrid ], 10. ** logemingrid, 10. ** logemaxgrid)
    #rm = gauss_int([1., 10. ** logecentergrid, .5], 10. ** logemingrid, 10. ** logemaxgrid)

# Create RM hdulist
hdulist = pf.np_to_rmf(rm,
                       (10. ** rm_erange_log).round(decimals=6),
                       (10. ** rm_ebounds_log).round(decimals=6),
                       1E-5,
                       telescope='CTASIM')

# Write RM to file
if write_output :
    hdulist.writeto(irf_file_name_base + '.rmf.fits')

#----------------------------------------------
# Read EA
h = irf_root_file.Get('EffectiveAreaEtrue') # ARF should be in true energy
if h == None :
    logging.info('ROOT file does not contain EffectiveAreaEtrue (EA vs E_true)')
    logging.info('Will use EffectiveArea (EA vs E_reco) for ARF')
    h = irf_root_file.Get('EffectiveArea')
ea = root_1dhist_to_array(h)[0]
# Read EA bin energy ranges
ea_erange_log = root_axis_to_array(h.GetXaxis())
# Re-sample EA to match RM
resample_ea_to_mrf = True
#resample_ea_to_mrf = False
if resample_ea_to_mrf:
        logging.info('Resampling effective area in log10(EA) vs log10(E) to match RM.')
        logea = np.log10(ea)
        logea[np.isnan(logea) + np.isinf(logea)] = 0.
        ea_spl = scipy.interpolate.UnivariateSpline((ea_erange_log[1:] + ea_erange_log[:-1]) / 2., logea, s=0, k=1)
        e = (rm_erange_log[1:] + rm_erange_log[:-1]) / 2.
        ea = 10. ** ea_spl(e)
        ea[ea < 1.] = 0.
        ea_erange_log = rm_erange_log

tbhdu = pf.np_to_arf(ea,
                     (10. ** ea_erange_log).round(decimals=6),
                     telescope='CTASIM')
# Write AR to file
if write_output :
    tbhdu.writeto(irf_file_name_base + '.arf.fits')

#----------------------------------------------
# Fit some distributions

# Broken power law fit function, normalized at break energy
bpl = lambda p,x : np.where(x < p[0], p[1] * (x / p[0]) ** -p[2],  p[1] * (x / p[0]) ** -p[3])
fitter = pf.ChisquareFitter(bpl)

h = irf_root_file.Get('BGRatePerSqDeg')
fit_th1(fitter, [3.2, 1E-5, 2., 1.], h, errscale=.2, range_=(.1, 100))
fitter.print_results()
bgrate_p1 = fitter.results[0]
fitx = np.linspace(-2., 2., 100.)

h = irf_root_file.Get('AngRes')
fit_th1(fitter, [1., .6, .5, .2], h, errscale=.1)
fitter.print_results()
angres68_p1 = fitter.results[0]

#----------------------------------------------
# Read extra information from response file

aux_tab = []
plt.figure(figsize=(10,8))

h = irf_root_file.Get('BGRate')
plt.subplot(331)
plot_th1(h,logy=1)
tbhdu = root_th1_to_fitstable(h, yunit='Hz', xunit='log(1/TeV)')
tbhdu.header.update('EXTNAME ', 'BGRATE', 'Name of this binary table extension')
aux_tab.append(tbhdu)

h = irf_root_file.Get('BGRatePerSqDeg')
plt.subplot(332)
plot_th1(h,logy=1)
plt.plot(fitx, bpl(bgrate_p1, 10. ** fitx))
plt.plot(fitx, bpl([9., 5E-4, 1.44, .49], 10. ** fitx))
tbhdu = root_th1_to_fitstable(h, yunit='Hz/deg^2', xunit='log(1/TeV)')
tbhdu.header.update('EXTNAME ', 'BGRATED', 'Name of this binary table extension')
aux_tab.append(tbhdu)

h = irf_root_file.Get('EffectiveArea')
plt.subplot(333)
plot_th1(h,logy=1)
tbhdu = root_th1_to_fitstable(h, yunit='m^2', xunit='log(1/TeV)')
tbhdu.header.update('EXTNAME ', 'EA', 'Name of this binary table extension')
aux_tab.append(tbhdu)

h = irf_root_file.Get('EffectiveArea80')
if h != None :
    plt.subplot(334)
    plot_th1(h, logy=True)
    tbhdu = root_th1_to_fitstable(h, yunit='m^2', xunit='log(1/TeV)')
    tbhdu.header.update('EXTNAME ', 'EA80', 'Name of this binary table extension')
    aux_tab.append(tbhdu)

h = irf_root_file.Get('EffectiveAreaEtrue')
if h != None :
    plt.subplot(335)
    plot_th1(h, logy=True)
    tbhdu = root_th1_to_fitstable(h, yunit='m^2', xunit='log(1/TeV)')
    tbhdu.header.update('EXTNAME ', 'EAETRUE', 'Name of this binary table extension')
    aux_tab.append(tbhdu)

h = irf_root_file.Get('AngRes')
plt.subplot(336)
plot_th1(h, logy=True)
plt.plot(fitx, bpl(angres68_p1, 10. ** fitx))
plt.plot(fitx, bpl([1.1, 5.5E-2, .42, .19], 10. ** fitx))
tbhdu = root_th1_to_fitstable(h, yunit='deg', xunit='log(1/TeV)')
tbhdu.header.update('EXTNAME ', 'ANGRES68', 'Name of this binary table extension')
aux_tab.append(tbhdu)

h = irf_root_file.Get('AngRes80')
plt.subplot(337)
plot_th1(h, logy=True)
tbhdu = root_th1_to_fitstable(h, yunit='deg', xunit='log(1/TeV)')
tbhdu.header.update('EXTNAME ', 'ANGRES80', 'Name of this binary table extension')
aux_tab.append(tbhdu)

h = irf_root_file.Get('ERes')
plt.subplot(339)
plot_th1(h)
tbhdu = root_th1_to_fitstable(h, xunit='log(1/TeV)')
tbhdu.header.update('EXTNAME ', 'ERES', 'Name of this binary table extension')
aux_tab.append(tbhdu)

plt.subplot(338)
#plt.set_cmap(plt.cm.Purples)
plt.set_cmap(plt.cm.jet)
#plt.imshow(np.log10(rm), origin='lower', extent=(rm_ebounds_log[0], rm_ebounds_log[-1], rm_erange_log[0], rm_erange_log[-1]))
plt.imshow(rm, origin='lower', extent=(rm_ebounds_log[0], rm_ebounds_log[-1], rm_erange_log[0], rm_erange_log[-1]))
plt.colorbar()
#plt.clim(-2., 1.)

plt.subplots_adjust(left=.08, bottom=.08, right=.97, top=.95, wspace=.3, hspace=.35)

# Create primary HDU and HDU list to be stored in the output file
hdu = pyfits.PrimaryHDU()
hdulist = pyfits.HDUList([hdu] + aux_tab)

# Write extra response data to file
if write_output :
    hdulist.writeto(irf_file_name_base + '.extra.fits')

#print tbhdu.header
#print tbhdu.data.field('BIN_LO')
#print tbhdu.data.field('BIN_HI')
#print tbhdu.data.field('VAL')
#print tbhdu.data.field('ERR')
# DEBUG
#t = np.transpose(root_2dhist_to_array(h))
#print 'XX', np.sum(rm, axis=0)
#print 'YY', np.sum(rm, axis=1)

#print 'XXX', np.sum(np.dot(np.ones(len(rm)), rm)), rm.shape, np.sum(np.sum(rm, axis=1))

#---------------------------------------------------------------------------
# Close CTA IRF root file

irf_root_file.Close()

#---------------------------------------------------------------------------
# DEBUG plots

#f = pyfits.open('SubarrayE_IFAE_50hours_20101102.rmf.fits')
#
#plt.set_cmap(plt.cm.Purples)
#
#plt.subplot(221)
#a,b,c,d = pf.rmf_to_np(f)
##print np.log10(b[1:]) - np.log10(b[:-1])
##print np.log10(c[1:]) - np.log10(c[:-1])
#extent=np.log10(np.array([c[0], c[-1], b[-1], b[0]]))
#plt.imshow((a - rm)[::-1], extent=extent)
#plt.colorbar()
#
#plt.subplot(222)
#
#plt.imshow(rm[::-1], extent=extent)
#plt.colorbar()
#
#plt.subplot(223)
#
#plt.imshow(a[::-1], extent=extent)
#plt.colorbar()
#
#plt.subplot(224)
#plt.hist(a - rm, bins=31)

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
