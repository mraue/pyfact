#===========================================================================
# Copyright (c) 2011, Martin Raue
# All rights reserved.
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

#===========================================================================
# Imports

import sys, time, logging, os, datetime, math
import numpy as np
import scipy.optimize
import scipy.special
import scipy.ndimage
import pyfits
import pyfact as pf

#===========================================================================
# Functions & classes

#---------------------------------------------------------------------------
class SkyCoord:
    """Sky coordinate in RA and Dec. All units should be degree."""
    
    def __init__(self, ra, dec) :
        """
        Sky coordinate in RA and Dec. All units should be degree.
        
        In the current implementation it should also work with arrays, though one has to be careful in dist.

        Parameters
        ----------
        ra : float/array
            Right ascention coordinate.
        dec : float/array
            Declination of the coordinate.
        """
        self.ra, self.dec = ra, dec

    def dist(self, c) :
        """
        Return the distance of the coordinates in degree following the haversine formula,
        see e.g. http://en.wikipedia.org/wiki/Great-circle_distance.

        c : SkyCoord

        Returns
        -------
        distance : float
            Return the distance of the coordinates in degree following the haversine formula,
            see e.g. http://en.wikipedia.org/wiki/Great-circle_distance.
        """
        return 2. * np.arcsin(np.sqrt(np.sin((self.dec - c.dec) / 360. * np.pi) ** 2.
                                      + np.cos(self.dec / 180. * np.pi) * np.cos(c.dec / 180. * np.pi)\
                                          * np.sin((self.ra - c.ra) / 360. * np.pi) ** 2.)) / np.pi * 180.

#---------------------------------------------------------------------------
class SkyCircle:
    """A circle on the sky."""
    
    def __init__(self, c, r) :
        """
        A circle on the sky.

        Parameters
        ----------
        coord : SkyCoord
            Coordinates of the circle center (RA, Dec)
        r : float
            Radius of the circle (deg).
        """
        self.c, self.r = c, r

    def contains(self, c) :
        """
        Checks if the coordinate lies inside the circle.

        c : SkyCoord

        Returns
        -------
        contains : bool
            True if c lies in the SkyCircle.
        """
        return self.c.dist(c) <= self.r

    def intersects(self, sc) :
        """
        Checks if two sky circles overlap.

        Parameters
        ----------
        sc : SkyCircle
        """
        return self.c.dist(sc.c) <= self.r + sc.r

#---------------------------------------------------------------------------
def get_cam_acc(camdist, rmax=4., nbins=0, exreg=None, fit=False, fitfunc=None, p0=None) :
    """
    Calculates the camera acceptance histogram from a given list with camera distances (event list).

    Parameters
    ----------
    camdist : array
        Numpy array of camera distances (event list).
    rmax : float, optional
        Maximum radius for the acceptance histogram.
    nbins : int, optional
        Number of bins for the acceptance histogram (default = 0.1 deg).
    exreg : array, optional
        Array of exclusion regions. Exclusion regions are given by an aray of size 2
        [r, d] with r = radius, d = distance to camera center
    fit : bool, optional
        Fit acceptance histogram (default=False).
    """
    if not nbins :
        nbins = int(rmax / .1)
    # Create camera distance histogram
    n, bins = np.histogram(camdist, bins=nbins, range=[0., rmax])
    nerr = np.sqrt(n)
    # Bin center array
    r = (bins[1:] + bins[:-1]) / 2.
    # Bin area (ring) array
    r_a = (bins[1:] ** 2. - bins[:-1] ** 2.) * np.pi
    # Deal with exclusion regions
    ex_a = None
    if exreg :
        ex_a = np.zeros(len(r))
        t =  np.ones(len(r))
        for reg in exreg :
            ex_a += (pf.circle_circle_intersection_a(bins[1:], t * reg[0], t * reg[1])
                     - pf.circle_circle_intersection_a(bins[:-1], t * reg[0], t * reg[1]))
        ex_a /= r_a
    # Fit the data
    fitter = None
    if fit :
        #fitfunc = lambda p, x: p[0] * x ** p[1] * (1. + (x / p[2]) ** p[3]) ** ((p[1] + p[4]) / p[3])
        if not fitfunc :
            fitfunc = lambda p, x: p[0] * x ** 0. * (1. + (x / p[1]) ** p[2]) ** ((0. + p[3]) / p[2])
        if not p0 :
            p0 = [n[0] / r_a[0], 1.5, 3., -5.] # Initial guess for the parameters
        fitter = pf.ChisquareFitter(fitfunc)
        m = (n > 0.) * (nerr > 0.) * (r_a != 0.) * ((1. - ex_a) != 0.)
        if np.sum(m) <= len(p0) :
            logging.error('Could not fit camera acceptance (dof={0}, bins={1})'.format(len(p0), np.sum(m)))
        else :
            # ok, this _should_ be improved !!!
            x, y, yerr =  r[m], n[m] / r_a[m] / (1. - ex_a[m]) , nerr[m] / r_a[m] / (1. - ex_a[m])
            m = np.isfinite(x) * np.isfinite(y) * np.isfinite(yerr) * (yerr != 0.)
            if np.sum(m) <= len(p0) :
                logging.error('Could not fit camera acceptance (dof={0}, bins={1})'.format(len(p0), np.sum(m)))
            else :
                fitter.fit_data(p0, x[m], y[m], yerr[m])
    return (n, bins, nerr, r, r_a, ex_a, fitter)

#---------------------------------------------------------------------------
def get_sky_mask_circle(r, bin_size) :
    """
    Returns a 2d numpy histogram with (2. * r / bin_size) bins per axis
    where a circle of radius has bins filled 1.s, all other bins are 0.

    Parameters
    ----------
    r : float
        Radius of the circle.
    bin_size : float
        Physical size of the bin, same units as rmin, rmax.
    """
    nbins = int(np.ceil(2. * r / bin_size))
    sky_x = np.ones((nbins, nbins)) *  np.linspace(bin_size / 2., 2. * r - bin_size / 2., nbins)
    sky_y = np.transpose(sky_x)
    sky_mask = np.where(np.sqrt((sky_x - r) ** 2. + (sky_y - r) ** 2.) < r, 1., 0.)
    return sky_mask

#---------------------------------------------------------------------------
def get_sky_mask_ring(rmin, rmax, bin_size) :
    """
    Returns a 2d numpy histogram with (2. * rmax / bin_size) bins per axis
    filled with a ring with inner radius rmin and outer radius rmax of 1.,
    all other bins are 0..

    Parameters
    ----------
    rmin : float
        Inner radius of the ring.
    rmax : float
        Outer radius of the ring.
    bin_size : float
        Physical size of the bin, same units as rmin, rmax.
    """
    nbins = int(np.ceil(2. * rmax / bin_size))
    sky_x = np.ones((nbins, nbins)) *  np.linspace(bin_size / 2., 2. * rmax - bin_size / 2., nbins)
    sky_y = np.transpose(sky_x)
    sky_mask = np.where((np.sqrt((sky_x - rmax) ** 2. + (sky_y - rmax) ** 2.) < rmax) * (np.sqrt((sky_x - rmax) ** 2. + (sky_y - rmax) ** 2.) > rmin), 1., 0.)
    return sky_mask


#---------------------------------------------------------------------------
def get_exclusion_region_map(map, rarange, decrange, exreg) :
    """
    Creates a map (2d numpy histogram) with all bins inside of exclusion regions set to 0. (others 1.).

    Dec is on the 1st axis (x), RA is on the 2nd (y).

    Parameters
    ----------
    map : 2d array
    rarange : array
    decrange : array
    exreg : array-type of SkyCircle
    """
    xnbins, ynbins = map.shape
    xstep, ystep = (decrange[1] - decrange[0]) / float(xnbins), (rarange[1] - rarange[0]) / float(ynbins)
    sky_mask = np.ones((xnbins, ynbins))
    for x, xval in enumerate(np.linspace(decrange[0] + xstep / 2., decrange[1] - xstep / 2., xnbins)) :
        for y, yval in enumerate(np.linspace(rarange[0] + ystep / 2., rarange[1] - ystep / 2., ynbins)) :
            for reg in exreg :
                if reg.contains(SkyCoord(yval, xval)) :
                    sky_mask[x, y] = 0.
    return sky_mask

#---------------------------------------------------------------------------
def oversample_sky_map(sky, mask, exmap=None) :
    """
    Oversamples a 2d numpy histogram with a given mask.

    Parameters
    ----------
    sky : 2d array 
    mask : 2d array
    exmap : 2d array
    """
    sky_nx, sky_ny =  sky.shape[0], sky.shape[1]
    mask_nx, mask_ny = mask.shape[0], mask.shape[1]
    mask_centerx, mask_centery = (mask_nx - 1) / 2, (mask_ny - 1) / 2

    # new oversampled sky plot
    sky_overs = np.zeros((sky_nx, sky_ny))

    # 2d hist keeping the number of bins used (alpha)
    sky_alpha = np.ones((sky_nx, sky_ny))

    sky_base = np.ones((sky_nx, sky_ny))
    if exmap != None :
        sky *= exmap
        sky_base *= exmap

    scipy.ndimage.convolve(sky, mask, sky_overs, mode='constant')
    scipy.ndimage.convolve(sky_base, mask, sky_alpha, mode='constant')

    return (sky_overs, sky_alpha)

#===========================================================================
