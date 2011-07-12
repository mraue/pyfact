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
        
        :param float/array ra: Right ascention coordinate.
        :param float/array dec: Declination of the coordinate.
        """
        self.ra, self.dec = ra, dec

    def dist(self, c) :
        """
        Return the distance of the coordinates in degree following the haversine formula,
        see e.g. http://en.wikipedia.org/wiki/Great-circle_distance .

        :param sky_coord c:
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

        :param SkyCoord coord: Coordinates of the circle center (RA, Dec)
        :param float r: Radius of the circle (deg).
        """
        self.c, self.r = c, r

    def contains(self, c) :
        """
        Checks if the coordinate lies inside the circle.

        :param SkyCoord c:
        """
        return self.c.dist(c) <= self.r

    def intersects(self, sc) :
        """
        Checks if two sky circles overlap.

        :param SkyCircle sc:
        """
        return self.c.dist(sc.c) <= self.r + sc.r

#---------------------------------------------------------------------------
def get_cam_acc(camdist, rmax=4., nbins=0, exreg=None, fit=False, fitfunc=None, p0=None) :
    """
    Calculates the camera acceptance histogram from a given list with camera distances (event list).

    :param array camdist: Numpy array of camera distances (event list).
    :param float rmax: Maximum radius for the acceptance histogram (optional).
    :param nbins nbins: Number of bins for the acceptance histogram (optional; default = 0.1 deg).
    :param array exreg:
        Array of exclusion regions. Exclusion regions are given by an aray of size 2 (optional)
        [r, d] with r = radius, d = distance to camera center
    :param bool fit: Fit acceptance histogram (optional; default=False).
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
            ex_a += (pf.circle_circle_intersection(bins[1:], t * reg[0], t * reg[1])
                     - pf.circle_circle_intersection(bins[:-1], t * reg[0], t * reg[1]))
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
        m = n != 0
        fitter.fit_data(p0, r[m], n[m] / r_a[m] / (1. - ex_a[m]), nerr[m] / r_a[m] / (1. - ex_a[m]))
    return (n, bins, nerr, r, r_a, ex_a, fitter)

#---------------------------------------------------------------------------
def get_sky_mask_circle(r, bin_size) :
    """
    Returns a 2d numpy histogram with (2. * r / bin_size) bins per axis
    where a circle of radius has bins filled 1.s, all other bins are 0. .

    :param float r: Radius of the circle.
    :param float bin_size: Physical size of the bin, same units as rmin, rmax.
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

    :param float rmin: Inner radius of the ring.
    :param float rmax: Outer radius of the ring.
    :param float bin_size: Physical size of the bin, same units as rmin, rmax.
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

    :param 2d array map:
    :param array rarange:
    :param array decrange
    :param array exreg: array-type of SkyCircle
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

    :param 2d array sky:
    :param 2d array mask:
    :param 2d array exmap:
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
