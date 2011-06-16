#===========================================================================
# Imports

import sys, time, logging, os, datetime, math
import numpy as np
import scipy.optimize
import scipy.special
import pyfits
import pyfact as pf

#===========================================================================
# Functions & classes

#---------------------------------------------------------------------------
class sky_coord :

    """
    Sky coordinate in RA and Dec. All units should be degree.

    In the current implementation it should also work with arrays, though one has to be careful in dist.

    Parameters
    ----------
    ra: float / array-type
        Right ascention coordinate.
    dec: float / array-type
        Declination of the coordinate.
    """

    def __init__(self, ra, dec) :
        self.ra, self.dec = ra, dec

    def dist(self, c) :
        """
        Return the distance of the coordinates in degree following the haversine formula,
        see e.g. http://en.wikipedia.org/wiki/Great-circle_distance .

        Parameters
        ----------
        c: sky_coord
        """
        return 2. * np.arcsin(np.sqrt(np.sin((self.dec - c.dec) / 360. * np.pi) ** 2.
                                      + np.cos(self.dec / 180. * np.pi) * np.cos(c.dec / 180. * np.pi)\
                                          * np.sin((self.ra - c.ra) / 360. * np.pi) ** 2.)) / np.pi * 180.


#---------------------------------------------------------------------------
class sky_circle :
    """
    A circle on the sky.

    Parameters
    ----------
    coord: sky_coord
        Coordinates of the circle center (RA, Dec).
    r: float
        Radius of the circle (deg).
    """

    def __init__(self, c, r) :
        self.c, self.r = c, r

    def contains(self, c) :
        """
        Checks if the coordinate lies inside the circle.

        Parameters
        ----------
        c : sky_coord
        """
        return self.c.dist(c) <= self.r

    def intersects(self, sc) :
        """
        Checks if two sky circles overlap.

        Parameters
        ----------
        sc: sky_circle
        """
        return self.c.dist(sc.c) <= self.r + sc.r

#---------------------------------------------------------------------------
def get_cam_acc(camdist, rmax=4., nbins=0, exreg=None, fit=False) :
    """
    Calculates the camera acceptance histogram from a given list with camera distances (event list).

    Parameters
    ----------
    camdist: array-type
        Numpy array of camera distances (event list).
    rmax: float, optional
        Maximum radius for the acceptance histogram.
    nbins: int, optional
        Number of bins for the acceptance histogram. Default is one bin per 0.1 deg.
    exreg: array-type, optional
        Array of exclusion regions. Exclusion regions are given by an aray of size 2
        [r, d] with r = radius, d = distance to camera center
    fit: bool, optional
        Fit acceptance histogram.
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
        fitfunc = lambda p, x: p[0] * x ** 0. * (1. + (x / p[1]) ** p[2]) ** ((0. + p[3]) / p[2])
        p0 = [n[0] / r_a[0], 1.5, 3., -5.] # Initial guess for the parameters
        fitter = pf.chisquare_fitter(fitfunc)
        fitter.fit_data(p0, r, n / r_a / (1. - ex_a), nerr / r_a / (1. - ex_a))
    return (n, bins, nerr, r, r_a, ex_a, fitter)

#---------------------------------------------------------------------------
def get_sky_mask_circle(r, bin_size) :
    """
    Returns a 2d numpy histogram with (2. * r / bin_size) bins per axis
    where a circle of radius has bins filled 1.s, all other bins are 0. .

    Parameters
    ----------
    r: float
        Radius of the circle.
    bin_size: float
        Physical size of the bin, same units as rmin, rmax.
    """
    nbins = int(np.ceil(2 * r / bin_size))
    sky_mask = np.zeros((nbins, nbins))
    for x in range(nbins) :
        for y in range(nbins) :
            d = np.sqrt((float(x) * bin_size + bin_size / 2. - r) ** 2.
                        + (float(y) * bin_size + bin_size / 2. - r) ** 2.)
            if d < r :
                sky_mask[x, y] = 1.
    return sky_mask

#---------------------------------------------------------------------------
def get_sky_mask_ring(rmin, rmax, bin_size) :
    """
    Returns a 2d numpy histogram with (2. * rmax / bin_size) bins per axis
    filled with a ring with inner radius rmin and outer radius rmax of 1.,
    all other bins are 0..

    Parameters
    ----------
    rmin: float
        Inner radius of the ring.
    rmax: float
        Outer radius of the ring.
    bin_size: float
        Physical size of the bin, same units as rmin, rmax.
    """
    nbins = int(np.ceil(2 * rmax / bin_size))
    sky_mask = np.zeros((nbins, nbins))
    for x in range(nbins) :
        for y in range(nbins) :
            d = np.sqrt((float(x) * bin_size + bin_size / 2. - rmax) ** 2.
                        + (float(y) * bin_size + bin_size / 2. - rmax) ** 2.)
            if d < rmax and d > rmin :
                sky_mask[x, y] = 1.
    return sky_mask

#---------------------------------------------------------------------------
def get_exclusion_region_map(map, rarange, decrange, exreg) :
    """
    Creates a map (2d numpy histogram) with all bins inside of exclusion regions set to 0. (others 1.).

    Dec is on the 1st axis (x), RA is on the 2nd (y).

    Parameters:
    -----------
    map: array-type (2d numpy)
    rarange: array-type
    decrange: array-type
    exreg: array-type of sky_circle
    """
    xnbins, ynbins = map.shape
    xstep, ystep = (decrange[1] - decrange[0]) / float(xnbins), (rarange[1] - rarange[0]) / float(ynbins)
    sky_mask = np.ones((xnbins, ynbins))
    for x, xval in enumerate(np.linspace(decrange[0] + xstep / 2., decrange[1] - xstep / 2., xnbins)) :
        for y, yval in enumerate(np.linspace(rarange[0] + ystep / 2., rarange[1] - ystep / 2., ynbins)) :
            for reg in exreg :
                if reg.contains(sky_coord(yval, xval)) :
                    sky_mask[x, y] = 0.
    return sky_mask

#---------------------------------------------------------------------------
def oversample_sky_map(sky, mask, exmap=None) :
    """
    Oversamples a 2d numpy histogram with a given mask.

    Parameters
    ----------
    sky : array like (2d numpy array)
    mask : array like (2d numpy array)
    exmap : array like (2d numpy array), optional
    """
    sky_nx, sky_ny =  sky.shape[0], sky.shape[1]
    mask_nx, mask_ny = mask.shape[0], mask.shape[1]
    mask_centerx, mask_centery = (mask_nx - 1) / 2, (mask_ny - 1) / 2

    # new oversampled sky plot
    sky_overs = np.zeros((sky_nx, sky_ny))

    # 2d hist keeping the number of bins used (alpha)
    sky_alpha = np.zeros((sky_nx, sky_ny))

    for x in range(sky_nx) :
        for y in range(sky_ny) :
            # Create new empty base mask of size of the sky hist + mask
            mask_full = np.zeros([sky_nx + mask_nx + 3, sky_ny + mask_ny + 3])
            # Add mask at the right position
            mask_full[x:x + mask_nx, y: y + mask_ny] = mask
            # Mask sky hist, calculate sum, and fill new sky hist
            mask_full = mask_full[mask_centerx:mask_centerx + sky_nx, mask_centery: mask_centery + sky_ny]
            if exmap != None :
                mask_full *= exmap
            sky_overs[x, y] = np.sum(mask_full * sky)
            sky_alpha[x, y] = np.sum(mask_full)

    return (sky_overs, sky_alpha)

#===========================================================================
