#===========================================================================
# Imports

import sys, time, logging, os, datetime, math
import numpy as np
import scipy.optimize
import scipy.special
import pyfits

#===========================================================================
# Functions & classes

#---------------------------------------------------------------------------
class range :
    """
    Simple class to hold a range.

    Parameters
    ----------
    min : float
        Lower bound of the range.
    max: float
        Upper bound of the range.
    """

    def __init__(self, min, max) :
        if min <= max :
            raise Exception('min is less then max {0} <= {1}'.format(min, max))
        else :
            self.min, self.max = min, max

    def inrange(self, v) :
        """
        Check if value lies in the range.
        """
        return min <= v <= max

#---------------------------------------------------------------------------
class chisquare_fitter :
    """
    Convenience class to perform Chi^2 fits.
    """

    fitfunc = None # Fitfunction
    results = None # Fit results from scipy.optimize.leastsq()
    chi_arr = None # Array with the final chi values
    chi2    = None # Summed Chi^2
    dof     = None # Degrees of freedom
    prob    = None # Probability of the fit

    def __init__(self, fitfunc) :
        self.fitfunc = fitfunc

    def fit_data(self, p0, x, y, y_err) :
        self.results = scipy.optimize.leastsq(self.chi_func, p0, args=(x, y, y_err), full_output=True)
        if self.results[4] :
            self.chi_arr = self.chi_func(self.results[0], x, y, y_err)
            self.chi2 = np.sum(np.power(self.chi_arr, 2.))
            self.dof = len(x) - len(p0)
            #self.prob = scipy.special.gammainc(.5 * self.dof, .5 * self.chi2) / scipy.special.gamma(.5 * self.dof)
            self.prob = 1. - scipy.special.gammainc(.5 * self.dof, .5 * self.chi2)
        return self.results[4]

    def chi_func(self, p, x, y, err):
        return (self.fitfunc(p, x) - y) / err # Distance to the target function

#---------------------------------------------------------------------------
def get_li_ma_sign(non, noff, alpha) :
    """
    Returns the statistical significance following Li & Ma (1983) Equ. 13.
    Works with numpy arrays.
    """
    sign = np.ones(non.shape)
    sign = (non < noff * alpha) * sign * -2. + sign
    return sign * np.sqrt(2. * (non * np.log((1 + alpha) / alpha * non / (non + noff))
                                + noff * np.log( (1 + alpha) * noff / (non + noff))))   

#---------------------------------------------------------------------------
def get_nice_time(t, sep='') :
    """
    Returns the time in a formatted string <x>d<x>h<x>m<x>s
    with variable separator string between the units.
    
    Parameters
    ----------
    t: float
        Time in seconds.
    sep: string, optional
        Separator string to be used between units.
    """
    s = ''
    if t > 86399. :
        d = math.floor(t / 86400.)
        t -= d * 86400.
        s += '{0}d'.format(int(d)) + sep
    if t > 3599. :
        h = math.floor(t / 3600.)
        t -= h * 3600.
        s += '{0}h'.format(int(h)) + sep
    if t > 59. :
        m = math.floor(t / 60.)
        t -= m * 60.
        s += '{0}m'.format(int(m)) + sep
    if t > 0. :
        s += '{0:.2f}s'.format(t)
    return s.strip()

#---------------------------------------------------------------------------
def circle_circle_intersection(R, r, d) :
    """
    Returns the intersecting area between two circles with radius R and r and distance d.

    Works with floats and numpy arrays, but the current implementation is not very elegant.

    Parameters
    ----------
    R: float / array
        Radius of the first circle.
    r: float / array
        Radius of the second circle
    d: float / array
        Distance of the two circle (center to center)
    """

    # Define a few useful lambda functions
    X = lambda R, r, d: (d * d + r * r - R * R) / (2. * d * r)
    Y = lambda R, r, d: (d * d + R * R - r * r) / (2. * d * R)
    Z = lambda R, r, d: (-d + r + R) * (d + r - R) * (d - r + R) * (d + r + R)

    # In case we have floats or arrays with only one entry
    if type(R) == float or len(R) == 1 :
        # If circle 1 lies in circle 2 or vice versa return full circle area
        if R > d + r :
            return np.pi * r ** 2.
        elif r > d + R :
            return np.pi * R ** 2.
        elif R + r > d :
            return (r ** 2.) * np.arccos(X(R, r, d)) + (R ** 2.) * np.arccos(Y(R, r, d)) - .5 * np.sqrt(Z(R, r, d))
        else :
            return 0.
    
    result = np.zeros(len(R))
    mask1 = R > d + r
    if mask1.any() :
        result[mask1] = np.pi * r[mask1] ** 2.
    mask2 = r > d + R
    if mask2.any() :
        result[mask2] = np.pi * r[mask2] ** 2.
    mask = (R + r > d) * np.invert(mask1) * np.invert(mask2)
    if mask.any() :
        r, R, d = r[mask], R[mask], d[mask]
        result[mask] = (r ** 2.) * np.arccos(X(R, r, d)) + (R ** 2.) * np.arccos(Y(R, r, d)) - .5 * np.sqrt(Z(R, r, d));
    return result

#---------------------------------------------------------------------------
def unique_base_file_name(name, extension=None) :
    """
    Checks if a given file already exists. If yes, creates a new unique filename.

    Parameters
    ----------
    """
    def filename_exists(name, extension) :
        exists = False
        if extension :
            try :
                len(extension)
                for ext in extension :
                    if os.path.exists(name + ext) :
                        exists = True
                        break
            except :
                if os.path.exists(name + extension) :
                    exists = True
                else :
                    if os.path.exists(name) :
                        exists = True
        return exists
    if filename_exists(name, extension) :
        name += datetime.datetime.now().strftime('_%Y%m%d-%H%M%S')
        if filename_exists(name, extension) :
            import random
            name += '_' + str(int(random.random() * 10000))
    return name

#===========================================================================
