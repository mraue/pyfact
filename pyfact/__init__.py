"""
PyFACT
Python & FITS based Analysis for Cherenkov Telescopes

The module is splitted in several files according to the functionality for better maintance.

tools - general tools and helpers
fits - functions to deal with input/output in fits format
map - functions to deal with the creation of skymaps


tools
    class range :
    class chisquare_fitter :
    def get_li_ma_sign(non, noff, alpha) :
    def get_nice_time(t, sep='') :
    def circle_circle_intersection(R, r, d) :
    def unique_base_file_name(name, extension=None) :

fits
    def map_to_primaryhdu(map, rarange, decrange) :

map
    class sky_coord :
    class sky_circle :
    def get_cam_acc(camdist, rmax=4., nbins=0, exreg=None, fit=False) :
    def get_sky_mask_circle(r, bin_size) :
    def get_sky_mask_ring(rmin, rmax, bin_size) :
    def get_exclusion_region_map(map, rarange, decrange, exreg) :
    def oversample_sky_map(sky, mask, exmap=None) :
"""

__version__ = '0.0.1'
__author__  = 'M. Raue // martin.raue@desy.de'

# Import all functions/classes from the different files
from tools import *
from fits import *
from map import *
