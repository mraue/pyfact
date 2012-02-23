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

"""
PyFACT
Python & FITS based Analysis for Cherenkov Telescopes

The module is splitted in several files according to the functionality for better maintance.

tools - general tools and helpers
fits - functions to deal with input/output in fits format
map - functions to deal with the creation of skymaps

tools
    class Range :
    class ChisquareFitter :
    def get_li_ma_sign(non, noff, alpha) :
    def get_nice_time(t, sep='') :
    def circle_circle_intersection(R, r, d) :
    def unique_base_file_name(name, extension=None) :
    def date_to_mjd(d) :

fits
    def map_to_primaryhdu(map, rarange, decrange) :
    def np_to_arf(ea, erange, telescope='DUMMY', instrument='DUMMY', filter='NONE') :
    def np_to_rmf(rm, erange, ebounds, minprob,
                  telescope='DUMMY', instrument='DUMMY', filter='NONE') :
    def rmf_to_np(hdulist) :
    def np_to_pha(channel, counts, exposure, dstart, dstop, dbase=None,
                  stat_err=None, quality=None, syserr=None,
                  obj_ra=0., obj_dec=0., obj_name='DUMMY', creator='DUMMY',
                  version='v0.0.0', telescope='DUMMY', instrument='DUMMY', filter_='NONE') :

map
    class SkyCoord:
    class SkyCircle:
    def get_cam_acc(camdist, rmax=4., nbins=0, exreg=None, fit=False) :
    def get_sky_mask_circle(r, bin_size) :
    def get_sky_mask_ring(rmin, rmax, bin_size) :
    def get_exclusion_region_map(map, rarange, decrange, exreg) :
    def oversample_sky_map(sky, mask, exmap=None) :
"""

__version__ = '0.0.4'
__author__  = 'M. Raue // martin.raue@desy.de'

# Import all functions/classes from the different files
from tools import *
from fits import *
from map import *
