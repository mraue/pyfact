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
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#===========================================================================
# Imports

import numpy as np
import pyfits

#===========================================================================
# Functions & classes

#---------------------------------------------------------------------------
def map_to_primaryhdu(map, rarange, decrange) :
    """
    Converts a 2d numpy array into a FITS primary HDU.

    Parameters
    ----------
    map : 2d array
        Skymap.
    rarange : array/tupel
        Tupel/Array with two entries giving the RA range of the map i.e. (ramin, ramax).
    decrange : array/tupel
        Tupel/Array with two entries giving the DEC range of the map i.e (decmin, decmax).

    Returns
    -------
    hdu : pyfits.PrimaryHDU
      FITS primary HDU containing the skymap.
    """
    decnbins, ranbins = map.shape

    decstep = (decrange[1] - decrange[0]) / float(decnbins)
    rastep = (rarange[1] - rarange[0]) / float(ranbins)

    hdu = pyfits.PrimaryHDU(map)
    hdr = hdu.header

    # Image definition
    hdr.update('CTYPE1', 'RA---CAR')
    hdr.update('CTYPE2', 'DEC--CAR')
    hdr.update('CUNIT1', 'deg')
    hdr.update('CUNIT2', 'deg')
    hdr.update('CRVAL1', rarange[0])
    hdr.update('CRVAL2', 0.) # Must be zero for the lines to rectalinear according to Calabretta (2002)
    hdr.update('CRPIX1', 1.)
    hdr.update('CRPIX2', - decrange[0] / decstep) # Pixel outside of the image at DEC = 0.
    hdr.update('CDELT1', rastep)
    hdr.update('CDELT2', decstep)
    hdr.update('RADESYS', 'FK5')
    hdr.update('BUNIT', 'count')

    # Extra data
    hdr.update('TELESCOP', 'HESS')
    hdr.update('OBJECT', 'TEST')
    hdr.update('AUTHOR', 'PyFACT - pfmakeskymap')

    # DEBUG
    #print hdr

    return hdu

#===========================================================================

