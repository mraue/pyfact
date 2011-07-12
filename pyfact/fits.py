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

    :param 2d array map: 2d numpy array containing a skymap.
    :param array rarange: Tupel/Array with two entries giving the RA range of the map i.e. (ramin, ramax).
    :param array decrange: Tupel/Array with two entries giving the DEC range of the map i.e (decmin, decmax).
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

