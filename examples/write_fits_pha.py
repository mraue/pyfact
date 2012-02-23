#===========================================================================
# Copyright (c) 2011-2012, Martin Raue
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

import logging
import datetime

import numpy as np
import pyfits

#===========================================================================
# Functions & objects

#----------------------------------------------------------------------
def date_to_mjd(d):
    """
    Returns MJD from a datetime or date object.

    Notes
    -----
    http://en.wikipedia.org/wiki/Julian_Day
    http://docs.python.org/library/datetime.html

    """
    a = (14 - d.month)//12
    y = d.year + 4800 - a
    m = d.month + 12*a - 3
    mjd =  d.day + ((153*m + 2)//5) + 365*y + y//4 - y//100 + y//400 - 32045 - 2400000.5
    if type(d) == datetime.datetime :
        mjd += d.hour / 24. + d.minute / 1440. + d.second / 86400.0 + d.microsecond / 86400.E3
    return mjd
    
#===========================================================================
# Main

#----------------------------------------------------------------------
# SETUP

# Setup fancy logging for output
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

#----------------------------------------------------------------------
# INPUT

# Create dummy data

dat = np.array([0., 0., 5., 10., 20., 15., 10., 5., 2.5, 1., 1., 0.]) # data [counts]
dat_err = np.sqrt(dat) # data error [counts]
chan = (np.arange(len(dat))) # channel numbers

telescope, instrument, filter_ = 'dummy', 'dummy', 'dummy'
exposure = 3600. # Exposure time [s]
obj_name, obj_ra, obj_dec = 'dummy', 0., 0. # ra/dec J2000 [deg]

creator, version = 'dummy', 'v0.0.0' # program that produced this file & program version

dstart = datetime.datetime(2011, 1, 1, 0, 0) # date/time of start of the first observation
dstop = datetime.datetime(2011, 1, 31, 0, 0) # date/time of end of the last observation
dbase = datetime.datetime(2011, 1, 1)

#----------------------------------------------------------------------
# CREATE FITS OUTPUT

# Create PHA FITS table extension from data
tbhdu = pyfits.new_table(
    [pyfits.Column(name='CHANNEL',
                  format='I',
                  array=chan,
                  unit='channel'),
     pyfits.Column(name='COUNTS',
                  format='1E',
                  array=dat,
                  unit='count'),
     pyfits.Column(name='STAT_ERR',
                  format='1E',
                  array=dat_err,
                  unit='count')
     ]
    )

tbhdu.header.update('EXTNAME ', 'SPECTRUM'          , 'name of this binary table extension')
tbhdu.header.update('TELESCOP', telescope, 'Telescope (mission) name')
tbhdu.header.update('INSTRUME', instrument, 'Instrument name')
tbhdu.header.update('FILTER  ', filter_, 'Instrument filter in use')
tbhdu.header.update('EXPOSURE', exposure, 'Exposure time')

tbhdu.header.update('BACKFILE', 'none', 'Background FITS file for this object')
tbhdu.header.update('CORRFILE', 'none', 'Correlation FITS file for this object')
tbhdu.header.update('RESPFILE', 'none', 'Redistribution matrix file (RMF)')
tbhdu.header.update('ANCRFILE', 'none', 'Ancillary response file (ARF)')

tbhdu.header.update('HDUCLASS', 'OGIP', 'Format conforms to OGIP/GSFC spectral standards')
tbhdu.header.update('HDUCLAS1', 'SPECTRUM', 'Extension contains a spectrum')
tbhdu.header.update('HDUVERS ', '1.2.1', 'Version number of the format')

tbhdu.header.update('POISSERR', False, 'Are Poisson Distribution errors assumed')

tbhdu.header.update('CHANTYPE', 'PHA', 'Channels assigned by detector electronics')
tbhdu.header.update('DETCHANS', len(chan), 'Total number of detector channels available')
tbhdu.header.update('TLMIN2  ', chan[0], 'Lowest Legal channel number')
tbhdu.header.update('TLMAX2  ', chan[-1], 'Highest Legal channel number')

tbhdu.header.update('XFLT0001', 'none', 'XSPEC selection filter description')
tbhdu.header.update('OBJECT  ', obj_name, 'OBJECT from the FIRST input file')
tbhdu.header.update('RA-OBJ  ', obj_ra, 'RA of First input object')
tbhdu.header.update('DEC-OBJ ', obj_dec, 'DEC of First input object')
tbhdu.header.update('EQUINOX ', 2000.00, 'Equinox of the FIRST object')
tbhdu.header.update('RADECSYS', 'FK5', 'Co-ordinate frame used for equinox')
tbhdu.header.update('DATE-OBS', dstart.strftime('%Y-%m-%d'), 'EARLIEST observation date of files')
tbhdu.header.update('TIME-OBS', dstart.strftime('%H:%M:%S'), 'EARLIEST time of all input files')
tbhdu.header.update('DATE-END', dstop.strftime('%Y-%m-%d'), 'LATEST observation date of files')
tbhdu.header.update('TIME-END', dstop.strftime('%H:%M:%S'), 'LATEST time of all input files')

tbhdu.header.update('CREATOR ', '{0} {1}'.format(creator, version), 'Program name that produced this file')

tbhdu.header.update('HDUCLAS2', 'NET', 'Extension contains a background substracted spectrum')
tbhdu.header.update('HDUCLAS3', 'COUNT', 'Extension contains counts')
tbhdu.header.update('HDUCLAS4', 'TYPE:I', 'Single PHA file contained')
tbhdu.header.update('HDUVERS1', '1.2.1', 'Obsolete - included for backwards compatibility')

tbhdu.header.update('SYS_ERR ', 0, 'No systematic error was specified')
tbhdu.header.update('GROUPING', 0, 'No grouping data has been specified')
tbhdu.header.update('QUALITY ', 0, 'No data quality information specified')
tbhdu.header.update('AREASCAL', 1., 'Nominal effective area')
tbhdu.header.update('BACKSCAL', 1., 'Background scale factor')
tbhdu.header.update('CORRSCAL', 0., 'Correlation scale factor')

tbhdu.header.update('FILENAME', 'several', 'Spectrum was produced from more than one file')
tbhdu.header.update('ORIGIN  ', 'dummy', 'origin of fits file')
tbhdu.header.update('DATE    ', datetime.datetime.today().strftime('%Y-%m-%d'), 'FITS file creation date (yyyy-mm-dd)')
tbhdu.header.update('PHAVERSN', '1992a', 'OGIP memo number for file format')

tbhdu.header.update('TIMESYS ', 'MJD', 'The time system is MJD')
tbhdu.header.update('TIMEUNIT', 's', 'unit for TSTARTI/F and TSTOPI/F, TIMEZERO')
tbhdu.header.update('MJDREF  ', date_to_mjd(dbase), '{0:.2f}'.format(dbase.year + (dbase.month - 1.) / 12. + (dbase.day - 1.) / 31.))
tbhdu.header.update('TSTART  ', (date_to_mjd(dstart) - date_to_mjd(dbase)) * 86400., 'Observation start time [s]')
tbhdu.header.update('TSTOP   ', (date_to_mjd(dstop) - date_to_mjd(dbase)) * 86400., 'Observation stop time [s]')

print tbhdu.header

#----------------------------------------------------------------------
# PRIMARY HEADER

#hdu = pyfits.PrimaryHDU()

#hdu.header.update('CONTENT ', 'SPECTRUM'           , 'light spectrum file')
#hdu.header.update('ORIGIN  ', 'dummy'          , 'origin of fits file')
#hdu.header.update('CREATOR ', '{0} {1}'.format(creator, version), 'Program name that produced this file')
#hdu.header.update('DATE    ', datetime.datetime.today().strftime('%Y-%m-%d'), 'FITS file creation date (yyyy-mm-dd)')
#hdu.header.update('TELESCOP', telescope, 'Telescope (mission) name')
#hdu.header.update('INSTRUME', instrument, 'Instrument used for observation')
#hdu.header.update('MJDREF  ', date_to_mjd(d_base), '{0:.2f}'.format(d_base.year + (d_base.month - 1.) / 12. + (d_base.day - 1.) / 31.))
#hdu.header.update('TSTART  ', (date_to_mjd(dstart) - date_to_mjd(d_base)) * 86400., 'Observation start time [s]')
#hdu.header.update('TSTOP   ', (date_to_mjd(dstop) - date_to_mjd(d_base)) * 86400., 'Observation stop time [s]')
##hdu.header.update('MJDREF  ',  4.9352000696574076E+04, '1993.0')
##hdu.header.update('TSTART  ',  7.32588800000000E+06, 'Observation start time')
##hdu.header.update('TSTOP   ',  7.32601600000000E+06, 'Observation stop time')
#hdu.header.update('OBJECT  ', obj_name, 'OBJECT from the FIRST input file')
#hdu.header.update('RA_OBJ  ', obj_ra, 'RA of First input object')
#hdu.header.update('DEC_OBJ ', obj_dec, 'DEC of First input object')
#hdu.header.update('EQUINOX ', 2000.00, 'Equinox of the FIRST object')
#hdu.header.update('RADECSYS', 'FK5     ', 'Co-ordinate frame used for equinox')
#hdu.header.update('DATE-OBS', dstart.strftime('%Y-%m-%d'), 'EARLIEST observation date of files')
#hdu.header.update('TIME-OBS', dstart.strftime('%H:%M:%S'), 'EARLIEST time of all input files')
#hdu.header.update('DATE-END', dstop.strftime('%Y-%m-%d'), 'LATEST observation date of files')
#hdu.header.update('TIME-END', dstop.strftime('%H:%M:%S'), 'LATEST time of all input files')

#----------------------------------------------------------------------
# WRITE FITS OUTPUT

tbhdu.writeto('pha_example.fits')

#----------------------------------------------------------------------
#----------------------------------------------------------------------
