#===========================================================================
# Copyright (c) 2011-2012, the PyFACT developers
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
# DISCLAIMED. IN NO EVENT SHALL THE PYFACT DEVELOPERS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#===========================================================================
# Imports

import datetime

import numpy as np
import pyfits

#===========================================================================
# Functions & classes

#---------------------------------------------------------------------------
def map_to_primaryhdu(map, rarange, decrange, telescope='DUMMY', object_='DUMMY', author='DUMMY') :
    """
    Converts a 2d numpy array into a FITS primary HDU (image).

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
    return map_to_hdu(map, rarange, decrange, primary=True, telescope=telescope, object_=object_, author=author)

#---------------------------------------------------------------------------
def map_to_hdu(map, rarange, decrange, primary=False, telescope='DUMMY', object_='DUMMY', author='DUMMY') :
    """
    Converts a 2d numpy array into a FITS primary HDU (image).

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

    hdu = None
    if primary :
        hdu = pyfits.PrimaryHDU(map)
    else :
        hdu = pyfits.ImageHDU(map)
    hdr = hdu.header

    # Image definition
    hdr.update('CTYPE1', 'RA---CAR')
    hdr.update('CTYPE2', 'DEC--CAR')
    hdr.update('CUNIT1', 'deg')
    hdr.update('CUNIT2', 'deg')
    hdr.update('CRVAL1', rarange[0])
    hdr.update('CRVAL2', 0.) # Must be zero for the lines to be rectalinear according to Calabretta (2002)
    hdr.update('CRPIX1', .5)
    hdr.update('CRPIX2', - decrange[0] / decstep + .5) # Pixel outside of the image at DEC = 0.
    hdr.update('CDELT1', rastep)
    hdr.update('CDELT2', decstep)
    hdr.update('RADESYS', 'FK5')
    hdr.update('BUNIT', 'count')

    # Extra data
    hdr.update('TELESCOP', telescope)
    hdr.update('OBJECT', object_)
    hdr.update('AUTHOR', author)

    # DEBUG
    #print hdr

    return hdu

#---------------------------------------------------------------------------
def np_to_arf(ea, erange, telescope='DUMMY', instrument='DUMMY', filter='NONE') :
    """
    Create ARF FITS table extension from numpy arrays.

    Parameters
    ----------
    ea : 1D float numpy array
       Effective area [m^2].
    erange : 1D float numpy array
       Bin limits E_true [TeV].

    Notes
    -----
    For more info on the ARF FITS file format see:
    http://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/summary/cal_gen_92_002_summary.html
    
    Recommended units for ARF tables are keV and cm^2, but TeV and m^2 are chosen here
    as the more natural units for IACTs
    """

    tbhdu = pyfits.new_table(
        [pyfits.Column(name='ENERG_LO',
                      format='1E',
                      array=erange[:-1],
                      unit='TeV'),
         pyfits.Column(name='ENERG_HI',
                      format='1E',
                      array=erange[1:],
                      unit='TeV'),
         pyfits.Column(name='SPECRESP',
                      format='1E',
                      array=ea,
                      unit='m^2')
         ]
        )

    # Write FITS extension header
    tbhdu.header.update('EXTNAME ', 'SPECRESP', 'Name of this binary table extension')
    tbhdu.header.update('TELESCOP', telescope, 'Mission/satellite name')
    tbhdu.header.update('INSTRUME', instrument, 'Instrument/detector')
    tbhdu.header.update('FILTER  ', filter, 'Filter information')
    tbhdu.header.update('HDUCLASS', 'OGIP', 'Organisation devising file format')
    tbhdu.header.update('HDUCLAS1', 'RESPONSE', 'File relates to response of instrument')
    tbhdu.header.update('HDUCLAS2', 'SPECRESP', 'Effective area data is stored')
    tbhdu.header.update('HDUVERS ', '1.1.0', 'Version of file format')

    # Optional
    #tbhdu.header.update('PHAFILE', '', 'PHA file for which ARF was produced')

    # Obsolet ARF headers, included for the benefit of old software
    tbhdu.header.update('ARFVERSN', '1992a', 'Obsolete')
    tbhdu.header.update('HDUVERS1', '1.0.0', 'Obsolete')
    tbhdu.header.update('HDUVERS2', '1.1.0', 'Obsolete')

    return tbhdu

#---------------------------------------------------------------------------
def arf_to_np(arf) :
    """
    Reads an arf file into numpy arrays

    Parameters
    ----------
    arf : arf table HDU

    Returns
    -------
    ea : numpy 1D array
       effective area
    erange : numpy 1D array
       bin energy ranges

    Notes
    -----
    For more info on the ARF FITS file format see:
    http://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/summary/cal_gen_92_002_summary.html
    
    Recommended units for ARF tables are keV and cm^2, but TeV and m^2 are chosen here
    as the more natural units for IACTs
    """
    return (arf.data.field('SPECRESP'), np.hstack([arf.data.field('ENERG_LO'), arf.data.field('ENERG_HI')[-1]]))

#---------------------------------------------------------------------------
def np_to_rmf(rm, erange, ebounds, minprob,
              telescope='DUMMY', instrument='DUMMY', filter='NONE') :
    """
    Converts a 2D numpy array to an RMF FITS file.

    Parameters
    ----------
    rm : 2D float numpy array
       Energy distribution matrix (probability density) E_true vs E_reco.
    erange : 1D float numpy array
       Bin limits E_true [TeV].
    ebounds : 1D float numpy array
       Bin limits E_reco [TeV].
    minprob : float
        Minimal probability density to be stored in the RMF.
    telescope, instrument, filter : string
        Keyword information for the FITS header.

    Returns
    -------
    hdulist : FITS hdulist

    Notes
    -----
    For more info on the RMF FITS file format see:
    http://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/summary/cal_gen_92_002_summary.html
    """
    
    # Intialize the arrays to be used to construct the RM extension
    n_rows = len(rm)
    energy_lo = np.zeros(n_rows) # Low energy bounds
    energy_hi = np.zeros(n_rows) # High energy bounds
    n_grp = [] # Number of channel subsets
    f_chan = [] # First channels in each subset
    n_chan = [] # Number of channels in each subset
    matrix = [] # Matrix elements

    # Loop over the matrix and fill the arrays
    for i, r in enumerate(rm) :
        energy_lo[i] = erange[i]
        energy_hi[i] = erange[i + 1]
        # Create mask for all matrix row values above the minimal probability
        m = r > minprob
        # Intialize variables & arrays for the row
        n_grp_row, n_chan_row_c = 0, 0
        f_chan_row, n_chan_row, matrix_row = [], [], []
        new_subset = True
        # Loop over row entries and fill arrays appropriately
        for j, v in enumerate(r) :
            if m[j] :
                if new_subset :
                    n_grp_row += 1
                    f_chan_row.append(j)
                    new_subset = False
                matrix_row.append(v)
                n_chan_row_c += 1
            else :
                if not new_subset :
                    n_chan_row.append(n_chan_row_c)
                    n_chan_row_c = 0
                    new_subset = True
        if not new_subset :
            n_chan_row.append(n_chan_row_c)
        n_grp.append(n_grp_row)
        f_chan.append(f_chan_row)
        n_chan.append(n_chan_row)
        matrix.append(matrix_row)

    # Create RMF FITS table extension from data
    tbhdu = pyfits.new_table(
        [pyfits.Column(name='ENERG_LO',
                      format='1E',
                      array=energy_lo,
                      unit='TeV'),
         pyfits.Column(name='ENERG_HI',
                      format='1E',
                      array=energy_hi,
                      unit='TeV'),
         pyfits.Column(name='N_GRP',
                       format='1I',
                       array=n_grp),
         pyfits.Column(name='F_CHAN',
                       format='PI()',
                       array=f_chan),
         pyfits.Column(name='N_CHAN',
                       format='PI()',
                       array=n_chan),
         pyfits.Column(name='MATRIX',
                       format='PE(()',
                       array=matrix)
         ]
        )

    # Write FITS extension header

    chan_min, chan_max, chan_n = 0, rm.shape[1] - 1, rm.shape[1]

    tbhdu.header.update('EXTNAME ', 'MATRIX', 'name of this binary table extension')
    tbhdu.header.update('TLMIN4  ', chan_min, 'First legal channel number')
    tbhdu.header.update('TLMAX4  ', chan_max, 'Highest legal channel number')
    tbhdu.header.update('TELESCOP', telescope, 'Mission/satellite name')
    tbhdu.header.update('INSTRUME', instrument, 'Instrument/detector')
    tbhdu.header.update('FILTER  ', filter, 'Filter information')
    tbhdu.header.update('CHANTYPE', 'PHA', 'Type of channels (PHA, PI etc)')
    tbhdu.header.update('DETCHANS', chan_n, 'Total number of detector PHA channels')
    tbhdu.header.update('LO_THRES', minprob, 'Lower probability density threshold for matrix')
    tbhdu.header.update('HDUCLASS', 'OGIP', 'Organisation devising file format')
    tbhdu.header.update('HDUCLAS1', 'RESPONSE', 'File relates to response of instrument')
    tbhdu.header.update('HDUCLAS2', 'RSP_MATRIX', 'Keyword information for Caltools Software.')
    tbhdu.header.update('HDUVERS ', '1.3.0', 'Version of file format')
    tbhdu.header.update('HDUCLAS3', 'DETECTOR', 'Keyword information for Caltools Software.')
    tbhdu.header.update('CCNM0001', 'MATRIX', 'Keyword information for Caltools Software.')
    tbhdu.header.update('CCLS0001', 'CPF', 'Keyword information for Caltools Software.')
    tbhdu.header.update('CDTP0001', 'DATA', 'Keyword information for Caltools Software.')

    # UTC date when this calibration should be first used (yyy-mm-dd)
    tbhdu.header.update('CVSD0001', '2011-01-01 ', 'Keyword information for Caltools Software.')

    # UTC time on the dat when this calibration should be first used (hh:mm:ss)
    tbhdu.header.update('CVST0001', '00:00:00', 'Keyword information for Caltools Software.')

    # String giving a brief summary of this data set
    tbhdu.header.update('CDES0001', r'dummy data', 'Keyword information for Caltools Software.')

    # Optional, but maybe useful (taken from the example in the RMF/ARF document)
    tbhdu.header.update('CBD10001', 'CHAN({0}- {1})'.format(chan_min, chan_max), 'Keyword information for Caltools Software.')
    tbhdu.header.update('CBD20001', 'ENER({0}-{1})TeV'.format(erange[0], erange[-1]), 'Keyword information for Caltools Software.')

    # Obsolet RMF headers, included for the benefit of old software
    tbhdu.header.update('RMFVERSN', '1992a', 'Obsolete')
    tbhdu.header.update('HDUVERS1', '1.1.0', 'Obsolete')
    tbhdu.header.update('HDUVERS2', '1.2.0', 'Obsolete')

    # Create EBOUNDS FITS table extension from data
    tbhdu2 = pyfits.new_table(
        [pyfits.Column(name='CHANNEL',
                       format='1I',
                       array=np.arange(len(ebounds) - 1)),
         pyfits.Column(name='E_MIN',
                      format='1E',
                      array=ebounds[:-1],
                      unit='TeV'),
         pyfits.Column(name='E_MAX',
                      format='1E',
                      array=ebounds[1:],
                      unit='TeV')
         ]
        )

    chan_min, chan_max, chan_n = 0, rm.shape[0] - 1, rm.shape[0]
    
    tbhdu2.header.update('EXTNAME ', 'EBOUNDS', 'Name of this binary table extension')
    tbhdu2.header.update('TELESCOP', telescope, 'Mission/satellite name')
    tbhdu2.header.update('INSTRUME', instrument, 'Instrument/detector')
    tbhdu2.header.update('FILTER  ', filter, 'Filter information')    
    tbhdu2.header.update('CHANTYPE', 'PHA', 'Type of channels (PHA, PI etc)')
    tbhdu2.header.update('DETCHANS', chan_n, 'Total number of detector PHA channels')
    tbhdu2.header.update('TLMIN1  ', chan_min, 'First legal channel number')
    tbhdu2.header.update('TLMAX1  ', chan_max, 'Highest legal channel number')
    tbhdu2.header.update('HDUCLASS', 'OGIP', 'Organisation devising file format')
    tbhdu2.header.update('HDUCLAS1', 'RESPONSE', 'File relates to response of instrument')
    tbhdu2.header.update('HDUCLAS2', 'EBOUNDS', 'This is an EBOUNDS extension')
    tbhdu2.header.update('HDUVERS ', '1.2.0', 'Version of file format')
    tbhdu2.header.update('HDUCLAS3', 'DETECTOR', 'Keyword information for Caltools Software.')
    tbhdu2.header.update('CCNM0001', 'EBOUNDS', 'Keyword information for Caltools Software.')
    tbhdu2.header.update('CCLS0001', 'CPF', 'Keyword information for Caltools Software.')
    tbhdu2.header.update('CDTP0001', 'DATA', 'Keyword information for Caltools Software.')

    # UTC date when this calibration should be first used (yyy-mm-dd)
    tbhdu2.header.update('CVSD0001', '2011-01-01 ', 'Keyword information for Caltools Software.')

    # UTC time on the dat when this calibration should be first used (hh:mm:ss)
    tbhdu2.header.update('CVST0001', '00:00:00', 'Keyword information for Caltools Software.')

    # Optional - name of the PHA file for which this file was produced
    #tbhdu2.header.update('PHAFILE', '', 'Keyword information for Caltools Software.')

    # String giving a brief summary of this data set
    tbhdu2.header.update('CDES0001', r'dummy description', 'Keyword information for Caltools Software.')

    # Obsolet EBOUNDS headers, included for the benefit of old software
    tbhdu2.header.update('RMFVERSN', '1992a', 'Obsolete')
    tbhdu2.header.update('HDUVERS1', '1.0.0', 'Obsolete')
    tbhdu2.header.update('HDUVERS2', '1.1.0', 'Obsolete')

    # Create primary HDU and HDU list to be stored in the output file
    hdu = pyfits.PrimaryHDU()
    hdulist = pyfits.HDUList([hdu, tbhdu, tbhdu2])

    return hdulist

#---------------------------------------------------------------------------
def rmf_to_np(hdulist) :
    """
    Converts an RMF FITS hdulist into numpy arrays

    Parameters
    ----------
    hdulist : FITS hdulist
        Primary extension should be the MATRIX, secondary extension the EBOUNDS

    Returns
    ----------
    rm : 2D float numpy array
        Energy distribution matrix (probability density) E_true vs E_reco.
    erange : 1D float numpy array
        Bin limits E_true [TeV].
    ebounds : 1D float numpy array
        Bin limits E_reco [TeV].
    minprob : float
        Minimal probability density to be stored in the RMF.

    Notes
    -----
    For more info on the RMF FITS file format see:
    http://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/summary/cal_gen_92_002_summary.html
    """

    tbhdu = hdulist[1]
    assert tbhdu.header['EXTNAME'] == 'MATRIX', 'First RMF extension need to be the MATRIX extension'
    d = tbhdu.data

    erange = np.hstack([d.field('ENERG_LO'), d.field('ENERG_HI')[-1]])

    rm = np.zeros([len(d), tbhdu.header['DETCHANS']])

    for i, l in enumerate(d) :
        if l.field('N_GRP') :
            m_start = 0
            for k in range(l.field('N_GRP')) :
                rm[i, l.field('F_CHAN')[k] : l.field('F_CHAN')[k] + l.field('N_CHAN')[k]] \
                = l.field('MATRIX')[m_start:m_start + l.field('N_CHAN')[k]]
                m_start += l.field('N_CHAN')[k]
    minprob = tbhdu.header['LO_THRES']

    tbhdu = hdulist[2]
    assert tbhdu.header['EXTNAME'] == 'EBOUNDS', 'Second RMF extension should be the EBOUNDS extension'
    d = tbhdu.data
    ebounds = np.hstack([d.field('E_MIN'), d.field('E_MAX')[-1]])

    return (rm, erange, ebounds, minprob)

#---------------------------------------------------------------------------
def np_to_pha(channel, counts, exposure, dstart, dstop, dbase=None, stat_err=None, quality=None, syserr=None,
              obj_ra=0., obj_dec=0., obj_name='DUMMY', creator='DUMMY',
              version='v0.0.0', telescope='DUMMY', instrument='DUMMY', filter_='NONE') :
    """
    Create PHA FITS table extension from numpy arrays.

    Parameters
    ----------
    dat : numpy 1D array float
        Binned spectral data [counts]
    dat_err : numpy 1D array float
        Statistical errors associated with dat [counts]
    chan : numpu 1D array int
        Corresponding channel numbers for dat
    exposure : float
        Exposure [s]
    dstart : datetime
        Observation start time.
    dstop : datetime
        Observation stop time.
    dbase : datetime
        Base date used for TSTART/TSTOP.
    quality : numpy 1D array integer
        Quality flags for the channels (optional)
    syserr : numpy 1D array float
        Fractional systematic error for the channel (optional)
    obj_ra/obj_dec : float
        Object RA/DEC J2000 [deg]

    Notes
    -----
    For more info on the PHA FITS file format see:
    http://heasarc.gsfc.nasa.gov/docs/heasarc/ofwg/docs/summary/ogip_92_007_summary.html
    """
    # Create PHA FITS table extension from data
    cols = [pyfits.Column(name='CHANNEL',
                          format='I',
                          array=channel,
                          unit='channel'),
            pyfits.Column(name='COUNTS',
                          format='1E',
                          array=counts,
                          unit='count')
            ]

    if stat_err is not None :
        cols.append(pyfits.Column(name='STAT_ERR',
                                  format='1E',
                                  array=stat_err,
                                  unit='count'))
    
    if syserr is not None :
        cols.append(pyfits.Column(name='SYS_ERR',
                                  format='E',
                                  array=syserr))

    if quality is not None :
        cols.append(pyfits.Column(name='QUALITY',
                                  format='I',
                                  array=quality))

    tbhdu = pyfits.new_table(cols)

    tbhdu.header.update('EXTNAME ', 'SPECTRUM'          , 'name of this binary table extension')
    tbhdu.header.update('TELESCOP', telescope, 'Telescope (mission) name')
    tbhdu.header.update('INSTRUME', instrument, 'Instrument name')
    tbhdu.header.update('FILTER  ', filter_, 'Instrument filter in use')
    tbhdu.header.update('EXPOSURE', exposure, 'Exposure time')

    tbhdu.header.update('BACKFILE', 'none', 'Background FITS file')
    tbhdu.header.update('CORRFILE', 'none', 'Correlation FITS file')
    tbhdu.header.update('RESPFILE', 'none', 'Redistribution matrix file (RMF)')
    tbhdu.header.update('ANCRFILE', 'none', 'Ancillary response file (ARF)')

    tbhdu.header.update('HDUCLASS', 'OGIP', 'Format conforms to OGIP/GSFC spectral standards')
    tbhdu.header.update('HDUCLAS1', 'SPECTRUM', 'Extension contains a spectrum')
    tbhdu.header.update('HDUVERS ', '1.2.1', 'Version number of the format')

    poisserr = False
    if stat_err is None :
        poisserr = True
    tbhdu.header.update('POISSERR', poisserr, 'Are Poisson Distribution errors assumed')

    tbhdu.header.update('CHANTYPE', 'PHA', 'Channels assigned by detector electronics')
    tbhdu.header.update('DETCHANS', len(channel), 'Total number of detector channels available')
    tbhdu.header.update('TLMIN1  ', channel[0], 'Lowest Legal channel number')
    tbhdu.header.update('TLMAX1  ', channel[-1], 'Highest Legal channel number')

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

    tbhdu.header.update('HDUCLAS2', 'NET', 'Extension contains a bkgr substr. spec.')
    tbhdu.header.update('HDUCLAS3', 'COUNT', 'Extension contains counts')
    tbhdu.header.update('HDUCLAS4', 'TYPE:I', 'Single PHA file contained')
    tbhdu.header.update('HDUVERS1', '1.2.1', 'Obsolete - included for backwards compatibility')

    if syserr is None :
        tbhdu.header.update('SYS_ERR ', 0, 'No systematic error was specified')

    tbhdu.header.update('GROUPING', 0, 'No grouping data has been specified')

    if quality is None :
        tbhdu.header.update('QUALITY ', 0, 'No data quality information specified')

    tbhdu.header.update('AREASCAL', 1., 'Nominal effective area')
    tbhdu.header.update('BACKSCAL', 1., 'Background scale factor')
    tbhdu.header.update('CORRSCAL', 0., 'Correlation scale factor')

    tbhdu.header.update('FILENAME', 'several', 'Spectrum was produced from more than one file')
    tbhdu.header.update('ORIGIN  ', 'dummy', 'origin of fits file')
    tbhdu.header.update('DATE    ', datetime.datetime.today().strftime('%Y-%m-%d'), 'FITS file creation date (yyyy-mm-dd)')
    tbhdu.header.update('PHAVERSN', '1992a', 'OGIP memo number for file format')

    if dbase :
        import pyfact as pf
        tbhdu.header.update('TIMESYS ', 'MJD', 'The time system is MJD')
        tbhdu.header.update('TIMEUNIT', 's', 'unit for TSTARTI/F and TSTOPI/F, TIMEZERO')
        tbhdu.header.update('MJDREF  ', pf.date_to_mjd(dbase), '{0:.2f}'.format(dbase.year + (dbase.month - 1.) / 12.
                                                                                + (dbase.day - 1.) / 31.))
        tbhdu.header.update('TSTART  ', (pf.date_to_mjd(dstart) - pf.date_to_mjd(dbase)) * 86400., 'Observation start time [s]')
        tbhdu.header.update('TSTOP   ', (pf.date_to_mjd(dstop) - pf.date_to_mjd(dbase)) * 86400., 'Observation stop time [s]')

    return tbhdu

#===========================================================================
