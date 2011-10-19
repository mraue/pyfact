#! /usr/bin/env python

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

import sys
import time
import logging
import os
import gc

import numpy as np
import pyfits
import scipy.interpolate
# Check if we have matplotlib for graphical output
has_matplotlib = True
try :
    import matplotlib.pyplot as plt
    import matplotlib.patches
except :
    has_matplotlib = False

# Add script parent directory to python search path to get access to the pyfact package
sys.path.append(os.path.abspath(sys.path[0].rsplit('/', 1)[0]))
import pyfact as pf

#===========================================================================
# Main
def create_spectrum(input_file_name,
                    analysis_position=None,
                    analysis_radius=.125,
                    write_output=False,
                    do_graphical_output=True,
                    loglevel='INFO') :
    # Time it!
    t_1 = time.clock()

    # Configure logging
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)
    logging.basicConfig(level=numeric_level,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    # Welcome user, print out basic information on package versions
    logging.info('This is {0}'.format(os.path.split(__file__)[1]))
    logging.info('We are running pyfact v{0}, numpy v{1}, and pyfits v{2}'.format(
        pf.__version__, np.__version__, pyfits.__version__
        ))

    #---------------------------------------------------------------------------
    # Loop over the file list, calculate quantities, & fill histograms

    # Exclusion radius [this should be generalized in future versions]
    rexdeg = .3

    # Intialize some variables
    objra, objdec, pntra, pntdec = None, None, None, None
    if analysis_position :
        objra, objdec = eval(analysis_position)
        logging.info('Analysis position: RA {0}, Dec {1}'.format(objra, objdec))
    else :
        logging.info('No analysis position given => will use object position from first file')

    logging.info('Analysis radius: {0} deg'.format(analysis_radius))

    theta2_hist_max, theta2_hist_nbins = .5 ** 2., 50
    theta2_on_hist, theta2_off_hist, theta2_offcor_hist = np.zeros(theta2_hist_nbins), np.zeros(theta2_hist_nbins), np.zeros(theta2_hist_nbins)
    non, noff, noffcor = 0., 0., 0.
    sky_ex_reg = None
    firstloop = True

    spec_nbins, spec_emin, spec_emax = 40, -2., 2.
    spec_on_hist, spec_off_hist = np.zeros(spec_nbins), np.zeros(spec_nbins)
    spec_hist_ebounds = np.linspace(spec_emin, spec_emax, spec_nbins + 1)

    # Read in input file, can be individual fits or bankfile
    logging.info('Opening input file(s) ..')

    # This list will hold the individual file names as strings
    file_list = []

    # Check if we are dealing with a single file or a bankfile
    # and create/read in the file list accordingly
    try :
        f = pyfits.open(input_file_name)
        f.close()
        file_list = [input_file_name]
    except :
        # We are dealing with a bankfile
        f = open(input_file_name)
        for l in f:
            l = l.strip(' \t\n')
            if l :
                file_list.append(l)
        f.close()

    # Shortcuts for commonly used functions
    cci_f, cci_a = pf.circle_circle_intersection_f, pf.circle_circle_intersection_a

    for file_name in file_list :

        logging.info('Processing file {0}'.format(file_name))

        # Open fits file
        hdulist = pyfits.open(file_name)

        # Print file info
        #hdulist.info()

        # Access header of second extension
        ex1hdr = hdulist[1].header

        # Print header of the first extension as ascardlist
        #print ex1hdr.ascardlist()

        # Access data of first extension
        tbdata = hdulist[1].data # assuming the first extension is a table

        # Print table columns
        #hdulist[1].columns.info()

        #---------------------------------------------------------------------------
        # Calculate some useful quantities and add them to the table

        if firstloop :
            # If skymap center is not set, set it to the target position of the first run
            if objra == None or objdec == None :
                objra, objdec = ex1hdr['RA_OBJ'], ex1hdr['DEC_OBJ']
                logging.info('Analysis position from header: RA {0}, Dec {1}'.format(objra, objdec))

        pntra, pntdec = ex1hdr['RA_PNT'], ex1hdr['DEC_PNT']
        obj_cam_dist = pf.SkyCoord(objra, objdec).dist(pf.SkyCoord(pntra, pntdec))

        logging.debug('Object camera distance = {0}'.format(obj_cam_dist))
        
        # Distance from the camera (FOV) center
        camdist = np.sqrt(tbdata.field('DETX    ') ** 2. + tbdata.field('DETY    ') ** 2.)

        # Distance from analysis position
        thetadist = pf.SkyCoord(objra, objdec).dist(pf.SkyCoord(tbdata.field('RA      '), tbdata.field('DEC    ')))

        ## cos(DEC)
        #cosdec = np.cos(tbdata.field('DEC     ') * np.pi / 180.)
        #cosdec_col = pyfits.Column(name='XCOSDEC', format='1E', array=cosdec)

        # Add new columns to the table
        coldefs_new = pyfits.ColDefs(
            [pyfits.Column(name='XCAMDIST',format='1E', unit='deg', array=camdist),
             pyfits.Column(name='XTHETA',format='1E', unit='deg', array=thetadist)
             ]
            )
        newtable = pyfits.new_table(hdulist[1].columns + coldefs_new)

        # Print new table columns
        #newtable.columns.info()

        # New table data
        tbdata = newtable.data

        #---------------------------------------------------------------------------
        # Select events

        # Select events with at least one tel above the required image amplitude/size (here iamin p.e.)
        # This needs to be changed for the new TELEVENT table scheme
        #iamin = 80.
        #iamask = (tbdata.field('HIL_TEL_SIZE')[:,0] > iamin) \
        #    + (tbdata.field('HIL_TEL_SIZE')[:,1] > iamin) \
        #    + (tbdata.field('HIL_TEL_SIZE')[:,2] > iamin) \
        #    + (tbdata.field('HIL_TEL_SIZE')[:,3] > iamin)

        # Select events between emin & emax TeV
        emin, emax = .01, 1000.
        emask = (tbdata.field('ENERGY  ') > emin) \
            * (tbdata.field('ENERGY  ') < emax)

        # Only use events with < 4 deg camera distance
        camdmask = tbdata.field('XCAMDIST') < 4.

        # Combine cuts for photons
        phomask = (tbdata.field('HIL_MSW ') < 1.1) * emask * camdmask
        hadmask = (tbdata.field('HIL_MSW ') > 1.3) * (tbdata.field('HIL_MSW ') < 10.) * emask * camdmask
        
        # Most important cut for the acceptance calculation: exclude source region
        exmask = np.invert(np.sqrt(((tbdata.field('RA      ') - objra) / np.cos(objdec * np.pi / 180.)) ** 2.
                                   + (tbdata.field('DEC     ') - objdec) ** 2.) < rexdeg)

        photbdata = tbdata[phomask]
        #hadtbdata = tbdata[hadmask * exmask]

        on_run = photbdata[photbdata.field('XTHETA') < analysis_radius]
        off_run = photbdata[((photbdata.field('XCAMDIST') < obj_cam_dist + analysis_radius)
                             * (photbdata.field('XCAMDIST') > obj_cam_dist - analysis_radius)
                             * np.invert(photbdata.field('XTHETA') < rexdeg))]

        spec_on_hist += np.histogram(np.log10(on_run.field('ENERGY')), bins=spec_nbins, range=(spec_emin, spec_emax))[0]
        
        non_run, noff_run = len(on_run), len(off_run)
        
        alpha_run = analysis_radius**2. / ((obj_cam_dist + analysis_radius) ** 2.
                                           - (obj_cam_dist - analysis_radius) ** 2.
                                           - cci_f(obj_cam_dist + analysis_radius, rexdeg, obj_cam_dist) / np.pi
                                           + cci_f(obj_cam_dist - analysis_radius, rexdeg, obj_cam_dist) / np.pi)

        logging.debug('{0} {1} {2} {3}'.format((obj_cam_dist + analysis_radius) ** 2.,
                                               (obj_cam_dist - analysis_radius) ** 2.,
                                               cci_f(obj_cam_dist + analysis_radius, rexdeg, obj_cam_dist) / np.pi,
                                               cci_f(obj_cam_dist - analysis_radius, rexdeg, obj_cam_dist) / np.pi
                                               )
                      )

        def print_stats(non, noff, alpha, pre='') :
            logging.debug(pre + 'N_ON = {0}, N_OFF = {1}, ALPHA = {2:.4f}'.format(non, noff, alpha))
            logging.debug(pre + 'EXCESS = {0:.2f}, SIGN = {1:.2f}'.format(non - alpha * noff, pf.get_li_ma_sign(non, noff, alpha)))

        non += non_run
        noff += noff_run
        noffcor += alpha_run * noff_run

        print_stats(non_run, noff_run, alpha_run, 'Run: ')
        print_stats(non, noff, noffcor/noff, 'All: ')
        
        theta2_on_hist += np.histogram(photbdata.field('XTHETA') ** 2., bins=theta2_hist_nbins, range=(0., theta2_hist_max))[0]

        theta2_off_run_hist, theta2_off_run_hist_edges = np.histogram(np.fabs((photbdata[np.invert(photbdata.field('XTHETA') < rexdeg)].field('XCAMDIST') - obj_cam_dist)** 2.), bins=theta2_hist_nbins, range=(0., theta2_hist_max))

        theta2_off_hist += theta2_off_run_hist

        h_edges_r = np.sqrt(theta2_off_run_hist_edges)

        a_tmp = (
            cci_a(obj_cam_dist + h_edges_r,
                  np.ones(theta2_hist_nbins + 1) * rexdeg,
                  np.ones(theta2_hist_nbins + 1) * obj_cam_dist) / np.pi
            + cci_a(obj_cam_dist -  h_edges_r,
                    np.ones(theta2_hist_nbins + 1) * rexdeg,
                    np.ones(theta2_hist_nbins + 1) * obj_cam_dist) / np.pi
            )
        
        theta2_off_hist_alpha = (
            (theta2_off_run_hist_edges[1:] - theta2_off_run_hist_edges[:-1])
            / (4. * obj_cam_dist * (h_edges_r[1:] - h_edges_r[:-1])
               - (a_tmp[1:] - a_tmp[:-1])
               )
            )
        #logging.debug('theta2_off_hist_alpha = {0}'.format( theta2_off_hist_alpha))
        theta2_offcor_hist += theta2_off_run_hist * theta2_off_hist_alpha
        
        firstloop = False

    #---------------------------------------------------------------------------
    # Write results to file

    if write_output :
        logging.info('Writing result to file ..')
        logging.info('The output files can be found in {0}'.format(os.getcwd()))

    #---------------------------------------------------------------------------
    # Plot results

    if has_matplotlib and do_graphical_output :

        import matplotlib
        logging.info('Plotting results (matplotlib v{0})'.format(matplotlib.__version__))

        def set_title_and_axlabel(label) :
            plt.xlabel('RA (deg)')
            plt.ylabel('Dec (deg)')
            plt.title(label, fontsize='medium')

        plt.figure(1)
        x = np.linspace(0., theta2_hist_max, theta2_hist_nbins + 1)
        x = (x[1:] + x[:-1]) / 2.
        plt.errorbar(x, theta2_on_hist, xerr=(theta2_hist_max / (2. *  theta2_hist_nbins)), yerr=np.sqrt(theta2_on_hist),
                     fmt='o', ms=3.5, label=r'N$_{ON}$', capsize=0.)
        plt.errorbar(x, theta2_offcor_hist, xerr=(theta2_hist_max / (2. *  theta2_hist_nbins)),
                     yerr=np.sqrt(theta2_off_hist) * theta2_offcor_hist / theta2_off_hist,
                     fmt='+', ms=3.5, label=r'N$_{OFF} \times \alpha$', capsize=0.)
        plt.axvline(analysis_radius ** 2., ls='--', label=r'$\theta^2$ cut')
        plt.xlabel(r'$\theta^2$ (deg$^2$)')
        plt.ylabel(r'N')
        plt.legend(numpoints=1)

        plt.figure(2)
        ax = plt.subplot(111)
        plt.errorbar((spec_hist_ebounds[1:] + spec_hist_ebounds[:-1]) / 2., spec_on_hist,
                     xerr=(spec_hist_ebounds[1] - spec_hist_ebounds[0]) / 2.,
                     yerr=np.sqrt(spec_on_hist), fmt='o')

        ax.set_yscale('log')

    #----------------------------------------
    # Time it!
    t_2 = time.clock()
    logging.info('Execution took {0}'.format(pf.get_nice_time(t_2 - t_1)))

    logging.info('Thank you for choosing {0}. Have a great day!'.format(os.path.split(__file__)[1]))

    #----------------------------------------
    plt.show()

#===========================================================================
# Main function
if __name__ == '__main__':
    # We should switch to argparse soon (python v2.7++)
    # http://docs.python.org/library/argparse.html#module-argparse
    import optparse
    parser = optparse.OptionParser(
        usage='%prog [options] FILE\nFILE can either be an indiviual .fits/.fits.gz file or a batch file.',
        description='Creates spectra from VHE event lists in FITS format.'
    )
    parser.add_option(
        '-p','--analysis-position',
        dest='analysis_position',
        type='str',
        default=None,
        help='Analysis position in RA and Dec (J2000) in degree. Format: \'(RA, Dec)\', including the quotation marks. If no center is given, the source position from the first input file is used.'
    )
    parser.add_option(
        '-r','--analysis-radius',
        dest='analysis_radius',
        type='float',
        default=.125,
        help='Aperture for the analysis in degree [default: %default].'
    )
    parser.add_option(
        '-w','--write-output',
        dest='write_output',
        action='store_true',
        default=False,
        help='Write results to FITS files in current directory [default: %default]'
    )
    parser.add_option(
        '--no-graphical-output',
        dest='graphical_output',
        action='store_false',
        default=True,
        help='Switch off graphical output.'
    )
    parser.add_option(
        '-l','--log-level',
        dest='loglevel',
        default='INFO',
        help='Amount of logging e.g. DEBUG, INFO, WARNING, ERROR [default: %default]'
    )

    options, args = parser.parse_args()

    if len(args) == 1 :
        create_spectrum(
            input_file_name=args[0],
            analysis_position=options.analysis_position,
            analysis_radius=options.analysis_radius,
            write_output=options.write_output,
            do_graphical_output=options.graphical_output,
            loglevel=options.loglevel
            )
    else :
        parser.print_help()

#===========================================================================
