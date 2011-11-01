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
def create_sky_map(input_file_name,
                   skymap_size=5.,
                   skymap_bin_size=0.05,
                   r_overs=.125,
                   ring_bg_radii=None,
                   template_background=True,
                   skymap_center=None,
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

    # Skymap definition
    #skymap_size, skymap_bin_size = 6., 0.05
    rexdeg = .25

    # Intialize some variables
    objra, objdec, pntra, pntdec = None, None, None, None
    if skymap_center :
        objra, objdec = eval(skymap_center)
        logging.info('Skymap center: RA {0}, Dec {1}'.format(objra, objdec))

    ring_bg_r_min, ring_bg_r_max = .3, .7
    if ring_bg_radii :
        ring_bg_r_min, ring_bg_r_max = eval(ring_bg_radii)

    if r_overs > ring_bg_r_min :
        logging.warning('Oversampling radius is larger than the inner radius chosen for the ring BG: {0} > {1}'.format(r_overs, ring_bg_r_min))

    logging.info('Skymap size         : {0} deg'.format(skymap_size))
    logging.info('Skymap bin size     : {0} deg'.format(skymap_bin_size))
    logging.info('Oversampling radius : {0} deg'.format(r_overs))
    logging.info('Ring BG radius      : {0} - {1} deg'.format(ring_bg_r_min, ring_bg_r_max))

    skymap_nbins, sky_dec_min, sky_dec_max, objcosdec, sky_ra_min, sky_ra_max = 0, 0., 0., 0., 0., 0.
    sky_hist, acc_hist, extent = None, None, None
    tpl_had_hist, tpl_acc_hist = None, None
    sky_ex_reg = None

    firstloop = True

    exposure = 0.

    # Read in input file, can be individual fits or bankfile
    logging.info('Opening input file ..')

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
            if l and (l[0] is not '#'):
                file_list.append(l.split()[0])
        f.close()

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

        # Distance from the camera (FOV) center
        camdist = np.sqrt(tbdata.field('DETX    ') ** 2. + tbdata.field('DETY    ') ** 2.)
        camdist_col = pyfits.Column(name='XCAMDIST', format='1E', unit='deg', array=camdist)

        ## cos(DEC)
        #cosdec = np.cos(tbdata.field('DEC     ') * np.pi / 180.)
        #cosdec_col = pyfits.Column(name='XCOSDEC', format='1E', array=cosdec)

        # Add new columns to the table
        #coldefs_new = pyfits.ColDefs([camdist_col, cosdec_col])
        coldefs_new = pyfits.ColDefs([camdist_col])
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
        emin, emax = .1, 100.
        emask = (tbdata.field('ENERGY  ') > emin) \
            * (tbdata.field('ENERGY  ') < emax)

        # Only use events with < 4 deg camera distance
        camdmask = tbdata.field('XCAMDIST') < 4.

        # Combine cuts for photons
        #phomask = (tbdata.field('HIL_MSW ') < 1.1) * emask * camdmask
        #  (HIL_MSW>-2.0&&HIL_MSW<0.9)&&(HIL_MSL>-2.0&&HIL_MSL<2.0)
        phomask = (tbdata.field('HIL_MSW ') > -2.) * (tbdata.field('HIL_MSW ') < .9) * (tbdata.field('HIL_MSL ') > -2.)  * (tbdata.field('HIL_MSL ') < 2.) * emask * camdmask
        #phomask = (tbdata.field('HIL_MSW ') > -2.)  * (tbdata.field('HIL_MSW ') < .9) * emask * camdmask
        hadmask = (tbdata.field('HIL_MSW ') > 1.) * (tbdata.field('HIL_MSW ') < 10.) * emask * camdmask

        if firstloop :
            # If skymap center is not set, set it to the target position of the first run
            if objra == None or objdec == None :
                objra, objdec = ex1hdr['RA_OBJ'], ex1hdr['DEC_OBJ']

        pntra, pntdec = ex1hdr['RA_PNT'], ex1hdr['DEC_PNT']
        obj_cam_dist = pf.SkyCoord(objra, objdec).dist(pf.SkyCoord(pntra, pntdec))

        exposure_run = ex1hdr['LIVETIME']
        exposure += exposure_run

        logging.info('RUN Start date/time : {0} {1}'.format(ex1hdr['DATE_OBS'], ex1hdr['TIME_OBS']))
        logging.info('RUN Stop date/time  : {0} {1}'.format(ex1hdr['DATE_END'], ex1hdr['TIME_END']))
        logging.info('RUN Exposure        : {0:.2f} [s]'.format(exposure_run))
        logging.info('RUN Pointing pos.   : RA {0:.4f} [deg], Dec {1:.4f} [deg]'.format(pntra, pntdec))
        logging.info('RUN Obj. cam. dist. : {0:.4f} [deg]'.format(obj_cam_dist))
        
        # Most important cut for the acceptance calculation: exclude source region
        exmask = np.invert(np.sqrt(((tbdata.field('RA      ') - objra) / np.cos(objdec * np.pi / 180.)) ** 2.
                                   + (tbdata.field('DEC     ') - objdec) ** 2.) < rexdeg)

        photbdata = tbdata[phomask * exmask]
        hadtbdata = tbdata[hadmask * exmask]

        if template_background and len(hadtbdata) < 100:
            logging.warning('No background type events for template background detected.')
            logging.info('Switching off template background.')
            template_background = False

        #---------------------------------------------------------------------------
        # Calculate camera acceptance

        n, bins, nerr, r, r_a, ex_a, fitter = pf.get_cam_acc(
            photbdata.field('XCAMDIST'),
            exreg=[[rexdeg, obj_cam_dist]],
            fit=True
            )

        # DEBUG
        #if logging.root.level is logging.DEBUG :
        #    fitter.print_results()

        # DEBUG plot
        #plt.errorbar(r, n / r_a / (1. - ex_a), nerr / r_a / (1. - ex_a))
        #plt.plot(r, fitter.fitfunc(fitter.results[0], r))
        #plt.show()

        had_acc, had_n, had_fit = None, None, None
        if template_background :
            had_acc = pf.get_cam_acc(
                hadtbdata.field('XCAMDIST'),
                exreg=[[rexdeg, obj_cam_dist]],
                fit=True
                )
            had_n, had_fit = had_acc[0], had_acc[6]
            logging.debug('Camera acceptance hadrons fit probability: {0}'.format(had_fit.prob))

        #---------------------------------------------------------------------------
        # Skymap - definitions/calculation

        # All photons including the exclusion regions
        photbdata = tbdata[phomask]
        hadtbdata = tbdata[hadmask]

        tpl_acc_cor_use_interp = True
        tpl_acc_f, tpl_acc = None, None
        if template_background :
            if tpl_acc_cor_use_interp :
                tpl_acc_f = scipy.interpolate.UnivariateSpline(r, n.astype(float) / had_n.astype(float), s=0, k=1)
            else :
                tpl_acc_f = lambda r: fitter.fitfunc(p1, r) / had_fit.fitfunc(had_fit.results[0], r)
            tpl_acc = tpl_acc_f(hadtbdata.field('XCAMDIST'))
            m = hadtbdata.field('XCAMDIST') > r[-1]
            tpl_acc[m] = tpl_acc_f(r[-1])
            m = hadtbdata.field('XCAMDIST') < r[0]
            tpl_acc[m] = tpl_acc_f(r[0])

        # Object position in the sky
        if firstloop :
            #objra, objdec, skymap_size = ex1hdr['RA_OBJ'], ex1hdr['DEC_OBJ'], 6.
            #if objra == None or objdec == None :
            #    objra, objdec = ex1hdr['RA_OBJ'], ex1hdr['DEC_OBJ']

            # Calculate skymap limits
            skymap_nbins = int(skymap_size / skymap_bin_size)
            sky_dec_min, sky_dec_max = objdec - skymap_size / 2., objdec + skymap_size / 2.
            objcosdec = np.cos(objdec * np.pi / 180.)
            sky_ra_min, sky_ra_max = objra - skymap_size / 2. / objcosdec, objra + skymap_size / 2. / objcosdec

        # Create sky map (i.e. bin events)
        # NOTE: In histogram2d the first axes is the vertical (y, DEC) the 2nd the horizontal axes (x, RA)
        sky = np.histogram2d(x=photbdata.field('DEC     '), y=photbdata.field('RA      '),
                             bins=[skymap_nbins, skymap_nbins],
                             range=[[sky_dec_min, sky_dec_max], [sky_ra_min, sky_ra_max]])

        if firstloop :
            # Just used to have the x-min/max, y-min/max saved
            H, xedges, yedges = sky
            # NOTE: The zero point of the histogram 2d is at the lower left corner while
            #       the pyplot routine imshow takes [0,0] at the upper left corner (i.e.
            #       we have to invert the 1st axis before plotting, see below).
            #       Also, imshow uses the traditional 1st axes = x = RA, 2nd axes = y = DEC
            #       notation for the extend keyword
            #extent = [yedges[0], yedges[-1], xedges[-1], xedges[0]]
            extent = [yedges[0], yedges[-1], xedges[0], xedges[-1]]
            sky_hist = sky[0]

            sky_ex_reg = pf.get_exclusion_region_map(sky_hist, (sky_ra_min, sky_ra_max), (sky_dec_min, sky_dec_max),
                                                     [pf.SkyCircle(pf.SkyCoord(objra, objdec), rexdeg)])

        else :
            sky_hist += sky[0]

        # Calculate camera acceptance
        dec_a = np.linspace(sky_dec_min, sky_dec_max, skymap_nbins + 1)
        ra_a = np.linspace(sky_ra_min, sky_ra_max, skymap_nbins + 1)
        xx, yy = np.meshgrid((ra_a[:-1] + ra_a[1:]) / 2. - pntra, (dec_a[:-1] + dec_a[1:]) / 2. - pntdec)
        rr = np.sqrt(xx ** 2. + yy ** 2.)
        p1 = fitter.results[0]
        acc = fitter.fitfunc(p1, rr) / fitter.fitfunc(p1, .1)
        m = rr > 4.
        acc[m] = fitter.fitfunc(p1, 4.) / fitter.fitfunc(p1, .1)

        # DEBUG plot
        #plt.imshow(acc[::-1], extent=extent, interpolation='nearest')
        #plt.colorbar()
        #plt.title('acc_bg_overs')
        #plt.show()
        
        if firstloop :
            acc_hist = acc * exposure_run # acc[0] before
        else :
            acc_hist += acc * exposure_run # acc[0] before

        if template_background :
            # Create hadron event like map for template background
            tpl_had = np.histogram2d(x=hadtbdata.field('DEC     '), y=hadtbdata.field('RA      '),
                                     bins=[skymap_nbins, skymap_nbins],
                                     #weights=1./accept,
                                     range=[[sky_dec_min, sky_dec_max], [sky_ra_min, sky_ra_max]])
            if firstloop :
                tpl_had_hist = tpl_had[0]
            else :
                tpl_had_hist += tpl_had[0]


            # Create acceptance map for template background
            tpl_acc = np.histogram2d(x=hadtbdata.field('DEC     '), y=hadtbdata.field('RA      '),
                                     bins=[skymap_nbins, skymap_nbins],
                                     weights=tpl_acc,
                                     range=[[sky_dec_min, sky_dec_max], [sky_ra_min, sky_ra_max]])
            if firstloop :
                tpl_acc_hist = tpl_acc[0]
            else :
                tpl_acc_hist += tpl_acc[0]

        # Close fits file
        hdulist.close()

        # Clean up memory
        gc.collect()

        firstloop = False

    #---------------------------------------------------------------------------
    # Calculate final skymaps

    logging.info('Processing final sky maps')

    # Calculate oversampled skymap, ring background, excess, and significances
    sc = pf.get_sky_mask_circle(r_overs, skymap_bin_size)
    sr = pf.get_sky_mask_ring(ring_bg_r_min, ring_bg_r_max, skymap_bin_size)
    acc_hist /= exposure

    logging.info('Calculating oversampled event map ..')
    sky_overs, sky_overs_alpha = pf.oversample_sky_map(sky_hist, sc)

    logging.info('Calculating oversampled ring background map ..')
    sky_bg_ring, sky_bg_ring_alpha = pf.oversample_sky_map(sky_hist, sr, sky_ex_reg)

    logging.info('Calculating oversampled event acceptance map ..')
    acc_overs, acc_overs_alpha = pf.oversample_sky_map(acc_hist, sc)

    logging.info('Calculating oversampled ring background acceptance map ..')
    acc_bg_overs, acc_bg_overs_alpha = pf.oversample_sky_map(acc_hist, sr, sky_ex_reg)

    sky_alpha = acc_overs / acc_bg_overs # camera acceptance
    sky_excess = sky_overs - sky_bg_ring * sky_alpha
    sky_sign = pf.get_li_ma_sign(sky_overs, sky_bg_ring, sky_alpha)

    tpl_had_overs, tpl_sig_overs, tpl_exc_overs, tpl_alpha_overs = None, None, None, None
    if template_background :

        logging.info('Calculating oversampled template background map ..')
        tpl_had_overs, tpl_had_overs_alpha = pf.oversample_sky_map(tpl_had_hist, sc)

        logging.info('Calculating oversampled template acceptance map ..')
        tpl_acc_overs, tpl_acc_overs_alpha = pf.oversample_sky_map(tpl_acc_hist, sc)
        
        tpl_exc_overs = sky_overs - tpl_acc_overs
        tpl_alpha_overs = tpl_acc_overs / tpl_had_overs
        tpl_sig_overs = pf.get_li_ma_sign(sky_overs, tpl_had_overs, tpl_alpha_overs)

    #---------------------------------------------------------------------------
    # Write results to file

    if write_output :

        logging.info('Writing result to file ..')

        rarange, decrange = (sky_ra_min, sky_ra_max), (sky_dec_min, sky_dec_max)

        outfile_base_name = 'skymap_ring'
        outfile_extensions = ['_events.fits', '_acceptance.fits', '_events_over.fits', '_background_over.fits',
                              '_significance_over.fits', '_excess_over.fits', '_alpha_overs.fits']
        outfile_base_name = pf.unique_base_file_name(outfile_base_name, outfile_extensions)

        pf.map_to_primaryhdu(sky_hist, rarange, decrange).writeto(outfile_base_name + outfile_extensions[0])
        pf.map_to_primaryhdu(acc_hist, rarange, decrange).writeto(outfile_base_name + outfile_extensions[1])
        pf.map_to_primaryhdu(sky_overs, rarange, decrange).writeto(outfile_base_name + outfile_extensions[2])
        pf.map_to_primaryhdu(sky_bg_ring, rarange, decrange).writeto(outfile_base_name + outfile_extensions[3])
        pf.map_to_primaryhdu(sky_sign, rarange, decrange).writeto(outfile_base_name + outfile_extensions[4])
        pf.map_to_primaryhdu(sky_excess, rarange, decrange).writeto(outfile_base_name + outfile_extensions[5])
        pf.map_to_primaryhdu(sky_alpha, rarange, decrange).writeto(outfile_base_name + outfile_extensions[6])

        if template_background :
            outfile_base_name = 'skymap_template'
            outfile_extensions = ['_background.fits', '_acceptance.fits', '_background_over.fits',
                                  '_significance_over.fits', '_excess_over.fits', '_alpha_over.fits']
            outfile_base_name = pf.unique_base_file_name(outfile_base_name, outfile_extensions)

            pf.map_to_primaryhdu(tpl_had_hist, rarange, decrange).writeto(outfile_base_name + outfile_extensions[0])
            pf.map_to_primaryhdu(tpl_acc_hist, rarange, decrange).writeto(outfile_base_name + outfile_extensions[1])
            pf.map_to_primaryhdu(tpl_had_overs, rarange, decrange).writeto(outfile_base_name + outfile_extensions[2])
            pf.map_to_primaryhdu(tpl_sig_overs, rarange, decrange).writeto(outfile_base_name + outfile_extensions[3])
            pf.map_to_primaryhdu(tpl_exc_overs, rarange, decrange).writeto(outfile_base_name + outfile_extensions[4])
            pf.map_to_primaryhdu(tpl_alpha_overs, rarange, decrange).writeto(outfile_base_name + outfile_extensions[5])

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

        cir_overs = matplotlib.patches.Circle(
            (sky_ra_min + .08 * (sky_ra_max - sky_ra_min) + r_overs / objcosdec,
             sky_dec_min + .08 * (sky_dec_max - sky_dec_min) + r_overs / objcosdec),
            radius=r_overs / objcosdec,
            fill=True,
            edgecolor='1.',
            facecolor='1.',
            alpha=.6
            )

        gauss_func = lambda p, x: p[0] * np.exp(- (x - p[1]) ** 2. / 2. / p[2] ** 2.)
            
        if template_background :
            #----------------------------------------
            plt.figure(1, figsize=(13,7))
            
            plt.subplots_adjust(wspace=.4, left=.07, right=.96, hspace=.25, top=.93)

            #----------------------------------------
            ax = plt.subplot(231) 

            # [::-1] - invert 1st axis
            plt.imshow(sky_overs[::-1], extent=extent, interpolation='nearest')

            cb = plt.colorbar()
            cb.set_label('Events')
            #plt.clim(-4., 4.)

            set_title_and_axlabel('Events')

            ax.add_patch(cir_overs)

            #----------------------------------------
            ax = plt.subplot(232) 

            plt.imshow(tpl_exc_overs[::-1], extent=extent, interpolation='nearest')

            cb = plt.colorbar()
            cb.set_label('Excess events')

            set_title_and_axlabel('Template BG - Excess')

            #----------------------------------------
            ax = plt.subplot(233) 

            plt.imshow(tpl_sig_overs[::-1], extent=extent, interpolation='nearest')

            cb = plt.colorbar()
            cb.set_label('Significance')

            set_title_and_axlabel('Template BG - Significance')

            #----------------------------------------
            ax = plt.subplot(234) 

            plt.imshow(tpl_had_overs[::-1], extent=extent, interpolation='nearest')

            cb = plt.colorbar()
            cb.set_label('Background events')

            set_title_and_axlabel('Template BG - Background')

            # Need 2nd instance to be able to add two circles to the same figure
            cir_overs2 = matplotlib.patches.Circle(
                (sky_ra_min + .08 * (sky_ra_max - sky_ra_min) + r_overs / objcosdec,
                 sky_dec_min + .08 * (sky_dec_max - sky_dec_min) + r_overs / objcosdec),
                radius=r_overs / objcosdec,
                fill=True,
                edgecolor='1.',
                facecolor='1.',
                alpha=.6
                )
            ax.add_patch(cir_overs2)

            #----------------------------------------
            ax = plt.subplot(235) 

            plt.imshow(tpl_alpha_overs[::-1], extent=extent, interpolation='nearest')

            cb = plt.colorbar()
            cb.set_label('Alpha')

            set_title_and_axlabel('Template BG - Alpha')

            #----------------------------------------
            ax = plt.subplot(236)

            n, bins, patches = plt.hist(tpl_sig_overs.flatten(), bins=100, range=(-8., 8.),
                                        histtype='stepfilled', color='SkyBlue', log=True)

            x = np.linspace(-5., 8., 100)
            plt.plot(x, gauss_func([float(n.max()), 0., 1.], x), label='Gauss ($\sigma=1.$, mean=0.)')

            plt.xlabel('Significance')
            plt.title('Template BG', fontsize='medium')

            plt.ylim(1., n.max() * 5.)
            plt.legend(loc='upper left', prop={'size': 'small'})

        #----------------------------------------
        plt.figure(2, figsize=(13,7))

        plt.subplots_adjust(wspace=.4, left=.07, right=.96, hspace=.25, top=.93)

        #----------------------------------------
        ax = plt.subplot(231) 

        plt.imshow(sky_overs[::-1], extent=extent, interpolation='nearest')

        cb = plt.colorbar()
        cb.set_label('Events')
        #plt.clim(-4., 4.)

        set_title_and_axlabel('Events')

        ax.add_patch(cir_overs)

        #----------------------------------------
        ax = plt.subplot(232) 

        plt.imshow(sky_excess[::-1], extent=extent, interpolation='nearest')

        cb = plt.colorbar()
        cb.set_label('Excess events')

        set_title_and_axlabel('Ring BG - Excess')

        #----------------------------------------
        ax = plt.subplot(233) 

        plt.imshow(sky_sign[::-1], extent=extent, interpolation='nearest')

        cb = plt.colorbar()
        cb.set_label('Significance')

        set_title_and_axlabel('Ring BG - Significance')

        #----------------------------------------
        ax = plt.subplot(234) 

        plt.imshow(sky_bg_ring[::-1], extent=extent, interpolation='nearest')

        cb = plt.colorbar()
        cb.set_label('Background events')

        set_title_and_axlabel('Ring BG - Background')

        # Plot ring background region
        plt_r_ra = sky_ra_min + .03 * (sky_ra_max - sky_ra_min) + ring_bg_r_max / objcosdec
        plt_r_dec = sky_dec_min + .03 * (sky_dec_max - sky_dec_min) + ring_bg_r_max / objcosdec
        cir = matplotlib.patches.Circle(
            (plt_r_ra, plt_r_dec),
            radius=ring_bg_r_max / objcosdec,
            fill=False,
            edgecolor='1.',
            facecolor='0.',
            linestyle='solid',
            linewidth=1.,
            )
        ax.add_patch(cir)

        cir = matplotlib.patches.Circle(
            (plt_r_ra, plt_r_dec),
            radius=ring_bg_r_min / objcosdec,
            fill=False,
            edgecolor='1.',
            facecolor='0.',
            linestyle='solid',
            linewidth=1.,
            )
        ax.add_patch(cir)

        cir = matplotlib.patches.Circle(
            (plt_r_ra, plt_r_dec),
            radius=r_overs / objcosdec,
            fill=True,
            edgecolor='1.',
            facecolor='1.',
            alpha=.6
            )
        ax.add_patch(cir)

        #----------------------------------------
        ax = plt.subplot(235) 

        plt.imshow(sky_alpha[::-1], extent=extent, interpolation='nearest')

        cb = plt.colorbar()
        cb.set_label('Alpha')

        set_title_and_axlabel('Ring BG - Alpha')

        #----------------------------------------
        ax = plt.subplot(236)

        sign_hist_r_max = 2.
        sky_ex_reg = pf.get_exclusion_region_map(sky_hist, (sky_ra_min, sky_ra_max), (sky_dec_min, sky_dec_max),
                                                 [pf.SkyCircle(pf.SkyCoord(objra, objdec), sign_hist_r_max)])
        n, bins, patches = plt.hist(sky_sign[sky_ex_reg == 0.].flatten(), bins=100, range=(-8., 8.),
                                    histtype='stepfilled', color='SkyBlue', log=True)

        x = np.linspace(-5., 8., 100)
        plt.plot(x, gauss_func([float(n.max()), 0., 1.], x), label='Gauss ($\sigma=1.$, mean=0.)')

        plt.xlabel('Significance R < {0}'.format(sign_hist_r_max))
        plt.title('Ring BG', fontsize='medium')

        plt.ylim(1., n.max() * 5.)
        plt.legend(loc='upper left', prop={'size': 'small'})

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
        description='Creates skymaps (excess/significance) from VHE event lists in FITS format.'
    )
    parser.add_option(
        '-s','--skymap-size',
        dest='skymap_size',
        type='float',
        default=6.,
        help='Diameter of the sky map in degree [default: %default].'
    )
    parser.add_option(
        '-b','--bin-size',
        dest='bin_size',
        type='float',
        default=.03,
        help='Bin size in degree [default: %default].'
    )
    parser.add_option(
        '-c','--skymap-center',
        dest='skymap_center',
        type='str',
        default=None,
        help='Center of the skymap in RA and Dec (J2000) in degree. Format: \'(RA, Dec)\', including the quotation marks. If no center is given, the source position from the first input file is used.'
    )
    parser.add_option(
        '-r','--oversampling-radius',
        dest='oversampling_radius',
        type='float',
        default=.125,
        help='Radius used to correlated the sky maps in degree [default: %default].'
    )
    parser.add_option(
        '--ring-bg-radii',
        dest='ring_bg_radii',
        default='(.3, .7)',
        help='Inner and outer radius of the ring used for the ring background. Format \'(r_in, r_out)\', including the quotation marks [default: \'%default\'].'
    )
    parser.add_option(
        '-w','--write-output',
        dest='write_output',
        action='store_true',
        default=False,
        help='Write results to FITS files in current directory [default: %default]'
    )
    parser.add_option(
        '--no-template-background',
        dest='template_background',
        action='store_false',
        default=True,
        help='Switch off template background.'
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
        create_sky_map(
            input_file_name=args[0],
            skymap_size=options.skymap_size,
            skymap_bin_size=options.bin_size,
            r_overs=options.oversampling_radius,
            ring_bg_radii=options.ring_bg_radii,
            template_background=options.template_background,
            skymap_center=options.skymap_center,
            write_output=options.write_output,
            do_graphical_output=options.graphical_output,
            loglevel=options.loglevel
            )
    else :
        parser.print_help()

#===========================================================================
