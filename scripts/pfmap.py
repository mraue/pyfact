#! /usr/bin/env python

#===========================================================================
#
# WARNING: pyfits.new_table has a memory leak in version 2.4.0
#          see e.g. http://physicsnlinux.wordpress.com/2011/03/28/pyfits-memory-leak-in-new_table/
#          or http://trac6.assembla.com/pyfits/ticket/49
#          It should be fixed in the trunk r925++ which you can check out via svn:
#          svn co http://svn6.assembla.com/svn/pyfits/trunk/ ./
# UPDATE:  The memory leak is fixed in r925, let's hope for a new release version soon ;)
#
#===========================================================================
# Imports

import sys, time, logging, os, gc
import numpy as np
import matplotlib.pyplot as plt
import pyfits

import scipy.interpolate

# Add parent directory to python search path to get access to the pyfact package
sys.path.append(os.path.abspath(sys.path[0].rsplit('/', 1)[0]))
import pyfact as pf

#===========================================================================
# Main
def create_sky_map(input_file_name,
                   sky_size=5.,
                   sky_high_bin_size=0.05,
                   skymap_center=None,
                   write_output=False) :
    # Time it!
    t_1 = time.clock()

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Welcome user, print out basic information on package versions
    logging.info('This is {0}, running pyfact v{1}, numpy v{2}, and pyfits v{3}.'.format(os.path.split(__file__)[1], pf.__version__, np.__version__, pyfits.__version__))

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
        f = open(input_file_name)
        for l in f:
            l = l.strip(' \t\n')
            if l :
                file_list.append(l)
        f.close()

    #---------------------------------------------------------------------------
    # Loop over the file list, calculate quantities, & fill histograms

    # Skymap definition
    #sky_size, sky_high_bin_size = 6., 0.05
    rexdeg = .2

    # Intialize some variables
    objra, objdec = None, None
    if skymap_center :
        objra, objdec = eval(skymap_center)
    sky_high_nbins, sky_dec_min, sky_dec_max, objcosdec, sky_ra_min, sky_ra_max = 0, 0., 0., 0., 0., 0.
    sky_hist, acc_hist, extent = None, None, None
    tplt_had_hist, tplt_acc_hist = None, None
    sky_ex_reg = None

    firstloop = True

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

        # cos(DEC)
        cosdec = np.cos(tbdata.field('DEC     ') * np.pi / 180.)
        cosdec_col = pyfits.Column(name='XCOSDEC', format='1E', array=cosdec)

        # Add new columns to the table
        coldefs_new = pyfits.ColDefs([camdist_col, cosdec_col])
        # WARNING: pyfits.new_table has a memory leak in version 2.4.0
        #          see e.g. http://physicsnlinux.wordpress.com/2011/03/28/pyfits-memory-leak-in-new_table/
        #          or http://trac6.assembla.com/pyfits/ticket/49
        newtable = pyfits.new_table(hdulist[1].columns + coldefs_new)

        # Print new table columns
        #newtable.columns.info()

        # New table data
        tbdata = newtable.data

        #---------------------------------------------------------------------------
        # Select events

        # Select events with at least one tel above the required image amplitude/size (here iamin p.e.)
        # This needs to be changed for the new TELEVENT table scheme
        iamin = 80.
        iamask = (tbdata.field('HIL_TEL_SIZE')[:,0] > iamin) \
            + (tbdata.field('HIL_TEL_SIZE')[:,1] > iamin) \
            + (tbdata.field('HIL_TEL_SIZE')[:,2] > iamin) \
            + (tbdata.field('HIL_TEL_SIZE')[:,3] > iamin)

        # Select events between emin & emax TeV
        emin, emax = .1, 100.
        emask = (tbdata.field('ENERGY  ') > emin) \
            * (tbdata.field('ENERGY  ') < emax)

        # Only use events with < 4 deg camera distance
        camdmask = tbdata.field('XCAMDIST') < 4.

        # Combine cuts for photons
        phomask = (tbdata.field('HIL_MSW ') < 1.1) * iamask * emask * camdmask
        hadmask = (tbdata.field('HIL_MSW ') > 1.3) * (tbdata.field('HIL_MSW ') < 10.) * iamask * emask * camdmask

        #objra, objdec, rexdeg = ex1hdr['RA_OBJ'], ex1hdr['DEC_OBJ'], 0.2

        if firstloop :
            #objra, objdec, sky_size = ex1hdr['RA_OBJ'], ex1hdr['DEC_OBJ'], 6.
            if objra == None or objdec == None :
                objra, objdec = ex1hdr['RA_OBJ'], ex1hdr['DEC_OBJ']

        # Most important cut for the acceptance calculation: exclude source region
        exmask = np.invert(np.sqrt(((tbdata.field('RA      ') - objra) / np.cos(objdec * np.pi / 180.)) ** 2.
                                   + (tbdata.field('DEC     ') - objdec) ** 2.) < rexdeg)

        photbdata = tbdata[phomask * exmask]
        hadtbdata = tbdata[hadmask * exmask]

        #---------------------------------------------------------------------------
        # Calculate camera acceptance

        n, bins, nerr, r, r_a, ex_a, fitter = pf.get_cam_acc(photbdata.field('XCAMDIST'),
                                                             exreg=[[rexdeg, .5]], fit=True)

        had_n = pf.get_cam_acc(hadtbdata.field('XCAMDIST'), exreg=[[rexdeg, .5]], fit=False)[0]

        tplt_acc_f = scipy.interpolate.UnivariateSpline(r, n.astype(float) / had_n.astype(float), s=0)

        #if firstloop :
        #    plt.figure(3)
        #    plt.plot(r, n.astype(float) / had_n.astype(float), 'd')
        #    x = np.linspace(0., 4., 100)
        #    plt.plot(x, tplt_acc_f(x))

        #---------------------------------------------------------------------------
        # Skymap - definitions/calculation

        # All photons including the exclusion regions
        photbdata = tbdata[phomask]
        hadtbdata = tbdata[hadmask]

        # Calculate camera acceptance for each event using the fit function
        p1 = fitter.results[0]
        accept = fitter.fitfunc(p1, photbdata.field('XCAMDIST')) / fitter.fitfunc(p1, .1)
        m = photbdata.field('XCAMDIST') > 4.
        accept[m] = fitter.fitfunc(p1, 4.) / fitter.fitfunc(p1, .1)

        tplt_acc = tplt_acc_f(hadtbdata.field('XCAMDIST'))
        m = hadtbdata.field('XCAMDIST') > r[-1]
        tplt_acc[m] = tplt_acc_f(r[-1])
        m = hadtbdata.field('XCAMDIST') < r[0]
        tplt_acc[m] = tplt_acc_f(r[0])

        # Object position in the sky
        if firstloop :
            #objra, objdec, sky_size = ex1hdr['RA_OBJ'], ex1hdr['DEC_OBJ'], 6.
            #if objra == None or objdec == None :
            #    objra, objdec = ex1hdr['RA_OBJ'], ex1hdr['DEC_OBJ']

            # Calculate skymap limits
            #sky_high_bin_size = 0.05 # Bin size for the highly sampled sky map (deg)
            sky_high_nbins = int(sky_size / sky_high_bin_size)
            sky_dec_min, sky_dec_max = objdec - sky_size / 2., objdec + sky_size / 2.
            objcosdec = np.cos(objdec * np.pi / 180.)
            sky_ra_min, sky_ra_max = objra - sky_size / 2. / objcosdec, objra + sky_size / 2. / objcosdec

        # Create sky map (i.e. bin events)
        # NOTE: In histogram2d the first axes is the vertical (y, DEC) the 2nd the horizontal axes (x, RA)
        sky = np.histogram2d(x=photbdata.field('DEC     '), y=photbdata.field('RA      '),
                             bins=[sky_high_nbins, sky_high_nbins],
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
                                                     [pf.sky_circle(pf.sky_coord(objra, objdec), .2)])

        else :
            sky_hist += sky[0]

        # Create acceptance corrected sky map
        acc = np.histogram2d(x=photbdata.field('DEC     '), y=photbdata.field('RA      '),
                             bins=[sky_high_nbins, sky_high_nbins],
                             weights=1./accept,
                             range=[[sky_dec_min, sky_dec_max], [sky_ra_min, sky_ra_max]])
        if firstloop :
            acc_hist = acc[0]
        else :
            acc_hist += acc[0]

        # Create hadron event like map for template background
        tplt_had = np.histogram2d(x=hadtbdata.field('DEC     '), y=hadtbdata.field('RA      '),
                                  bins=[sky_high_nbins, sky_high_nbins],
                                  #weights=1./accept,
                                  range=[[sky_dec_min, sky_dec_max], [sky_ra_min, sky_ra_max]])
        if firstloop :
            tplt_had_hist = tplt_had[0]
        else :
            tplt_had_hist += tplt_had[0]


        # Create acceptance map for template background
        tplt_acc = np.histogram2d(x=hadtbdata.field('DEC     '), y=hadtbdata.field('RA      '),
                                  bins=[sky_high_nbins, sky_high_nbins],
                                  weights=tplt_acc,
                                  range=[[sky_dec_min, sky_dec_max], [sky_ra_min, sky_ra_max]])
        if firstloop :
            tplt_acc_hist = tplt_acc[0]
        else :
            tplt_acc_hist += tplt_acc[0]

        # Close fits file
        hdulist.close()

        # Clean up memory
        gc.collect()

        firstloop = False

    #---------------------------------------------------------------------------
    # Calculate final skymaps

    logging.info('Processing final sky maps')

    # DEBUG
    #sky = [np.random.rand(100, 100)]
    #sky_high_bin_size = 0.1
    #extent = [0., 1., 0., 1.]

    # Calculate oversampled skymap, ring background, excess, and significances

    sc = pf.get_sky_mask_circle(.125, sky_high_bin_size)
    sr = pf.get_sky_mask_ring(.3, 1., sky_high_bin_size)

    logging.info('Calculating oversampled event map ..')
    sky_overs, sky_overs_alpha = pf.oversample_sky_map(sky_hist, sc)

    logging.info('Calculating oversampled ring background map ..')
    #sky_bg_ring, sky_bg_ring_alpha = pf.oversample_sky_map(sky_hist * sky_ex_reg, sr)
    sky_bg_ring, sky_bg_ring_alpha = pf.oversample_sky_map(sky_hist, sr, sky_ex_reg)

    logging.info('Calculating oversampled event acceptance map ..')
    acc_overs, acc_overs_alpha = pf.oversample_sky_map(acc_hist, sc)

    logging.info('Calculating oversampled ring background acceptance map ..')
    #acc_bg_overs, acc_bg_overs_alpha = pf.oversample_sky_map(acc_hist * sky_ex_reg, sr)
    acc_bg_overs, acc_bg_overs_alpha = pf.oversample_sky_map(acc_hist, sr, sky_ex_reg)

    logging.info('Calculating oversampled template background map ..')
    tplt_had_overs, tplt_had_overs_alpha = pf.oversample_sky_map(tplt_had_hist, sc)

    logging.info('Calculating oversampled template acceptance map.')
    tplt_acc_overs, tplt_acc_overs_alpha = pf.oversample_sky_map(tplt_acc_hist, sc)

    sky_alpha = sky_overs_alpha / sky_bg_ring_alpha # geometry
    sky_alpha *=  acc_bg_overs / sky_bg_ring / acc_overs * sky_overs # camera acceptance
    sky_excess = sky_overs - sky_bg_ring * sky_alpha
    sky_sign = pf.get_li_ma_sign(sky_overs, sky_bg_ring, sky_alpha)

    #---------------------------------------------------------------------------
    # Write results to file

    if write_output :

        logging.info('Writing result to file ..')

        outfile_base_name = 'skymap_ring'
        outfile_extensions = ['_ev-n.fits', '_ac-n.fits', '_ev-c.fits', '_bg-c.fits',
                              '_si-c.fits', '_ex-c.fits', '_al-c.fits']
        outfile_base_name = pf.unique_base_file_name(outfile_base_name, outfile_extensions)

        rarange, decrange = (sky_ra_min, sky_ra_max), (sky_dec_min, sky_dec_max)

        pf.map_to_primaryhdu(sky_hist, rarange, decrange).writeto(outfile_base_name + outfile_extensions[0])
        pf.map_to_primaryhdu(acc_hist, rarange, decrange).writeto(outfile_base_name + outfile_extensions[1])
        pf.map_to_primaryhdu(sky_overs, rarange, decrange).writeto(outfile_base_name + outfile_extensions[2])
        pf.map_to_primaryhdu(sky_bg_ring, rarange, decrange).writeto(outfile_base_name + outfile_extensions[3])
        pf.map_to_primaryhdu(sky_sign, rarange, decrange).writeto(outfile_base_name + outfile_extensions[4])
        pf.map_to_primaryhdu(sky_excess, rarange, decrange).writeto(outfile_base_name + outfile_extensions[5])
        pf.map_to_primaryhdu(sky_alpha, rarange, decrange).writeto(outfile_base_name + outfile_extensions[6])

        logging.info('The output files can be found in {0}'.format(os.getcwd()))

    #---------------------------------------------------------------------------
    # Plot results
    plt.figure(1, figsize=(12,10))

    plt.subplots_adjust(hspace=0.3, wspace=0.3, left=.1, bottom=.1, right=.95, top=.94)

    #----------------------------------------
    plt.subplot(331)
    # [::-1] inverts the first axis of the array
    plt.imshow(sky_overs[::-1], extent=extent, interpolation='nearest')
    cb = plt.colorbar()

    plt.xlabel('RA (deg)')
    plt.ylabel('Dec (deg)')
    plt.title('Oversampled skymap', fontsize='medium')

    cb.set_label('Number of events')

    #----------------------------------------
    plt.subplot(332)
    plt.imshow(sky_bg_ring[::-1], extent=extent, interpolation='nearest')
    cb = plt.colorbar()

    plt.xlabel('RA (deg)')
    plt.ylabel('Dec (deg)')
    plt.title('Ring background', fontsize='medium')

    cb.set_label('Number of events')

    #----------------------------------------
    plt.subplot(333)
    plt.imshow(sky_excess[::-1], extent=extent, interpolation='nearest')
    cb = plt.colorbar()
    cb.set_label('Number of events')

    plt.xlabel('RA (deg)')
    plt.ylabel('Dec (deg)')
    plt.title('Excess map from ring background', fontsize='medium')

    #----------------------------------------
    plt.subplot(334)
    plt.imshow((sky_overs_alpha / sky_bg_ring_alpha)[::-1], extent=extent, interpolation='nearest')
    cb = plt.colorbar()
    cb.set_label('Alpha')

    plt.xlabel('RA (deg)')
    plt.ylabel('Dec (deg)')
    plt.title('Alpha factor from geometry', fontsize='medium')

    #----------------------------------------
    plt.subplot(335)

    plt.imshow(sky_sign[::-1], extent=extent, interpolation='nearest')

    cb = plt.colorbar()
    cb.set_label('Significance')
    #plt.clim(-4., 4.)

    plt.xlabel('RA (deg)')
    plt.ylabel('Dec (deg)')
    plt.title('Significance map', fontsize='medium')

    #----------------------------------------
    plt.subplot(336)

    # Cut away the outer border of the skymap for the significance/excess distribution
    nb = int(1. / sky_high_bin_size)
    #n, bins, patches = plt.hist(sky_sign[nb:-nb, nb:-nb].flatten(), bins=100, range=(-8., 8.),
    #                            histtype='stepfilled', color='SkyBlue', log=True)

    # Only consider a circle of 2.5 deg around the source
    sky_ex_reg = pf.get_exclusion_region_map(sky_hist, (sky_ra_min, sky_ra_max), (sky_dec_min, sky_dec_max),
                                             [pf.sky_circle(pf.sky_coord(objra, objdec), 2.2)])
    n, bins, patches = plt.hist(sky_sign[sky_ex_reg == 0.].flatten(), bins=100, range=(-8., 8.),
                                histtype='stepfilled', color='SkyBlue', log=True)

    #plt.axvline(sky_sign[np.invert(np.isnan(sky_sign))].mean(), color='0.', linestyle='--', label='Mean')

    gauss = lambda p, x: p[0] * np.exp(- (x - p[1]) ** 2. / 2. / p[2] ** 2.)

    #fitter = chisquare_fitter(gauss)
    #p0 = [float(n.max()), 0., 1.]
    #fitter.fit_data(p0, (bins[1:] + bins[:-1]) / 2., n, n * .5)
    #p1 = fitter.results[0]

    #print fitter.results

    x = np.linspace(-5., 8., 100)

    plt.plot(x, gauss([float(n.max()), 0., 1.], x), label='Gauss ($\sigma=1.$)')
    #plt.plot(x, gauss(p1, x), '--',  label='Gauss fit')

    plt.xlabel("Significance")

    plt.ylim(1., n.max() * 5.)

    plt.legend(loc='upper left', prop={'size': 'small'})

    #----------------------------------------
    plt.subplot(337)

    n, bins, patches = plt.hist(sky_excess[nb:-nb, nb:-nb].flatten(), bins=100, range=(-30., 30.),
                                histtype='stepfilled', color='SkyBlue', log=False)

    plt.axvline(sky_sign[np.invert(np.isnan(sky_sign))].mean(), color='0.', linestyle='--')

    plt.xlabel("Excess")

    #----------------------------------------
    plt.subplot(338)
    plt.imshow(sky_alpha[::-1], extent=extent, interpolation='nearest')
    cb = plt.colorbar()
    cb.set_label('Alpha')

    plt.xlabel('RA (deg)')
    plt.ylabel('Dec (deg)')
    plt.title('Alpha factor', fontsize='medium')

    #DEBUG
    plt.figure(2)
    #
    #plt.imshow(sky_ex_reg[::-1], extent=extent, interpolation='nearest')
    #cb = plt.colorbar()

    #plt.subplot(221)
    #plt.imshow(tplt_had_overs[::-1], extent=extent, interpolation='nearest')
    #cb = plt.colorbar()

    plt.subplot(222)
    plt.imshow(tplt_acc_overs[::-1], extent=extent, interpolation='nearest')
    #plt.imshow(np.where(sky_hist != 0., 1., 0.), extent=extent, interpolation='nearest')
    cb = plt.colorbar()

    plt.subplot(223)
    plt.imshow((sky_overs - tplt_acc_overs)[::-1], extent=extent, interpolation='nearest')
    cb = plt.colorbar()
    #plt.clim(-30., 100.)

    plt.subplot(224)
    tplt_sig_overs = pf.get_li_ma_sign(sky_overs, tplt_had_overs, tplt_acc_overs / tplt_had_overs)
    plt.imshow(tplt_sig_overs[::-1], extent=extent, interpolation='nearest')
    cb = plt.colorbar()

    plt.subplot(221)
    # Only consider a circle of 2.5 deg around the source
    sky_ex_reg = pf.get_exclusion_region_map(sky_hist, (sky_ra_min, sky_ra_max), (sky_dec_min, sky_dec_max),
                                             [pf.sky_circle(pf.sky_coord(objra, objdec), 3.)])
    #n, bins, patches = plt.hist(tplt_sig_overs[sky_ex_reg == 0.].flatten(), bins=100, range=(-8., 8.),
    n, bins, patches = plt.hist(tplt_sig_overs.flatten(), bins=100, range=(-8., 8.),
                                histtype='stepfilled', color='SkyBlue', log=True)

    gauss = lambda p, x: p[0] * np.exp(- (x - p[1]) ** 2. / 2. / p[2] ** 2.)

    x = np.linspace(-5., 8., 100)

    plt.plot(x, gauss([float(n.max()), 0., 1.], x), label='Gauss ($\sigma=1.$)')

    plt.xlabel("Significance")

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
    # We should use to switch to argparse soon (python v2.7++)
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
        default=5.,
        help='Diameter of the sky map in degree [default: %default].'
    )
    parser.add_option(
        '-b','--bin-size',
        dest='bin_size',
        type='float',
        default=.05,
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
        '-o','--write-output',
        dest='write_output',
        action="store_true",
        default=False,
        help='Write results to file [default: %default]'
    )

    options, args = parser.parse_args()

    #input_file_name = '/Users/mraue/Stuff/work/cta/2011/fits/data/run_00058896_eventlist.fits.gz'
    #input_file_name = '/Users/mraue/Stuff/work/cta/2011/fits/data/test.bnk'

    if len(args) == 1 :
        create_sky_map(
            input_file_name=args[0],
            sky_size=options.skymap_size,
            sky_high_bin_size=options.bin_size,
            skymap_center=options.skymap_center,
            write_output=options.write_output,
            )
    else :
        parser.print_help()

#===========================================================================
