#! /usr/bin/env python

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
import scipy.optimize
import scipy.integrate
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
def sim_evlist(flux=.1,
               obstime=.5,
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
    obstime *= 3600. #observation time in seconds
    
    obj_ra, obj_dec = 0., .5
    pnt_ra, pnt_dec = 0., 0.
    t_min = 24600.

    objcosdec = np.cos(obj_dec * np.pi / 180.)

    input_file_name = '/Users/mraue/Stuff/work/cta/2011/fits/irf/cta/SubarrayE_IFAE_50hours_20101102.log'
    #input_file_name = '/Users/mraue/Stuff/work/cta/2011/fits/irf/cta/SubarrayB_IFAE_50hours_20101102.log'

    # open effective area file
    irf_data = np.loadtxt(input_file_name)

    #---------------------------------------------------------------------------
    # Signal

    log_e_cen = (irf_data[:,0] + irf_data[:,1]) / 2.

    func_pl = lambda x, p: p[0] * x ** (-p[1])

    e_bin_w = 10. ** irf_data[:,1] - 10. ** irf_data[:,0]
    pho_rate = func_pl(10. ** log_e_cen, (1E-11, 3.)) * irf_data[:,7] * 1E4 * e_bin_w

    func_dbpl = lambda p, x: p[0] * x ** (-p[1]) * (1. + (x/p[2]) ** 2.) ** ((p[1] - p[3]) / 2.) * (1. + (x/p[4]) ** 1.) ** ((p[3] - p[5]) / 1.)
    #p0 = [1E10, -5., .1, 10., -1., 1., 10., -.5]
    p0 = [1E13, -6., .05,-.5, 1E3, 10.]

    fitter = pf.ChisquareFitter(func_dbpl)
    #fitter.fit_data(p0, 10. ** log_e_cen, irf_data[:,7], irf_data[:,7] * .12)
    fitter.fit_data(p0, 10. ** log_e_cen, irf_data[:,7] / irf_data[:,4], irf_data[:,7]  / irf_data[:,4] * .1) # correct for leakage?
    fitter.print_results()

    f_test = lambda x : func_pl(x, (3.45E-11 * flux, 2.63)) * func_dbpl(fitter.results[0], x) * 1E4

    #print pho_rate * obstime
    #print 'x1 : ', scipy.integrate.quad(f_test, 0.01, 100.)
    #print 'x2 : ', np.sum(pho_rate)

    log_e_steps = np.linspace(-2., 2., 150)
    int_rate = np.zeros(150)
    for i, e in enumerate(log_e_steps) :
        int_rate[i] = scipy.integrate.quad(f_test, 0.01, 10. ** e)[0]

    ev_gen_f = scipy.interpolate.UnivariateSpline(
        int_rate / scipy.integrate.quad(f_test, 0.01, 100.)[0],
        log_e_steps,
        s=0,
        k=1
        )

    n_events =  scipy.integrate.quad(f_test, 0.01, 100.)[0] * obstime

    logging.info('Number of photons :{0}'.format(n_events))

    evlist_e = ev_gen_f(np.random.rand(n_events))

    evlist_psf = np.where(evlist_e > 1., .05, 5.6E-2 * np.exp(-.9 * evlist_e))
    #evlist_psf = np.where(evlist_e > 1., .05, 3.3E-2 * np.exp(-1.4 * evlist_e))
    evlist_dec = obj_dec + np.random.randn(n_events) * evlist_psf
    evlist_ra =  obj_ra + np.random.randn(n_events) * evlist_psf / objcosdec

    evlist_t = t_min + obstime * np.random.rand(n_events) / 86400.

    #---------------------------------------------------------------------------
    # Background

    #plt.figure(1)

    log_e_cen = (irf_data[:,0] + irf_data[:,1]) / 2.
    # Protons + electron
    p_rate_area = (irf_data[:,10] + irf_data[:,11]) / irf_data[:,5] / np.pi

    logging.info('Total proton rate = {0}'.format(np.sum(irf_data[:,10])))

    p_rate_total =  np.sum(p_rate_area)

    ev_gen_f = scipy.interpolate.UnivariateSpline(
        np.cumsum(p_rate_area) / np.sum(p_rate_area),
        log_e_cen,
        s=0,
        k=1
        )

    cam_acc = lambda p, x: p[0] * x ** 0. * (1. + (x / p[1]) ** p[2]) ** ((0. + p[3]) / p[2])
    cam_acc_par = (1.,1.7, 6., -5.5)

    r_steps = np.linspace(0.001, 4., 150)
    int_cam_acc = np.zeros(150)
    for i, r in enumerate(r_steps) :
        int_cam_acc[i] = scipy.integrate.quad(lambda x: cam_acc(cam_acc_par, x) * x * 2. * np.pi, 0., r)[0]

    n_events_bg = int(p_rate_total * obstime * int_cam_acc[-1])

    logging.info('Number of protons :{0}'.format(n_events_bg))

    tplt_multi = 5
    evlist_bg_e = ev_gen_f(np.random.rand(n_events_bg * (tplt_multi + 1)))

    ev_gen_f2 = scipy.interpolate.UnivariateSpline(
        int_cam_acc / scipy.integrate.quad(lambda x: cam_acc(cam_acc_par, x) * 2. * x * np.pi, 0., 4.)[0],
        r_steps,
        s=0,
        k=1
        )

    evlist_bg_r = ev_gen_f2(np.random.rand(n_events_bg * (tplt_multi + 1)))

    r_max = 4.
    #evlist_bg_r = np.sqrt(np.random.rand(n_events_bg * (tplt_multi + 1))) * r_max
    rnd = np.random.rand(n_events_bg * (1 + tplt_multi))
    evlist_bg_rx = np.sqrt(rnd) * evlist_bg_r * np.where(np.random.randint(2, size=(n_events_bg * (tplt_multi + 1))) == 0, -1., 1.)
    evlist_bg_ry = np.sqrt(1. - rnd) * evlist_bg_r * np.where(np.random.randint(2, size=(n_events_bg * (tplt_multi + 1))) == 0, -1., 1.)

    #evlist_bg_sky_r = np.sqrt(np.random.rand(n_events_bg * (tplt_multi + 1))) * r_max
    #evlist_bg_sky_r = ev_gen_f2(np.random.rand(n_events_bg * (tplt_multi + 1)))
    rnd = np.random.rand(n_events_bg * (tplt_multi + 1))
    evlist_bg_ra = np.sin(2. * np.pi * rnd) * evlist_bg_r / objcosdec
    evlist_bg_dec = np.cos(2. * np.pi * rnd)  * evlist_bg_r

    #plt.hist(evlist_bg_rx ** 2. + evlist_bg_ry**2., bins=50)

    #print float(n_events_bg * (tplt_multi + 1)) / np.sum(p_rate_area) / 86400.
    evlist_bg_t = t_min + obstime *  np.random.rand(n_events_bg * (tplt_multi + 1)) / 86400.

    #---------------------------------------------------------------------------
    # Plots & debug
    
    plt.figure(3)

    objra, objdec = 0., 0.
    H, xedges, yedges = np.histogram2d(
        np.append(evlist_bg_dec, evlist_dec),
        np.append(evlist_bg_ra, evlist_ra),
        bins=[100, 100],
        range=[[objra - 3., objra + 3.], [objdec - 3., objdec + 3.]]
        )
    extent = [yedges[0], yedges[-1], xedges[-1], xedges[0]]
    plt.imshow(H, extent=extent, interpolation='nearest')
    cb = plt.colorbar()
    cb.set_label('Number of events')

    plt.xlabel('RA (deg)')
    plt.ylabel('Dec (deg)')

    test_r = np.sqrt(evlist_bg_ra ** 2. + evlist_bg_dec ** 2.)
    logging.info('Number of BG events in a circle of area 1 deg^2 = {0}'.format(np.sum(test_r[0:n_events_bg] < np.sqrt(1. / np.pi))))
    logging.info('Expected number of BG event per area 1 deg^2 = {0}'.format(p_rate_total * obstime))

    obj_r = np.sqrt(((obj_ra - evlist_ra) / objcosdec) ** 2. + (obj_dec - evlist_dec) ** 2.)

    thetamax_on, thetamax_off = .1, .22
    non = np.sum(obj_r < thetamax_on) + np.sum(test_r[0:n_events_bg] < thetamax_on)
    noff = np.sum(test_r[0:n_events_bg] < thetamax_off)
    alpha = thetamax_on ** 2. / thetamax_off ** 2.

    logging.info('N_ON = {0}, N_OFF = {1}, ALPHA = {2}, SIGN = {3}'.format(
        non, noff, alpha, pf.get_li_ma_sign(non, noff, alpha)))

    plt.figure(2)

    plt.hist(obj_r ** 2., bins=30)

    #---------------------------------------------------------------------------
    # Output to file
    
    if write_output:
        coldefs_new = pyfits.ColDefs([
            pyfits.Column(name='TIME', format='1E', unit='deg', array=np.append(evlist_t, evlist_bg_t)),
            pyfits.Column(name='RA', format='1E', unit='deg', array=np.append(evlist_ra, evlist_bg_ra)),
            pyfits.Column(name='DEC', format='1E', unit='deg', array=np.append( evlist_dec, evlist_bg_dec)),
            pyfits.Column(name='DETX', format='1E', unit='deg', array=np.append(np.zeros(n_events), evlist_bg_rx)),
            pyfits.Column(name='DETY', format='1E', unit='deg', array=np.append(np.ones(n_events) * .5, evlist_bg_ry)),
            pyfits.Column(name='ENERGY', format='1E', unit='tev', array=10. ** np.append(evlist_e,evlist_bg_e)),
            pyfits.Column(name='HIL_MSW', format='1E', array=np.append(np.ones(n_events + n_events_bg),
                                                                       5. * np.ones(n_events_bg * tplt_multi))) 
            ])

        newtable = pyfits.new_table(coldefs_new)
        hdr = newtable.header
        hdr.update('RA_OBJ' , obj_ra)
        hdr.update('DEC_OBJ', obj_dec)
        hdr.update('RA_PNT', pnt_ra)
        hdr.update('DEC_PNT', pnt_dec)
        newtable.writeto('table5.fits')

    #----------------------------------------
    # Time it!
    t_2 = time.clock()
    logging.info('Execution took {0}'.format(pf.get_nice_time(t_2 - t_1)))

    logging.info('Thank you for choosing {0}. Have a great day!'.format(os.path.split(__file__)[1]))

    #----------------------------------------
    if do_graphical_output:
        plt.show()

#===========================================================================
# Main function
if __name__ == '__main__':
    # We should switch to argparse soon (python v2.7++)
    # http://docs.python.org/library/argparse.html#module-argparse
    import optparse
    parser = optparse.OptionParser(
        usage='%prog [options]',
        description='Creates IACT eventlist from IRFs.'
    )
    parser.add_option(
        '-t','--observation-time',
        dest='observation_time',
        type='float',
        default=.5,
        help='Observation time in hours [default: %default].'
    )
    parser.add_option(
        '-f','--flux',
        dest='flux',
        type='float',
        default=.1,
        help='Flux in units of Crab [default: %default].'
    )
    parser.add_option(
        '-w','--write-output',
        dest='write_output',
        action='store_true',
        default=False,
        help='Write results to FITS files in current directory [default: %default].'
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
        help='Amount of logging e.g. DEBUG, INFO, WARNING, ERROR [default: %default].'
    )

    options, args = parser.parse_args()

    #if len(args) == 1 :
    sim_evlist(
        flux=options.flux,
        obstime=options.observation_time,
        write_output=options.write_output,
        do_graphical_output=options.graphical_output,
        loglevel=options.loglevel
        )
    #else :
    #    parser.print_help()

#===========================================================================
