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
#import exception
import datetime

import numpy as np
import pyfits
import scipy
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

def D(s,*f) :
    if len(f) :
        logging.debug(s.format(*f))
    else :
        logging.debug(s)

#===========================================================================
# Main
def sim_evlist(flux=.1,
               obstime=.5,
               arf=None,
               rmf=None,
               extra=None,
               output_filename_base=None,
               write_pha=False,
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
    logging.info('This is {0} (pyfact v{1})'.format(os.path.split(__file__)[1], pf.__version__))
    logging.info('We are running with numpy v{0}, scipy v{1}, and pyfits v{2}'.format(
        np.__version__, scipy.__version__, pyfits.__version__
        ))

    #---------------------------------------------------------------------------

    logging.info('Exposure: {0} h'.format(obstime))
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
    # Read ARF, RMF, and extra file

    logging.info('ARF: {0}'.format(arf))
    ea, ea_erange = pf.arf_to_np(pyfits.open(arf)[1])

    # DEBUG
    #ea /= irf_data[:,4]
    
    rm, rm_erange, rm_ebounds, rm_minprob = None, ea_erange, None, None
    if rmf :
        logging.info('RMF: {0}'.format(rmf))
        rm, rm_erange, rm_ebounds, rm_minprob = pf.rmf_to_np(pyfits.open(rmf))

    if extra :
        logging.info('Extra file: {0}'.format(extra))
        extraf = pyfits.open(extra)
        logging.info('Using effective area with 80% containment from extra file')
        ea = extraf['EA80'].data.field('VAL') / .8 # 100% effective area
        ea_erange = 10. ** np.hstack([extraf['EA80'].data.field('BIN_LO'), extraf['EA80'].data.field('BIN_HI')[-1]])
    else :
        logging.info('Assuming energy dependent 90% cut efficiency')
        ea /= .9
        
    #---------------------------------------------------------------------------
    # Signal

    #log_e_cen = (irf_data[:,0] + irf_data[:,1]) / 2.
    e_cen = 10. ** ((np.log10(ea_erange[1:]*ea_erange[:-1])) / 2.)

    ea_loge_step_mean = np.log10(ea_erange[1:]/ea_erange[:-1]).mean().round(4)
    D('ea_loge_step_mean = {0}', ea_loge_step_mean)

    # DEBUG
    #ea_s, e_cen_s = ea, e_cen

    # Resample effective area to increase precision
    if ea_loge_step_mean > .1 :
        elog10step = .05
        logging.info('Resampling effective area in log10(EA) vs log10(E) (elog10step = {0})'.format(elog10step))
        ea_spl = scipy.interpolate.UnivariateSpline(e_cen, np.log10(ea), s=0, k=1)
        e_cen = 10. ** np.arange(np.log10(e_cen[0]), np.log10(e_cen[-1]), step=elog10step)
        ea = 10. ** ea_spl(e_cen)
    
    # DEBUG plot
    #plt.loglog(e_cen_s, ea_s, )
    #plt.loglog(e_cen, ea, '+')
    #plt.show()

    func_pl = lambda x, p: p[0] * x ** (-p[1])
    flux_f =  lambda x : func_pl(x, (3.45E-11 * flux, 2.63))

    f_test = scipy.interpolate.UnivariateSpline(e_cen, ea * flux_f(e_cen) * 1E4, s=0, k=1) # m^2 > cm^2

    log_e_steps = np.log10(rm_erange)

    # Calculate event numbers for the RMF bins
    def get_int_rate(emin, emax) :
        if emin < e_cen[0] or emax > e_cen[-1] :
            return 0.
        else :
            return f_test.integral(emin, emax)
    int_rate = np.array([get_int_rate(10. ** el, 10. ** eh) for (el, eh) in zip(log_e_steps[:-1], log_e_steps[1:])])
    # Sanity
    int_rate[int_rate < 0.] = 0.

    # DEBUG
    #int_rate_s = int_rate

    if rmf :
        D('Photon rate before RM = {0}', np.sum(int_rate))
        # Apply energy distribution matrix
        int_rate = np.dot(int_rate, rm)
        D('Photon rate after RM = {0}', np.sum(int_rate))

    # DEBUG plots
    #plt.figure(1)
    #plt.semilogy(log_e_steps[:-1], int_rate_s, 'o', label='PRE RMF')
    ##plt.plot(log_e_steps[:-1], int_rate_s, 'o', label='PRE RMF')
    #if rmf :
    #    plt.semilogy(np.log10(rm_ebounds[:-1]), int_rate, '+', label='POST RMF')
    #plt.ylim(1E-6,1.)
    #plt.legend()
    #plt.show()
    ##sys.exit(0)

    # Calculate cumulative event numbers
    int_all = np.sum(int_rate)
    int_rate = np.cumsum(int_rate)

    if rmf :
        #log_e_steps = (np.log10(rm_ebounds[1:]) + np.log10(rm_ebounds[:-1])) / 2.
        log_e_steps = np.log10(rm_ebounds)

    # Filter out low and high values to avoid spline problems at the edges
    istart = np.sum(int_rate == 0.) - 1
    if istart < 0 :
        istart = 0
    istop = np.sum(int_rate / int_all > 1. - 1e-4) # This value dictates the dynamic range at the high energy end

    D('istart = {0}, istop = {1}', istart, istop)

    # DEBUG plots
    #plt.plot(int_rate[istart:-istop] / int_all, log_e_steps[istart + 1:-istop], '+')
    #plt.show()

    # DEBUG plots
    #plt.hist(int_rate[istart:-istop] / int_all)
    #plt.show()

    ev_gen_f = scipy.interpolate.UnivariateSpline(
        int_rate[istart:-istop] / int_all,
        log_e_steps[istart + 1:-istop],
        s=0,
        k=1
        )

    ## DEBUG plot
    #plt.plot(np.linspace(0.,1.,100), ev_gen_f(np.linspace(0.,1.,100)), 'o')
    #plt.show()

    # Test event generator function
    n_a_t = 100.
    a_t = ev_gen_f(np.linspace(0.,1., n_a_t))
    D('Test ev_gen_f, (v = 0 / #v) = {0}, (v = NaN / #v) = {1}',
      np.sum(a_t == 0.) / n_a_t, np.sum(np.isnan(a_t)) / n_a_t)

    if (np.sum(a_t == 0.) / n_a_t > 0.05) or (np.sum(np.isnan(a_t)) / n_a_t > .05) :
        raise Exception('Could not generate event generator function for photons. Try to decrease the upper cut-off value in the code.')

    # Calculate total number of photon events
    n_events = int_all * obstime

    logging.debug('Number of photons : {0}'.format(n_events))

    # Generate energy event list
    evlist_e = ev_gen_f(np.random.rand(n_events))
    
    """
    # OLD OLD
    log_e_cen = (irf_data[:,0] + irf_data[:,1]) / 2.

    func_pl = lambda x, p: p[0] * x ** (-p[1])

    e_bin_w = 10. ** irf_data[:,1] - 10. ** irf_data[:,0]
    pho_rate = func_pl(10. ** log_e_cen, (1E-11, 3.)) * irf_data[:,7] * 1E4 * e_bin_w

    plt.loglog(10. ** log_e_cen, irf_data[:,7], '+')
    plt.show()

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

    #DEBUG plot
    plt.semilogy(np.linspace(-2.,2.,100), f_test(10. ** np.linspace(-2.,2.,100)))
    plt.show()

    logging.debug('Number of photons : {0}'.format(n_events))

    # DEBUG plot
    plt.hist(evlist_e, range=[-2.,2.], bins=40, histtype='step', label='arf')

    evlist_e = ev_gen_f(np.random.rand(n_events))
    #OLD ENDS
    plt.hist(evlist_e, range=[-2.,2.], bins=40, histtype='step', label='txt')
    plt.legend()
    plt.show()
    """
    
    # Sanity
    D('Number of photons with E = NaN : {0}', np.sum(np.isnan(evlist_e)))
    evlist_e[np.isnan(evlist_e)] = 0.

    ## DEBUG plot
    #plt.figure(1)
    #plt.hist(evlist_e, range=[-2.,2.], bins=20)
    ##plt.show()
    #sys.exit(0)

    #------------------------------------------------------
    # Apply PSF

    # Broken power law fit function, normalized at break energy
    bpl = lambda p,x : np.where(x < p[0], p[1] * (x / p[0]) ** -p[2],  p[1] * (x / p[0]) ** -p[3])
    evlist_psf = None
    if extra :
        d = extraf['ANGRES68'].data
        g = scipy.interpolate.UnivariateSpline((d.field('BIN_LO') + d.field('BIN_HI')) / 2., d.field('VAL'), s=0, k=1)
        evlist_psf = g(evlist_e)
    else :
        psf_p1 = [1.1, 5.5E-2, .42, .19] # Fit from SubarrayE_IFAE_50hours_20101102
        evlist_psf = bpl(psf_p1, 10. ** evlist_e)
        logging.warning('Using dummy PSF extracted from SubarrayE_IFAE_50hours_20101102')

    # OLD OLD OLD
    #evlist_psf = np.where(evlist_e > 1., .05, 5.6E-2 * np.exp(-.9 * evlist_e))
    #evlist_psf = np.where(evlist_e > 1., .05, 3.3E-2 * np.exp(-1.4 * evlist_e))    
    
    evlist_dec = obj_dec + np.random.randn(n_events) * evlist_psf
    evlist_ra =  obj_ra + np.random.randn(n_events) * evlist_psf / objcosdec

    evlist_t = t_min + obstime * np.random.rand(n_events) / 86400.

    #---------------------------------------------------------------------------
    # Background

    #plt.figure(1)

    p_rate_area, log_e_cen = None, None
    if extra :
        d = extraf['BGRATED'].data
        p_rate_area = d.field('VAL')
        log_e_cen = (d.field('BIN_LO') + d.field('BIN_HI')) / 2
        #g = scipy.interpolate.UnivariateSpline((d.field('BIN_LO') + d.field('BIN_HI')) / 2., d.field('VAL'), s=0, k=1)
    else :        
        logging.warning('Using dummy background rate extracted from SubarrayE_IFAE_50hours_20101102')
        bgrate_p1 = [9., 5.E-4, 1.44, .49] # Fit from SubarrayE_IFAE_50hours_20101102
        log_e_cen = np.linspace(-1.5, 2., 35.)
        p_rate_area = bpl(bgrate_p1, 10. ** log_e_cen)
        p_rate_area[log_e_cen < -1.] = .6

    # DEBUG plot
    #plt.semilogy(log_e_cen, p_rate_area)
    #plt.show()

    #log_e_cen = (irf_data[:,0] + irf_data[:,1]) / 2.
    # Protons + electron
    #p_rate_area = (irf_data[:,10] + irf_data[:,11]) / irf_data[:,5] / np.pi

    #logging.debug('Total proton rate = {0}'.format(np.sum(irf_data[:,10])))

    p_rate_total = np.sum(p_rate_area)

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

    logging.debug('Number of protons : {0}'.format(n_events_bg))

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
    logging.debug('Number of BG events in a circle of area 1 deg^2 = {0}'.format(np.sum(test_r[0:n_events_bg] < np.sqrt(1. / np.pi))))
    logging.debug('Expected number of BG event per area 1 deg^2 = {0}'.format(p_rate_total * obstime))

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
    
    if output_filename_base:
        logging.info('Writing eventlist to file {0}.eventlist.fits'.format(output_filename_base))
        
        newtable = pyfits.new_table(
            pyfits.ColDefs([
                pyfits.Column(name='TIME', format='1E', unit='deg', array=np.append(evlist_t, evlist_bg_t)),
                pyfits.Column(name='RA', format='1E', unit='deg', array=np.append(evlist_ra, evlist_bg_ra)),
                pyfits.Column(name='DEC', format='1E', unit='deg', array=np.append( evlist_dec, evlist_bg_dec)),
                pyfits.Column(name='DETX', format='1E', unit='deg', array=np.append(np.zeros(n_events), evlist_bg_rx)),
                pyfits.Column(name='DETY', format='1E', unit='deg', array=np.append(np.ones(n_events) * .5, evlist_bg_ry)),
                pyfits.Column(name='ENERGY', format='1E', unit='tev', array=10. ** np.append(evlist_e,evlist_bg_e)),
                pyfits.Column(name='HIL_MSW', format='1E', array=np.append(np.zeros(n_events + n_events_bg),
                                                                           5. * np.ones(n_events_bg * tplt_multi))),
                pyfits.Column(name='HIL_MSL', format='1E', array=np.append(np.zeros(n_events + n_events_bg),
                                                                           5. * np.ones(n_events_bg * tplt_multi))) 
                ])
            )
        #newtable = pyfits.new_table(coldefs_new)
        dstart = datetime.datetime(2011, 1, 1, 0, 0) # date/time of start of the first observation
        dstop = dstart + datetime.timedelta(seconds=obstime) # date/time of end of the last observation
        dbase = datetime.datetime(2011, 1, 1)

        hdr = newtable.header
        
        hdr.update('RA_OBJ' , obj_ra, 'Target position RA [deg]')
        hdr.update('DEC_OBJ', obj_dec, 'Target position dec [deg]')
        hdr.update('RA_PNT', pnt_ra, 'Observation position RA [deg]')
        hdr.update('DEC_PNT', pnt_dec, 'Observation position dec [deg]')
        hdr.update('EQUINOX ', 2000.0, 'Equinox of the object')
        hdr.update('RADECSYS', 'FK5', 'Co-ordinate frame used for equinox')
        hdr.update('CREATOR', 'pfsim v{0}'.format(pf.__version__) , 'Program')
        hdr.update('DATE', datetime.datetime.today().strftime('%Y-%m-%d'), 'FITS file creation date (yyyy-mm-dd)')
        hdr.update('TELESCOP', 'CTASIM', 'Instrument name')
        hdr.update('EXTNAME', 'EVENTS' , 'HESARC standard')
        hdr.update('DATE_OBS', dstart.strftime('%Y-%m-%d'), 'Obs. start date (yy-mm-dd)')
        hdr.update('TIME_OBS', dstart.strftime('%H:%M:%S'), 'Obs. start time (hh:mm::ss)')
        hdr.update('DATE_END', dstop.strftime('%Y-%m-%d'), 'Obs. stop date (yy-mm-dd)')
        hdr.update('TIME_END', dstop.strftime('%H:%M:%S'), 'Obs. stop time (hh:mm::ss)')
        hdr.update('TSTART', 0., 'Mission time of start of obs [s]')
        hdr.update('TSTOP', obstime, 'Mission time of end of obs [s]')
        hdr.update('MJDREFI', int(pf.date_to_mjd(dstart)), 'Integer part of start MJD [s] ')
        hdr.update('MJDREFF', pf.date_to_mjd(dstart) - int(pf.date_to_mjd(dstart)), 'Fractional part of start MJD')
        hdr.update('TIMEUNIT', 'days' , 'Time unit of MJD')
        hdr.update('TIMESYS', 'TT', 'Terrestrial Time')
        hdr.update('TIMEREF', 'local', '')
        hdr.update('TELAPSE', obstime, 'Diff of start and end times')
        hdr.update('ONTIME', obstime, 'Tot good time (incl deadtime)') # No deadtime assumed
        hdr.update('LIVETIME', obstime, 'Deadtime=ONTIME/LIVETIME') # No deadtime assumed
        hdr.update('DEADC', 1., 'Deadtime fraction') # No deadtime assumed
        hdr.update('TIMEDEL', 1., 'Time resolution')
        hdr.update('EUNIT', 'TeV', 'Energy unit')
        hdr.update('EVTVER', 'v1.0.0', 'Event-list version number')
        #hdr.update('', , '')

        # Write eventlist to file
        newtable.writeto('{0}.eventlist.fits'.format(output_filename_base))

        if write_pha :
            logging.info('Writing PHA to file {0}.pha.fits'.format(output_filename_base))
            # Prepare data
            dat, t = np.histogram(10. ** evlist_e, bins=rm_ebounds)
            dat = np.array(dat, dtype=float)
            dat_err = np.sqrt(dat)
            chan = np.arange(len(dat))
            # Data to PHA
            tbhdu = pf.np_to_pha(counts=dat, stat_err=dat_err, channel=chan, exposure=obstime, obj_ra=obj_ra, obj_dec=obj_dec,
                                 quality=np.where((dat == 0), 1, 0),
                                 dstart=dstart, dstop=dstop, dbase=dbase, creator='pfsim', version=pf.__version__,
                                 telescope='CTASIM')
            tbhdu.header.update('ANCRFILE', os.path.basename(arf), 'Ancillary response file (ARF)')
            if rmf :
                tbhdu.header.update('RESPFILE', os.path.basename(rmf), 'Redistribution matrix file (RMF)')

            # Write PHA to file
            tbhdu.writeto('{0}.pha.fits'.format(output_filename_base))

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
        usage='%prog <arf_file_name> [options]',
        description='Simulates IACT eventlist using an ARF file.'
    )
    parser.add_option(
        '-t','--exposure-time',
        dest='exposure_time',
        type='float',
        default=.5,
        help='Exposure time in hours [default: %default].'
    )
    parser.add_option(
        '-f','--flux',
        dest='flux',
        type='float',
        default=.1,
        help='Flux in units of Crab [default: %default].'
    )
    parser.add_option(
        '-r','--rmf-file',
        dest='rmf',
        type='string',
        default=None,
        help='Response matrix file (RMF), optional [default: %default].'
    )
    parser.add_option(
        '-e','--extra-file',
        dest='extra',
        type='string',
        default=None,
        help='Extra file with auxiliary information e.g. bg-rate, psf, etc. [default: %default].'
    )
    parser.add_option(
        '-o','--output-filename-base',
        dest='output_filename_base',
        type='string',
        default=None,
        help='Output filename base. If set, output files will be written [default: %default].'
    )
    parser.add_option(
        '--write-pha',
        dest='write_pha',
        action='store_true',
        default=False,
        help='Write photon PHA file [default: %default].'
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

    if len(args) is 1 :
        sim_evlist(
            flux=options.flux,
            obstime=options.exposure_time,
            arf=args[0],
            rmf=options.rmf,
            extra=options.extra,
            output_filename_base=options.output_filename_base,
            write_pha=options.write_pha,
            do_graphical_output=options.graphical_output,
            loglevel=options.loglevel
            )
    else :
        parser.print_help()

#===========================================================================
