
.. toctree::
   :maxdepth: 2

===========================
PyFACT Tutorial
===========================

---------------------------------
Skymaps with pfmap.py
---------------------------------

The script pfmap.py is used to create binned skymaps from FITS event lists. It calculates background maps using the ring and the template background method (if a corresponding template background eventlist is provided) and produces signal, background, excess, and significance maps. These maps can be written to fits files and then viewed and analyzed with standard fits tools, e.g., fv, ds9, or sherpa.

Using pfmap.py is straight forward. To create skymaps from a file data.fits using the object position from the header of the file as center of skymap and writing the skymaps  to FITS files (option: -w) use::

  $ python scripts/pfmap.py data.fits -w

pfmap.py can also handle compressed files (gzip).

If you want to analyse several files together, you need to create an ASCII file containing the filenames (first string per row is used; bankfile), e.g.,::

  $ ls *.fits.gz > my.bnk
  $ python scripts/pfmap.py my.bnk -w

You can change the parameters of the skymap via command line options, e.g.,::

  $ python scripts/pfmap.py my.bnk -w -s 4. -b 0.1 -r 0.1 -c "(83.633, 22.0145)"

creating a skymap of size 4 deg (-s) with a bin size 0.1 deg (-c) and correlation radius for the oversampled skymap of 0.1 deg (-r). The center of the map is set to RA=83.633 deg, Dec=22.0145 deg (J2000; -) . Check the --help option for more details on the different command line options.

After running pfmap.py with the option -w you will find a couple of new FITS files in you working directory starting with skymap_ring (or skymap_templ). Files containing the string overs contain correlated/oversampled maps. The other string identifier are as follow: ac = Acceptance;  al = Alpha factor;  bg = Background;  ev = Events;  ex = Excess;  si = Significance. You can view the files with  with standard fits tools, e.g., fv or ds9.

Find below an example python script, which shows to fit an excess skymap with a 2D double gaussian function with sherpa. For this to work it is assumed that you have the python packages sherpa, pyfits, and kapteyn installed on your machine. ::

  import sherpa.astro.ui as ui
  from kapteyn import wcs, positions
  import pyfits
  
  filename = 'skymap_ex.fits'
  nomposstr = '05h34m31.94s 22d00m52.2s'
  hdr = pyfits.getheader(filename)
  proj = wcs.Projection(hdr)
  xc, yc = float(hdr['NAXIS1']) / 2., float(hdr['NAXIS2']) / 2.
  ui.load_image(filename)
  ui.notice2d('circle({0}, {1}, {2})'.format(xc, yc, float(hdr['NAXIS2']) / 4.))
  ui.set_source(ui.gauss2d.g1 + ui.gauss2d.g2)
  g1.xpos = xc
  g1.ypos = yc
  g2.fwhm = g1.fwhm = 3.
  ui.link(g2.xpos, g1.xpos)
  ui.link(g2.ypos, g1.ypos)
  g2.ampl = 50.
  g1.ampl = 50.
  ui.guess()
  ui.fit()
  ui.image_fit()
  ui.covar()
  conf = ui.get_covar_results()
  conf_dict = dict([(n,(v, l, h)) for n,v,l,h in
                     zip(conf.parnames, conf.parvals, conf.parmins, conf.parmaxes)])
  x, y = proj.toworld((conf_dict['g1.xpos'][0], conf_dict['g1.ypos'][0]))
  xmin, ymin = proj.toworld((conf_dict['g1.xpos'][0] + conf_dict['g1.xpos'][1],
                             conf_dict['g1.ypos'][0] + conf_dict['g1.ypos'][1]))
  xmax, ymax = proj.toworld((conf_dict['g1.xpos'][0] + conf_dict['g1.xpos'][2],
                             conf_dict['g1.ypos'][0] + conf_dict['g1.ypos'][2]))
  nompos = positions.str2pos(nomposstr, proj)    
  print '{0} ({1}-{2}) vs {3}'.format(x, xmin, xmax, nompos[0][0][0])
  print '{0} ({1}-{2}) vs {3}'.format(y, ymin, ymax, nompos[0][0][1])

------------------------------------
Spectra with pfspec
------------------------------------

The script pfspec produces pha spectra files from FITS event lists, which can be analyzed with tools like xspec. The instrument response is taken from ARF (effective area) and RMF (energy distribution matrix) files and is assumed to be constant over the duration of a data segment (run). The background is estimated using a ring at the same camera/FoV distance as the source, cutting out the source position.

Per data file, pfspec needs three inputs: the name of the data file and the corresponding ARF and RMF file names. These can be given via command line but usually it is more efficient to create an ASCII file (bankfile), with each row giving the data file name, the ARF and the RMF file names, separate by spaces. We assume, such a bankfile has been created for the data called my.bnk.

To create the pha files run::

  $ python scripts/pfspec.py my.bnk -w -r 0.125

The option -r} denotes the radius in degree of the circular source region from which the spectrum will be extracted (theta cut). This should match the cut used in the ARF files. This will produce three pha} files per data file in the working directory: bg = Background; excess = Excess; signal = Signal (i.e. excess = signal - background).

The output pha files can be analyzed with spectra fitting tools like xspec or sherpa. Find below an example session for xspec. Note that xspec and sherpa do not recognize the units given in the ARF/RMF files correctly, always assuming keV and cm^2. Therefore, the fit results have to be converted correspondingly. Some output has been omitted below and has been replaced with dots (...).::


   $ xspec
  
  		XSPEC version: 12.7.0
  	Build Date/Time: Tue Jun  7 23:17:42 2011
  
  XSPEC12>cpd /xw
  XSPEC12>data CTA1DC-HESS-run_00023523_eventlist_signal.pha.fits \
  CTA1DC-HESS-run_00023526_eventlist_signal.pha.fits
  
  2 spectra  in use
   
  Spectral Data File: CTA1DC-HESS-run_00023523_eventlist_signal.pha.fits  Spectrum 1
  Net count rate (cts/s) for Spectrum:1  1.174e-01 +/- 9.183e-03 (88.8 % total)
   Assigned to Data Group 1 and Plot Group 1
    Noticed Channels:  1-33
    Telescope: unknown Instrument: unknown  Channel Type: PHA
    Exposure Time: 1582 sec
   Using fit statistic: chi
   Using Background File                CTA1DC-HESS-run_00023523_eventlist_bg.pha.fits
    Background Exposure Time: 1582 sec
   Using Response (RMF) File            run023523_dummy_s0.1.rmf.fits for Source 1
   Using Auxiliary Response (ARF) File  CTA1DC-HESS-run023523_std_arf.fits.gz
  
  ...
  
  XSPEC12>plot data
  XSPEC12>model powerlaw
  
  Input parameter value, delta, min, bot, top, and max values for ...
                1       0.01(      0.01)         -3         -2          9         10
  1:powerlaw:PhoIndex>2.7
                1       0.01(      0.01)          0          0      1e+24      1e+24
  2:powerlaw:norm>1E-8
  
  ========================================================================
  Model powerlaw<1> Source No.: 1   Active/On
  Model Model Component  Parameter  Unit     Value
   par  comp
     1    1   powerlaw   PhoIndex            2.70000      +/-  0.0          
     2    1   powerlaw   norm                1.00000E-08  +/-  0.0          
  ________________________________________________________________________
  
  
   Chi-Squared =         336.01 using 67 PHA bins.
   Reduced chi-squared =         5.1694 for     65 degrees of freedom 
   Null hypothesis probability =   3.604967e-38
   Current data and model not fit yet.
  XSPEC12>fit
  
  ...
  
  ========================================================================
  Model powerlaw<1> Source No.: 1   Active/On
  Model Model Component  Parameter  Unit     Value
   par  comp
     1    1   powerlaw   PhoIndex            2.58471      +/-  9.17193E-02  
     2    1   powerlaw   norm                3.49673E-07  +/-  2.17838E-08  
  ________________________________________________________________________
  
  
   Chi-Squared =          76.55 using 67 PHA bins.
   Reduced chi-squared =          1.178 for     65 degrees of freedom 
   Null hypothesis probability =   1.548459e-01
  XSPEC12>plot data delchi
