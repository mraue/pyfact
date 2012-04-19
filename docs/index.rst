.. pyfact documentation master file, created by
   sphinx-quickstart on Thu Mar  1 15:16:15 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyFACT - Python and FITS Analysis for Cherenkov Telescopes
============================================================

Index
--------------

.. toctree::
   :maxdepth: 2

   requirements
   installation
   tutorial
   development
   tools
   fits
   map


Overview
----------------

PyFACT is a collection of python tools for the analysis of Imaging
Atmospheric Cherenkov Telescope (IACT) data in the eventlist FITS
format. Currently it is developed and maintained by Martin Raue
(martin.raue@desy.de) and Christoph Deil
(christoph.deil@mpi-hd.mpg.de) and distributed under the modified BSD
license (see LICENSE).

PyFACT is not (yet) a full fledged analysis package. It is collection of tools, which allow for a quick look into the data and which interface with existing tools like xspec or sherpa.

PyFACT can be downloaded from github: https://github.com/mraue/pyfact

Sub-modules
-----------------

The module is splitted in several sub-modules according to the functionality for better maintance.

:doc:`tools`
  General tools and helpers
:doc:`fits`
  Functions to deal with input/output in fits format
:doc:`map`
  Functions to deal with the creation of skymaps

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

