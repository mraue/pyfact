+-------------------------------------------------+
|  [Py]thon &                                     |
|  [F]ITS                                         |
|  [A]nalysis for                     < v0.0.1 >  |
|  [C]herenkov                      Jun 15, 2011  |
|  [T]elescopes               www.desy.de/pyfact  |
+-------------------------------------------------+

============
INTRODUCTION

PyFACT is a collection of python tools for the analysis of Imaging Atmospheric Cherenkov Telescope (IACT) data in FITS format. It is developed and maintained by XXX (XXX@XXX).

============
REQUIREMENTS

The module is written for Python 2.6 and requires the following packages:

* numpy 1.5.1 (1.6 is currently incompatible with pyfits)
  http://numpy.scipy.org/

* scipy 0.9.0
  http://www.scipy.org/

* matplotlib 1.0.1
  http://matplotlib.sourceforge.net/

* pyfits 2.4.0
  http://www.stsci.edu/resources/software_hardware/pyfits/

The data needs to be in CTA FITS format v1.0.0 or higher.
<download link>

============
INSTALLATION

No installation of the package is required. Simply untar the archive and you are good to go.

=====
USAGE

Tasks are performed using the python scripts located in the pyfact-vX.X.X/scripts folder. The scripts are excecutable python scripts. To execute them simply type something like:

./scripts/pfmap.py

If you prefer to run them with a specific python version, you can also execute them as normal python scripts:

my_python ./scripts/pfmap.py

They can also run inside an interactive session in e.g. ipython:

In [0]: %run ./scripts/pfmap.py

The scripts come with a builtin help text, which is displayed when the scripts are called without any argument or with the -h/--help option:

./scripts/pfmap.py
./scripts/pfmap.py -h
./scripts/pfmap.py --help


==================
FEEDBACK & SUPPORT

========
LISCENSE

end.
