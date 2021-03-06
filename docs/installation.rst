
.. toctree::
   :maxdepth: 2

==============
Installation
==============

There is no compilation or installation of PyFACT required, simply download the package from https://github.com/mraue/pyfact (follow the Download link in the upper right) or via command line tools, e.g.::

    $ curl -L  https://github.com/mraue/pyfact/tarball/master > mraue-pyfact.tar.gz
    or
    $ wget https://github.com/mraue/pyfact/tarball/master

and then untar the package ::

    $ tar xzvf mraue-pyfact.tar.gz
    or
    $ tar xzvf master

If you have git installed, you can also clone the repository: ::

    $ git clone git://github.com/mraue/pyfact.git

You will find the (executable) scripts, e.g., ``pfmap.py`` and ``pfspec.py``, in the ``script/`` folder. They can be run from the command line, e.g. ::

    $ python scripts/pfmap.py --help

or from the python or ipython interactive shell.

Check the ``README`` file for latest updates etc.

The master branch always holds the latest stable version. If you are
working on PyFACT development download the develop branch::

    $ wget https://github.com/mraue/pyfact/tarball/develop

