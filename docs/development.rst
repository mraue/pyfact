.. toctree::
   :maxdepth: 2

===========================
Development guidelines
===========================

---------------------------------
Code documentation
---------------------------------

The code is document using the autodocumentation feature from sphinx:

http://sphinx.pocoo.org/tutorial.html

The syntax follows the numpy documentation style as described in :doc:`code-doc_np_HOWTO_DOCUMENT` (copied from https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt).

The npdoc sphinx extension needed to parse the inline comments comes bundled with PyFACT (docs/npdoc).

---------------------------------
Branching model
---------------------------------

The branching model follow the "successful-git-branching-model" from nvie.com:

http://nvie.com/posts/a-successful-git-branching-model/

The major difference to many other models is that the ``master`` branch always hold the latest stable release version (including all bugfixes). Development is performed in a ``develop`` branch. Every feature is developed in its own branch, branched off the ``develop`` branch. When the feature is ready it is merged into the ``develop`` branch. Releases are staged from feature freezed ``release`` branch, branched of ``develop``. ``release`` branches are, after testing, tagged and merged into ``master``.

^^^^^^^^^^^^^^^^^^^^^^^^^
Creating a feature branch
^^^^^^^^^^^^^^^^^^^^^^^^^

When starting work on a new feature, branch off from the develop branch::

    $ git checkout -b myfeature develop

Switched to a new branch "myfeature"

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Incorporating a finished feature on develop
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Finished features may be merged into the develop branch definitely add them to the upcoming release::

    $ git checkout develop
    Switched to branch 'develop'
    $ git merge --no-ff myfeature
    Updating ea1b82a..05e9557
    (Summary of changes)
    $ git branch -d myfeature
    Deleted branch myfeature (was 05e9557).
    $ git push origin develop

^^^^^^^^^^^^^^^^^^^^^^^^^
Creating a release branch
^^^^^^^^^^^^^^^^^^^^^^^^^

::

    $ git checkout -b release-1.2 develop
    Switched to a new branch "release-1.2"
    $ ./bump-version.sh 1.2
    # Change release version / dummy script
    Files modified successfully, version bumped to 1.2.
    $ git commit -a -m "Bumped version number to 1.2"
    [release-1.2 74d9424] Bumped version number to 1.2
    1 files changed, 1 insertions(+), 1 deletions(-)

^^^^^^^^^^^^^^^^^^^^^^^^^^
Finishing a release branch
^^^^^^^^^^^^^^^^^^^^^^^^^^

::

   $ git checkout master
   Switched to branch 'master'
   $ git merge --no-ff release-1.2
   Merge made by recursive.
   (Summary of changes)
   $ git tag -a 1.2
   
---------------------------------
External contribution
---------------------------------

For external contribution we advocate the forked repo/pull request approach:

http://help.github.com/fork-a-repo/

http://astropy.readthedocs.org/en/latest/development/workflow/development_workflow_advanced.html
