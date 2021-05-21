Installation
=============

Multyscale is a true Python package, but not (yet) published on PyPI [#fn_PyPI]_.
For now, multyscale can be retrieved primarily from the `GitHub repository`_

.. code-block:: bash

  git clone git@github.com:computational-psychology/multyscale.git

After downloading the repository, it can be installed (from the main directory):

.. tabs::

   .. tab:: pip

      .. code-block:: python

         pip install .

      This will install the package in your local Python packages,
      after which the cloned repository can be removed
      without removing the installed package.

      To remove the installed package completely, using pip

      .. code-block:: python

         pip uninstall multyscale

   .. tab:: pip, for developers

      .. code-block:: python
      
         pip install -e .

      This will only create a link to the local repository,
      so that changes to the files in this repository are reflected in Python
      without the necessity of reinstalling the package.

      As a result, removing the repository removes the packages,
      but it's also still a good idea to run

      .. code-block:: python

         pip uninstall multyscale


.. _GitHub repository: https://github.com/computational-psychology/multyscale
.. [#fn_PyPI] This will be fixed in an upcoming update.

.. TODO: add conda installation
