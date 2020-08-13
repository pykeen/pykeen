Installation
============
The latest stable version of PyKEEN can be downloaded and installed from
`PyPI <https://pypi.org/project/pykeen>`_ with:

.. code-block:: bash

    $ pip install pykeen

The latest version of PyKEEN can be installed directly from the
source on `GitHub <https://github.com/pykeen/pykeen>`_ with:

.. code-block:: bash

    $ pip install git+https://github.com/pykeen/pykeen.git

Alternatively, the latest code can be installed in development mode
with:

.. code-block:: bash

    $ git clone https://github.com/pykeen/pykeeen.git pykeen
    $ cd pykeen
    $ pip install -e .

If you're interested in making contributions, please see our
`contributing guide <https://github.com/pykeen/pykeen/blob/master/CONTRIBUTING.md>`_.

To automatically ensure compliance to our style guide, please install pre-commit
hooks using the following code block from in the same directory.

.. code-block:: bash

    $ pip install pre-commit
    $ pre-commit install

Extras
------
PyKEEN has several extras for installation that are defined in the ``[options.extras_require]`` section
of the ``setup.cfg``. They can be included with installation using the bracket notation like in 
``pip install pykeen[docs]`` or ``pip install -e .[docs]``. Several can be listed, comma-delimited like in
``pip install pykeen[docs,plotting]``.

==============  =======================================================
Name            Description
==============  =======================================================
``plotting``    Plotting with ``seaborn`` and generation of word clouds
``mlflow``      Tracking of results with ``mlflow``
``wandb``       Tracking of results with ``wandb``
``docs``        Building of the documentation
``templating``  Building of templated documentation, like the README
==============  =======================================================
