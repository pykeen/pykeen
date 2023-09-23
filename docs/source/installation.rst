Installation
============
Linux and Mac Users
-------------------
The latest stable version of PyKEEN can be downloaded and installed from
`PyPI <https://pypi.org/project/pykeen>`_ with:

.. code-block:: bash

    $ pip install pykeen

The latest version of PyKEEN can be installed directly from the
source on `GitHub <https://github.com/pykeen/pykeen>`_ with:

.. code-block:: bash

    $ pip install git+https://github.com/pykeen/pykeen.git

Google Colab and Kaggle Users
-----------------------------
`Google Colab <https://colab.research.google.com>`_ and `Kaggle <https://www.kaggle.com>`_ both provide
a hosted version of Google's custom Jupyter notebook environment that work similarly. After opening
a new notebook on one of these service, start your notebook with the following two lines:

.. code-block::

    ! pip install git+https://github.com/pykeen/pykeen.git
    pykeen.env()

This will install the latest code, then output relevant system and environment information with :func:`pykeen.env`.
It works because Jupyter interprets any line beginning with a bang ``!`` that the remainder of the
line should be interpreted as a bash command. If you want to make your notebook compatible on both
hosted and local installations, change it slightly to check if PyKEEN is already installed:

.. code-block::

    ! python -c "import pykeen" || pip install git+https://github.com/pykeen/pykeen.git
    pykeen.env()

.. note::

    Old versions of PyKEEN that used :mod:`class_resolve` version 0.3.4 and below loaded
    datasets via entrypoints. This was unpredictable on Kaggle and Google Colab, so it was
    removed in https://github.com/pykeen/pykeen/pull/832. More information can also be found
    on `PyKEEN issue #373 <https://github.com/pykeen/pykeen/issues/373>`_.

To enable GPU usage, go to the Runtime -> Change runtime type menu to enable a GPU with your notebook.

Windows Users
-------------
We've added experimental support for Windows as of `!95 <https://github.com/pykeen/pykeen/pull/95>`_.
However, be warned, it's much less straightforward to install PyTorch and therefore PyKEEN on Windows.

First, to install PyTorch, you must install `Anaconda <https://www.anaconda.com/>`_ and follow
the instructions on the `PyTorch website <https://pytorch.org/get-started/locally/>`_.
Then, assuming your `python` and `pip` command are linked to the same place where conda is installing,
you can proceed with the normal installation (or the installation from GitHub as shown above):

.. code-block:: bash

    $ pip install pykeen

If you're having trouble with ``pip`` or ``sqlite``, you might also have to use
``conda install pip setuptools wheel sqlite``. See our
`GitHub Actions configuration <https://github.com/pykeen/pykeen/blob/master/.github/workflows/tests.yml>`_
on GitHub for inspiration.

If you know better ways to install on Windows or would like to share some references,
we'd really appreciate it.

Development
-----------
The latest code can be installed in development mode with:

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

================  =========================================================================================
Name              Description
================  =========================================================================================
``templating``    Building of templated documentation, like the README
``plotting``      Plotting with ``seaborn`` and generation of word clouds
``mlflow``        Tracking of results with ``mlflow``
``wandb``         Tracking of results with ``wandb``
``neptune``       Tracking of results with ``neptune``
``tensorboard``   Tracking of results with :mod:`tensorboard` via :mod:`torch.utils.tensorboard`
``transformers``  Label-based initialization with ``transformers``.
``tests``         Code needed to run tests. Typically handled with ``tox -e py``
``docs``          Building of the documentation
``opt_einsum``    Improve performance of :func:`torch.einsum` by replacing with :func:`opt_einsum.contract`
``biomedicine``   Use of :mod:`pyobo` for lookup of biomedical entity labels
================  =========================================================================================
