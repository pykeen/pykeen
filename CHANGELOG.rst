Change Log
==========
All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <http://keepachangelog.com/>`_
and this project adheres to `Semantic Versioning <http://semver.org/>`_

`Unreleased <https://github.com/pykeen/pykeen/compare/v1.5.0...HEAD>`_
-----------------------------------------------------------------------
Added
~~~~~
- Tutorial in using checkpoints when bringing your own data (https://github.com/pykeen/pykeen/pull/498)

`1.5.0 <https://github.com/pykeen/pykeen/compare/v1.4.0...v1.5.0>`_ - 2021-06-13
--------------------------------------------------------------------------------
New Metrics
~~~~~~~~~~~
- Adjusted Arithmetic Mean Rank Index (https://github.com/pykeen/pykeen/pull/378)
- Add harmonic, geometric, and median rankings (https://github.com/pykeen/pykeen/pull/381)

New Trackers
~~~~~~~~~~~~
- Console Tracker (https://github.com/pykeen/pykeen/pull/440)
- Tensorboard Tracker (https://github.com/pykeen/pykeen/pull/416; thanks @sbonner0)

New Models
~~~~~~~~~~
- QuatE (https://github.com/pykeen/pykeen/pull/367)
- CompGCN (https://github.com/pykeen/pykeen/pull/382)
- CrossE (https://github.com/pykeen/pykeen/pull/467)
- Reimplementation of LiteralE with arbitrary combination (g) function (https://github.com/pykeen/pykeen/pull/245)

New Negative Samplers
~~~~~~~~~~~~~~~~~~~~~
- Pseudo-typed Negative Sampler (https://github.com/pykeen/pykeen/pull/412)

Datasets
~~~~~~~~
- Removed invalid datasets (OpenBioLink filtered sets; https://github.com/pykeen/pykeen/pull/https://github.com/pykeen/pykeen/pull/439)
- Added WK3k-15K (https://github.com/pykeen/pykeen/pull/403)
- Added WK3l-120K (https://github.com/pykeen/pykeen/pull/403)
- Added CN3l (https://github.com/pykeen/pykeen/pull/403)

Added
~~~~~
- Documentation on using PyKEEN in Google Colab and Kaggle (https://github.com/pykeen/pykeen/pull/379,
  thanks `@jerryIsHere <https://github.com/jerryIsHere>`_)
- Pass custom training loops to pipeline (https://github.com/pykeen/pykeen/pull/334)
- Compatibility later for the fft module (https://github.com/pykeen/pykeen/pull/288)
- Official Python 3.9 support, now that PyTorch has it (https://github.com/pykeen/pykeen/pull/223)
- Utilities for dataset analysis (https://github.com/pykeen/pykeen/pull/16, https://github.com/pykeen/pykeen/pull/392)
- Filtering of negative sampling now uses a bloom filter by default (https://github.com/pykeen/pykeen/pull/401)
- Optional embedding dropout (https://github.com/pykeen/pykeen/pull/422)
- Added more HPO suggestion methods and docs (https://github.com/pykeen/pykeen/pull/446)
- Training callbacks (https://github.com/pykeen/pykeen/pull/429)
- Class resolver for datasets (https://github.com/pykeen/pykeen/pull/473)

Updated
~~~~~~~
- R-GCN implementation now uses new-style models and is super idiomatic (https://github.com/pykeen/pykeen/pull/110)
- Enable passing of interaction function by string in base model class (https://github.com/pykeen/pykeen/pull/384,
  https://github.com/pykeen/pykeen/pull/387)
- Bump scipy requirement to 1.5.0+
- Updated interfaces of models and negative samplers to enforce kwargs (https://github.com/pykeen/pykeen/pull/445)
- Reorganize filtering, negative sampling, and remove triples factory from most objects (
  https://github.com/pykeen/pykeen/pull/400, https://github.com/pykeen/pykeen/pull/405,
  https://github.com/pykeen/pykeen/pull/406, https://github.com/pykeen/pykeen/pull/409,
  https://github.com/pykeen/pykeen/pull/420)
- Update automatic memory optimization (https://github.com/pykeen/pykeen/pull/404)
- Flexibly define positive triples for filtering (https://github.com/pykeen/pykeen/pull/398)
- Completely reimplemented negative sampling interface in training loops (https://github.com/pykeen/pykeen/pull/427)
- Completely reimplemented loss function in training loops (https://github.com/pykeen/pykeen/pull/448)
- Forward-compatibility of embeddings in old-style models and updated docs on
  how to use embeddings (https://github.com/pykeen/pykeen/pull/474)

Fixed
~~~~~
- Regularizer passing in the pipeline and HPO (https://github.com/pykeen/pykeen/pull/345)
- Saving results when using multimodal models (https://github.com/pykeen/pykeen/pull/349)
- Add missing diagonal constraint on MuRE Model (https://github.com/pykeen/pykeen/pull/353)
- Fix early stopper handling (https://github.com/pykeen/pykeen/pull/419)
- Fixed saving results from pipeline (https://github.com/pykeen/pykeen/pull/428, thanks @kantholtz)
- Fix OOM issues with early stopper and AMO (https://github.com/pykeen/pykeen/pull/433)
- Fix ER-MLP functional form (https://github.com/pykeen/pykeen/pull/444)

`1.4.0 <https://github.com/pykeen/pykeen/compare/v1.3.0...v1.4.0>`_ - 2021-03-04
--------------------------------------------------------------------------------
New Datasets
~~~~~~~~~~~~
- Countries (https://github.com/pykeen/pykeen/pull/314)
- DB100K (https://github.com/pykeen/pykeen/issues/316)

New Models
~~~~~~~~~~
- MuRE (https://github.com/pykeen/pykeen/pull/311)
- PairRE (https://github.com/pykeen/pykeen/pull/309)
- Monotonic affine transformer (https://github.com/pykeen/pykeen/pull/324)

New Algorithms
~~~~~~~~~~~~~~
If you're interested in any of these, please get in touch with us
regarding an upcoming publication.

- Dataset Similarity (https://github.com/pykeen/pykeen/pull/294)
- Dataset Deterioration (https://github.com/pykeen/pykeen/pull/295)
- Dataset Remix (https://github.com/pykeen/pykeen/pull/296)

Added
~~~~~
- New-style models (https://github.com/pykeen/pykeen/pull/260) for direct usage of interaction
  modules
- Ability to train ``pipeline()`` using an Interaction module rather than a Model
  (https://github.com/pykeen/pykeen/pull/326, https://github.com/pykeen/pykeen/pull/330).

Changes
~~~~~~~
- Lookup of assets is now mediated by the ``class_resolver`` package (https://github.com/pykeen/pykeen/pull/321,
  https://github.com/pykeen/pykeen/pull/327)
- The ``docdata`` package is now used to parse structured information out of the model and dataset documentation
  in order to make a more informative README with links to citations (https://github.com/pykeen/pykeen/pull/303).

`1.3.0 <https://github.com/pykeen/pykeen/compare/v1.1.0...v1.3.0>`_ - 2021-02-15
--------------------------------------------------------------------------------
We skipped version 1.2.0 because we made an accidental release before this version
was ready. We're only human, and are looking into improving our release workflow
to live in CI/CD so something like this doesn't happen again. However, as an end user,
this won't have an effect on you.

New Datasets
~~~~~~~~~~~~
- CSKG (https://github.com/pykeen/pykeen/pull/249)
- DBpedia50 (https://github.com/pykeen/pykeen/issues/278)

New Trackers
~~~~~~~~~~~~
- General file-based Tracker (https://github.com/pykeen/pykeen/pull/254)
- CSV Tracker (https://github.com/pykeen/pykeen/pull/254)
- JSON Tracker (https://github.com/pykeen/pykeen/pull/254)

Fixed
~~~~~
- Fixed ComplEx's implementation (https://github.com/pykeen/pykeen/pull/313)
- Fixed OGB's reuse entity identifiers (https://github.com/pykeen/pykeen/pull/318, thanks @tgebhart)

Added
~~~~~
- ``pykeen version`` command for more easily reporting your environment in issues
  (https://github.com/pykeen/pykeen/issues/251)
- Functional forms of all interaction models (e.g., TransE, RotatE) (https://github.com/pykeen/pykeen/issues/238,
  `pykeen.nn.functional documentation <https://pykeen.readthedocs.io/en/latest/reference/nn/functional.html>`_). These
  can be generally reused, even outside of the typical PyKEEN workflows.
- Modular forms of all interaction models (https://github.com/pykeen/pykeen/issues/242,
  `pykeen.nn.modules documentation <https://pykeen.readthedocs.io/en/latest/reference/nn/modules.html>`_). These wrap
  the functional forms of interaction models and store hyper-parameters such as the ``p`` value for the L_p norm in
  TransE.
- The initializer, normalizer, and constrainer for the entity and relation embeddings are now exposed through the
  ``__init__()`` function of each KGEM class and can be configured. A future update will enable HPO on these as well
  (https://github.com/pykeen/pykeen/issues/282).

Refactoring and Future Preparation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This release contains a few big refactors. Most won't affect end-users, but if you're writing your own PyKEEN
models, these are important. Many of them are motivated to make it possible to introduce a new interface that makes
it much easier for researchers (who shouldn't have to understand the inner workings of PyKEEN) to make new models.

- The regularizer has been refactored (https://github.com/pykeen/pykeen/issues/266,
  https://github.com/pykeen/pykeen/issues/274). It no longer accepts a ``torch.device`` when instantiated.
- The ``pykeen.nn.Embedding`` class has been improved in several ways:
  - Embedding Specification class makes it easier to write new classes (https://github.com/pykeen/pykeen/issues/277)
  - Refactor to make shape of embedding explicit (https://github.com/pykeen/pykeen/issues/287)
  - Specification of complex datatype (https://github.com/pykeen/pykeen/issues/292)
- Refactoring of the loss model class to provide a meaningful class hierarchy
  (https://github.com/pykeen/pykeen/issues/256, https://github.com/pykeen/pykeen/issues/262)
- Refactoring of the base model class to provide a consistent interface (https://github.com/pykeen/pykeen/issues/246,
  https://github.com/pykeen/pykeen/issues/248, https://github.com/pykeen/pykeen/issues/253,
  https://github.com/pykeen/pykeen/issues/257). This allowed for simplification of the loss computation based on
  the new hierarchy and also new implementation of regularizer class.
- More automated testing of typing with MyPy (https://github.com/pykeen/pykeen/issues/255) and automated checking
  of documentation with ``doctests`` (https://github.com/pykeen/pykeen/issues/291)

Triples Loading
~~~~~~~~~~~~~~~
We've made some improvements to the ``pykeen.triples.TriplesFactory`` to facilitate loading even larger datasets
(https://github.com/pykeen/pykeen/issues/216). However, this required an interface change. This will affect any
code that loads custom triples. If you're loading triples from a path, you should now use:

.. code-block:: python

    path = ...

    # Old (doesn't work anymore)
    tf = TriplesFactory(path=path)

    # New
    tf = TriplesFactory.from_path(path)

Predictions
~~~~~~~~~~~
While refactoring the base model class, we excised the prediction functionality to a new module
``pykeen.models.predict`` (docs: https://pykeen.readthedocs.io/en/latest/reference/predict.html#functions).
We also renamed some of the prediction functions inside the base model to make them more consistent, but we now
recommend you use the functions from ``pykeen.models.predict`` instead.

- ``Model.predict_heads()`` -> ``Model.get_head_prediction_df()``
- ``Model.predict_relations()`` -> ``Model.get_head_prediction_df()``
- ``Model.predict_tails()`` -> ``Model.get_head_prediction_df()``
- ``Model.score_all_triples()`` -> ``Model.get_all_prediction_df()``

Fixed
~~~~~
- Do not create inverse triples for validation and testing factory (https://github.com/pykeen/pykeen/issues/270)
- Treat nonzero applied to large tensor error as OOM for batch size search (https://github.com/pykeen/pykeen/issues/279)
- Fix bug in loading ConceptNet (https://github.com/pykeen/pykeen/issues/290). If your experiments relied on this
  dataset, you should rerun them.

`1.1.0 <https://github.com/pykeen/pykeen/compare/v1.0.5...v1.1.0>`_ - 2021-01-20
--------------------------------------------------------------------------------
New Datasets
~~~~~~~~~~~~
- CoDEx (https://github.com/pykeen/pykeen/pull/154)
- DRKG (https://github.com/pykeen/pykeen/pull/156)
- OGB (https://github.com/pykeen/pykeen/pull/159)
- ConceptNet (https://github.com/pykeen/pykeen/pull/160)
- Clinical Knowledge Graph (https://github.com/pykeen/pykeen/pull/209)

New Trackers
~~~~~~~~~~~~
- Neptune.ai (https://github.com/pykeen/pykeen/pull/183)

Added
~~~~~
- Add MLFlow set tags function (https://github.com/pykeen/pykeen/pull/139; thanks @sunny1401)
- Add score_t/h function for ComplEx (https://github.com/pykeen/pykeen/pull/150)
- Add proper testing for literal datasets and literal models (https://github.com/pykeen/pykeen/pull/199)
- Checkpoint functionality (https://github.com/pykeen/pykeen/pull/123)
- Random triple generation (https://github.com/pykeen/pykeen/pull/201)
- Make negative sampler corruption scheme configurable (https://github.com/pykeen/pykeen/pull/209)
- Add predict with inverse tripels pipeline (https://github.com/pykeen/pykeen/pull/208)
- Add generalize p-norm to regularizer (https://github.com/pykeen/pykeen/pull/225)

Changed
~~~~~~~
- New harness for resetting parameters (https://github.com/pykeen/pykeen/pull/131)
- Modularize embeddings (https://github.com/pykeen/pykeen/pull/132)
- Update first steps documentation (https://github.com/pykeen/pykeen/pull/152; thanks @TobiasUhmann )
- Switched testing to GitHub Actions (https://github.com/pykeen/pykeen/pull/165 and
  https://github.com/pykeen/pykeen/pull/194)
- No longer support Python 3.6
- Move automatic memory optimization (AMO) option out of model and into
  training loop (https://github.com/pykeen/pykeen/pull/176)
- Improve hyper-parameter defaults and HPO defaults (https://github.com/pykeen/pykeen/pull/181
  and https://github.com/pykeen/pykeen/pull/179)
- Switch internal usage to ID-based triples (https://github.com/pykeen/pykeen/pull/193 and
  https://github.com/pykeen/pykeen/pull/220)
- Optimize triples splitting algorithm (https://github.com/pykeen/pykeen/pull/187)
- Generalize metadata storage in triples factory (https://github.com/pykeen/pykeen/pull/211)
- Add drop_last option to data loader in training loop (https://github.com/pykeen/pykeen/pull/217)

Fixed
~~~~~
- Whitelist support in HPO pipeline (https://github.com/pykeen/pykeen/pull/124)
- Improve evaluator instantiation (https://github.com/pykeen/pykeen/pull/125; thanks @kantholtz)
- CPU fallback on AMO (https://github.com/pykeen/pykeen/pull/232)
- Fix HPO save issues (https://github.com/pykeen/pykeen/pull/235)
- Fix GPU issue in plotting (https://github.com/pykeen/pykeen/pull/207)

`1.0.5 <https://github.com/pykeen/pykeen/compare/v1.0.4...v1.0.5>`_ - 2020-10-21
--------------------------------------------------------------------------------
Added
~~~~~
- Added testing on Windows with AppVeyor and documentation for installation on Windows
  (https://github.com/pykeen/pykeen/pull/95)
- Add ability to specify custom datasets in HPO and ablation studies (https://github.com/pykeen/pykeen/pull/54)
- Add functions for plotting entities and relations (as well as an accompanying tutorial)
  (https://github.com/pykeen/pykeen/pull/99)

Changed
~~~~~~~
- Replaced BCE loss with BCEWithLogits loss (https://github.com/pykeen/pykeen/pull/109)
- Store default HPO ranges in loss classes (https://github.com/pykeen/pykeen/pull/111)
- Use entrypoints for datasets (https://github.com/pykeen/pykeen/pull/115) to allow
  registering of custom datasets
- Improved WANDB results tracker (https://github.com/pykeen/pykeen/pull/117, thanks @kantholtz)
- Reorganized ablation study generation and execution (https://github.com/pykeen/pykeen/pull/54)

Fixed
~~~~~
- Fixed bug in the initialization of ConvE (https://github.com/pykeen/pykeen/pull/100)
- Fixed cross-platform issue with random integer generation (https://github.com/pykeen/pykeen/pull/98)
- Fixed documentation build on ReadTheDocs (https://github.com/pykeen/pykeen/pull/104)

`1.0.4 <https://github.com/pykeen/pykeen/compare/v1.0.3...v1.0.4>`_ - 2020-08-25
--------------------------------------------------------------------------------
Added
~~~~~
- Enable restricted evaluation on a subset of entities/relations (https://github.com/pykeen/pykeen/pull/62,
  https://github.com/pykeen/pykeen/pull/83)

Changed
~~~~~~~
- Use number of epochs as step instead of number of checks (https://github.com/pykeen/pykeen/pull/72)

Fixed
~~~~~
- Fix bug in early stopping (https://github.com/pykeen/pykeen/pull/77)

`1.0.3 <https://github.com/pykeen/pykeen/compare/v1.0.2...v1.0.3>`_ - 2020-08-13
--------------------------------------------------------------------------------
Added
~~~~~
- Side-specific evaluation (https://github.com/pykeen/pykeen/pull/44)
- Grid Sampler (https://github.com/pykeen/pykeen/pull/52)
- Weights & Biases Tracker (https://github.com/pykeen/pykeen/pull/68), thanks @migalkin!

Changed
~~~~~~~
- Update to Optuna 2.0 (https://github.com/pykeen/pykeen/pull/52)
- Generalize specification of tracker (https://github.com/pykeen/pykeen/pull/39)

Fixed
~~~~~
- Fix bug in triples factory splitter (https://github.com/pykeen/pykeen/pull/59)
- Device mismatch bug (https://github.com/pykeen/pykeen/pull/50)

`1.0.2 <https://github.com/pykeen/pykeen/compare/v1.0.1...v1.0.2>`_ - 2020-07-10
--------------------------------------------------------------------------------
Added
~~~~~
- Add default values for margin and adversarial temperature in NSSA loss (https://github.com/pykeen/pykeen/pull/29)
- Added FTP uploader (https://github.com/pykeen/pykeen/pull/35)
- Add AWS S3 uploader (https://github.com/pykeen/pykeen/pull/39)

Changed
~~~~~~~
- Improved MLflow support (https://github.com/pykeen/pykeen/pull/40)
- Lots of improvements to documentation!

Fixed
~~~~~
- Fix triples factory splitting bug (https://github.com/pykeen/pykeen/pull/21)
- Fix problem with tensors' device during prediction (https://github.com/pykeen/pykeen/pull/41)
- Fix RotatE relation embeddings re-initialization (https://github.com/pykeen/pykeen/pull/26)

`1.0.1 <https://github.com/pykeen/pykeen/compare/v1.0.0...v1.0.1>`_ - 2020-07-02
--------------------------------------------------------------------------------
Added
~~~~~
- Add fractional hits@k (https://github.com/pykeen/pykeen/pull/17)
- Add link prediction pipeline (https://github.com/pykeen/pykeen/pull/10)

Changed
~~~~~~~
- Update documentation (https://github.com/pykeen/pykeen/pull/10)
