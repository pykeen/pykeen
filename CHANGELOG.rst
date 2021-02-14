Change Log
==========
All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <http://keepachangelog.com/>`_
and this project adheres to `Semantic Versioning <http://semver.org/>`_

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
