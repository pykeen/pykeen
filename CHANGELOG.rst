Change Log
==========
All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <http://keepachangelog.com/>`_
and this project adheres to `Semantic Versioning <http://semver.org/>`_

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
