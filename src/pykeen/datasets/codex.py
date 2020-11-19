# -*- coding: utf-8 -*-

"""CoDEx datasets.

- GitHub Repository: https://github.com/tsafavi/codex
- Paper: https://arxiv.org/pdf/2009.07810.pdf

Citation:

.. [safavi2020] Safavi, T. & Koutra, D. (2020). `CoDEx: A Comprehensive Knowledge Graph
   Completion Benchmark <http://arxiv.org/abs/2009.07810>`_.  *arXiv* 2009.07810.
"""

BASE_URL = 'https://github.com/tsafavi/codex/raw/master/data/triples'
SMALL_VALID_URL = f'{BASE_URL}/codex-s/valid.txt'
SMALL_TEST_URL = f'{BASE_URL}/codex-s/test.txt'
SMALL_TRAIN_URL = f'{BASE_URL}/codex-s/train.txt'

MEDIUM_VALID_URL = f'{BASE_URL}/codex-m/valid.txt'
MEDIUM_TEST_URL = f'{BASE_URL}/codex-m/test.txt'
MEDIUM_TRAIN_URL = f'{BASE_URL}/codex-m/train.txt'

LARGE_VALID_URL = f'{BASE_URL}/codex-l/valid.txt'
LARGE_TEST_URL = f'{BASE_URL}/codex-l/test.txt'
LARGE_TRAIN_URL = f'{BASE_URL}/codex-l/train.txt'
