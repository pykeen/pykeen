# -*- coding: utf-8 -*-

"""Setup.py for pykeen ."""

import codecs
import os
import re

import setuptools

PYTHON_VERSION = "3.6"
MODULE = 'pykeen'
PACKAGES = setuptools.find_packages(where='src')
META_PATH = os.path.join('src', MODULE, '__init__.py')
KEYWORDS = ['Knowledge Graph Embeddings', 'Machine Learning', 'Data Mining', 'Linked Data']
CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'Environment :: Console',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    f'Programming Language :: Python :: {PYTHON_VERSION}',
    'Topic :: Scientific/Engineering :: Bio-Informatics'
]
INSTALL_REQUIRES = [
    'numpy==1.14.2; python_version == "3.6"',
    'numpy; python_version == "3.7"',
    'scikit-learn==0.19.1; python_version == "3.6"',
    'scikit-learn; python_version == "3.7"',
    'scipy',
    'click==6.7',
    'torch==0.4.0; python_version == "3.6"',
    'torch==0.4.1; python_version == "3.7"',
    'torchvision==0.2.1',
    'prompt_toolkit',
    'tqdm',
    'pandas',
]
EXTRAS_REQUIRE = {
    'docs': [
        'sphinx',
        'sphinx-rtd-theme',
        'sphinx-click',
    ],
}
ENTRY_POINTS = {
    'console_scripts': [
        'pykeen = pykeen.cli:main',
        'pykeen-summarize = pykeen.cli:summarize',
    ]
}

HERE = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    """Build an absolute path from *parts* and return the contents of the resulting file. Assume UTF-8 encoding."""
    with codecs.open(os.path.join(HERE, *parts), 'rb', 'utf-8') as f:
        return f.read()


META_FILE = read(META_PATH)


def find_meta(meta):
    """Extract __*meta*__ from META_FILE."""
    meta_match = re.search(
        r'^__{meta}__ = ["\']([^"\']*)["\']'.format(meta=meta),
        META_FILE, re.M
    )
    if meta_match:
        return meta_match.group(1)
    raise RuntimeError('Unable to find __{meta}__ string'.format(meta=meta))


def get_long_description():
    """Get the long_description from the README.rst file. Assume UTF-8 encoding."""
    with codecs.open(os.path.join(HERE, 'README.rst'), encoding='utf-8') as f:
        long_description = f.read()
    return long_description


if __name__ == '__main__':
    setuptools.setup(
        name=find_meta('title'),
        version=find_meta('version'),
        description=find_meta('description'),
        long_description=get_long_description(),
        url=find_meta('url'),
        author=find_meta('author'),
        author_email=find_meta('email'),
        maintainer=find_meta('author'),
        maintainer_email=find_meta('email'),
        license=find_meta('license'),
        classifiers=CLASSIFIERS,
        keywords=KEYWORDS,
        packages=PACKAGES,
        package_dir={'': 'src'},
        include_package_data=True,
        install_requires=INSTALL_REQUIRES,
        entry_points=ENTRY_POINTS,
        zip_safe=False,
    )
