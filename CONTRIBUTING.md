# Contributing

Contributions, whether big or small, are appreciated! You can get involved by submitting an
issue, making a suggestion, or adding code to the project. PyKEEN is young and wants to address
the problems the KGE community is currently facing, and has a lot of excited people working on it!

## Having a Problem? Submit an Issue.

1. Check that you have the latest version of `pykeen`
2. Check that StackOverflow hasn't already solved your problem
3. Go here: https://github.com/pykeen/pykeen/issues
4. Check that this issue hasn't been solved
5. Click "new issue"
6. Choose the appropriate issue template then follow its instructions.
   Issues not following the template may be discarded without review.

## Have a Question or Suggestion?

Same drill! Submit an issue and we'll have a nice conversation in the thread.

## Code Contribution

This project uses the [GitHub Flow](https://guides.github.com/introduction/flow)
model for code contributions. Follow these steps:

1. [Create a fork](https://help.github.com/articles/fork-a-repo) of the upstream
   repository
   at [`pykeen/pykeen`](https://github.com/pykeen/pykeen)
   on your GitHub account (or in one of your organizations)
2. [Clone your fork](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository)
   with `git clone https://github.com/<your namespace here>/pykeen.git`
3. Make and commit changes to your fork with `git commit`
4. Push changes to your fork with `git push`
5. Repeat steps 3 and 4 as needed
6. Submit a pull request back to the upstream repository

### Merge Model

This project
uses [squash merges](https://docs.github.com/en/github/collaborating-with-pull-requests/incorporating-changes-from-a-pull-request/about-pull-request-merges#squash-and-merge-your-pull-request-commits)
to group all related commits in a given pull request into a single commit upon
acceptance and merge into the main branch. This has several benefits:

1. Keeps the commit history on the main branch focused on high-level narrative
2. Enables people to make lots of small commits without worrying about muddying
   up the commit history
3. Commits correspond 1-to-1 with pull requests

### Code Style

This project encourages the use of optional static typing. It
uses [`mypy`](http://mypy-lang.org/) as a type checker
and [`sphinx_autodoc_typehints`](https://github.com/agronholm/sphinx-autodoc-typehints)
to automatically generate documentation based on type hints. You can check if
your code passes `mypy` with `tox -e mypy`.

This project uses [`black`](https://github.com/psf/black) to automatically
enforce a consistent code style. You can apply `black` and other pre-configured
linters with `tox -e lint`.

This project uses [`flake8`](https://flake8.pycqa.org) and several plugins for
additional checks of documentation style, security issues, good variable
nomenclature, and more (
see [`tox.ini`](tox.ini) for a list of flake8 plugins). You can check if your
code passes `flake8` with `tox -e flake8`.

Each of these checks are run on each commit using GitHub Actions as a continuous
integration service. Passing all of them is required for accepting a
contribution. If you're unsure how to address the feedback from one of these
tools, please say so either in the description of your pull request or in a
comment, and we will help you.

### Logging

Python's builtin `print()` should not be used (except when writing to files),
it's checked by the
[`flake8-print`](https://github.com/jbkahn/flake8-print) plugin to `flake8`. If
you're in a command line setting or `main()` function for a module, you can use
`click.echo()`. Otherwise, you can use the builtin `logging` module by adding
`logger = logging.getLogger(__name__)` below the imports at the top of your
file.

### Documentation

All public functions (i.e., not starting with an underscore `_`) should be
documented using
the [sphinx documentation format](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html#the-sphinx-docstring-format).
The [`darglint`](https://github.com/terrencepreilly/darglint) plugin to `flake8`
reports on functions that are not fully documented.

This project uses [`sphinx`](https://www.sphinx-doc.org) to automatically build
documentation into a narrative structure. You can check that the documentation
properly builds with `tox -e docs-test` and build the docs locally with
`tox -e docs`.

### Testing

Functions in this repository should be unit tested. These can either be written
using the `unittest` framework in the `tests/` directory or as embedded
doctests. You can check that the unit tests pass with `tox -e py` and that the
doctests pass with `tox -e doctests`. These tests are required to pass for
accepting a contribution.

### Syncing your fork

If other code is updated before your contribution gets merged, you might need to
resolve conflicts against the main branch. After cloning, you should add the
upstream repository with

```shell
$ git remote add pykeen https://github.com/pykeen/pykeen.git
```

Then, you can merge upstream code into your branch. You can also use the GitHub
UI to do this by following [this tutorial](https://docs.github.com/en/github/collaborating-with-pull-requests/working-with-forks/syncing-a-fork).

### Python Version Compatibility

This project aims to support all versions of Python that have not passed their
end-of-life dates. After end-of-life, the version will be removed from the Trove
qualifiers in the [`setup.cfg`](setup.cfg) and from the GitHub Actions testing
configuration.

See https://endoflife.date/python for a timeline of Python release and
end-of-life dates.

## Making a Release

PyKEEN uses single source versioning. This means that there's a variable
`pykeen.version.VERSION` which is the canonical value used as the version.

Management of this value is done by `bumpversion` via `tox`. When you're
ready to make a release, do the following:

1. Make sure there are no uncommitted changes.
2. Run `tox -e bumpversion release`
3. Push to GitHub
4. Draft a new release at https://github.com/pykeen/pykeen/releases/new.
   Name the release based on the version that was just bumped to with the form
   vX.Y.Z where X is major release, Y is minor release, and Z is patch. By default,
   there's a box that says `Target: master`. If you're not 100% sure the last commit
   made before making a tag/release was the bump commit, click it, click "Recent Commits"
   then click the commit with the text `Bump version: X.Y.Z-dev -> X.Y.Z`.

### Upload to PyPI

Directly after making a release, you can easily upload to PyPI using another `tox`
command:

1. `tox -e release` prepares the code and uploads it to PyPI.
2. `tox -e bumpversion patch` to bump the version. **DO NOT** do this before uploading to
   PyPI, otherwise the version on PyPI will have `-dev` as a suffix.
3. Push to GitHub

The process of bumping the version (release), pushing to GitHub, making a release to PyPI,
bumping the version (patch), and pushing to GitHub one more time has been automated with
`tox -e finish`. If you use this, make sure you go to GitHub and manually find the right
commit for making a tag/release, since it will not be the most recent one.
