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

Same drill! Submit an issue, and we'll have a nice conversation in the thread.

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

This repository
uses [squash merges](https://docs.github.com/en/github/collaborating-with-pull-requests/incorporating-changes-from-a-pull-request/about-pull-request-merges#squash-and-merge-your-pull-request-commits)
to group all related commits in a given pull request into a single commit upon
acceptance and merge into the main branch. This has several benefits:

1. Keeps the commit history on the main branch focused on high-level narrative
2. Enables people to make lots of small commits without worrying about muddying
   up the commit history
3. Commits correspond 1-to-1 with pull requests

### Code Style

This project uses `tox` for running code quality checks. Start by installing
`tox` and `tox-uv` with `pip install tox tox-uv`.

This project encourages the use of optional static typing. It
uses [`mypy`](http://mypy-lang.org/) as a type checker. You can check if
your code passes `mypy` with `tox -e mypy`.

This project uses [`ruff`](https://docs.astral.sh/ruff/) to automatically
enforce a consistent code style. You can apply `ruff format` and other pre-configured
formatters with `tox -e format`.

This project uses [`ruff`](https://docs.astral.sh/ruff/) and several plugins for
additional checks of documentation style, security issues, good variable
nomenclature, and more (see `pyproject.toml` for a list of Ruff plugins). You can check if your
code passes `ruff check` with `tox -e lint`.

Each of these checks are run on each commit using GitHub Actions as a continuous
integration service. Passing all of them is required for accepting a
contribution. If you're unsure how to address the feedback from one of these
tools, please say so either in the description of your pull request or in a
comment, and we will help you.

### Logging

Python's builtin `print()` should not be used (except when writing to files),
it's checked by the
[`flake8-print` (T20)](https://docs.astral.sh/ruff/rules/#flake8-print-t20) plugin to `ruff`. If
you're in a command line setting or `main()` function for a module, you can use
`click.echo()`. Otherwise, you can use the builtin `logging` module by adding
`logger = logging.getLogger(__name__)` below the imports at the top of your
file.

### Documentation

All public functions (i.e., not starting with an underscore `_`) must be
documented using
the [sphinx documentation format](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html#the-sphinx-docstring-format).
The [`darglint2`](https://github.com/akaihola/darglint2) tool
reports on functions that are not fully documented.

This project uses [`sphinx`](https://www.sphinx-doc.org) to automatically build
documentation into a narrative structure. You can check that the documentation
builds properly in an isolated environment with `tox -e docs-test` and actually
build it locally with `tox -e docs`.

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
UI to do this by
following [this tutorial](https://docs.github.com/en/github/collaborating-with-pull-requests/working-with-forks/syncing-a-fork).

### Python Version Compatibility

This project aims to support all versions of Python that have not passed their
end-of-life dates. After end-of-life, the version will be removed from the Trove
qualifiers in the `pyproject.toml` and from the GitHub Actions testing
configuration.

See https://endoflife.date/python for a timeline of Python release and
end-of-life dates.

## Acknowledgements

These code contribution guidelines are derived from
the [cthoyt/cookiecutter-snekpack](https://github.com/cthoyt/cookiecutter-snekpack)
Python package template. They're free to reuse and modify as long as they're properly acknowledged.
