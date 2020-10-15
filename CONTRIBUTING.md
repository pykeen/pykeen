# Contributing

Contributions, whether big or small, are appreciated! You can get involved by submitting an
issue, making a suggestion, or adding code to the project. PyKEEN is young and wants to address
the problems the KGE community is currently facing, and has a lot of excited people working on it!

## Having a Problem? Submit an Issue.

1. Check that you have the latest version of :code:`PyKEEN`
2. Check that StackOverflow hasn't already solved your problem
3. Go here: https://github.com/pykeen/pykeen/issues
4. Check that this issue hasn't been solved
5. Click "new issue"
6. Choose the appropriate issue template then follow its instructions.
   Issues not following the template may be discarded without review.

## Have a Question or Suggestion?

Same drill! Submit an issue and we'll have a nice conversation in the thread.

## Want to Contribute Code?

1. Get the code. Fork the repository from GitHub using the big green button in the top-right corner of
   https://github.com/pykeen/pykeen
2. Clone your directory with

    $ git clone https://github.com/<YourUsername>/pykeen

3. Install with :code:`pip`. The flag, :code:`-e`, makes your installation editable, so your changes will be reflected
   automatically in your installation.

    $ cd pykeen
    $ python3 -m pip install -e .

4. Make a branch off of develop, then make contributions! This line makes a new branch and checks it out

    $ git checkout -b feature/<YourFeatureName>

5. This project should be well tested, so write unit tests in the :code:`tests/` directory
6. Check that all tests are passing and code coverage is good with :code:`tox` before committing.

    $ tox

## Pull Requests

Once you've got your feature or bugfix finished (or if its in a partially complete state but you want to publish it
for comment), push it to your fork of the repository and open a pull request against the develop branch on GitHub.

Make a descriptive comment about your pull request, perhaps referencing the issue it is meant to fix (something along
the lines of "fixes issue #10" will cause GitHub to automatically link to that issue). The maintainers will review your
pull request and perhaps make comments about it, request changes, or may pull it in to the develop branch! If you need
to make changes to your pull request, simply push more commits to the feature branch in your fork to GitHub and they
will automatically be added to the pull. You do not need to close and reissue your pull request to make changes!

If you spend a while working on your changes, further commits may be made to the main :code:`PyKEEN` repository (called
"upstream") before you can make your pull request. In keep your fork up to date with upstream by pulling the
changes--if your fork has diverged too much, it becomes difficult to properly merge pull requests without conflicts.

To pull in upstream changes::

    $ git remote add upstream https://github.com/pykeen
    $ git fetch upstream develop

Check the log to make sure the upstream changes don't affect your work too much::

    $ git log upstream/develop

Then merge in the new changes::

    $ git merge upstream/develop

More information about this whole fork-pull-merge process can be found `here on Github's
website <https://help.github.com/articles/fork-a-repo/>`_.

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

## Upload to PyPI

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
