# Contributing

Welcome supporter of this project! 👋 We are happy that you found your way to this page and appreciate every
contribution, helping us to develop new features or fixing potential issues of this software.
In order to keep this process transparent and allow us to keep track of things, we would like you to follow the
following guidelines:

1. [Report issues and problems or request new new features](#issues-and-feature-requests)
2. [Contribute to the software](#pull-request)
3. [Seek support or asking a question](#support)


## Issues and Feature Requests

To report errors in the software follow the steps for [bug reports](#bug-report). In case you would like to
request a useful feature or add-on for this software you should follow the steps for
[feature requests](#feature-request).


### Bug report

We track bugs for this software as [GitHub Issues](https://guides.github.com/features/issues/).
To file a bug report, please [submit an issue](#https://github.com/pykeen/pykeen/issues/new?assignees=&labels=&template=bug_report.md&title=)
and provide the therein requested information.

In case you don't have a GitHub account, you can use [this template](./ISSUE_TEMPLATE/bug_report.md)
and send it via e-mail to the [project team.](./project_team.md)


### Feature request

We track feature requests for this software as [GitHub Issues](https://guides.github.com/features/issues/).
To file a feature request, please [submit an issue](#https://github.com/pykeen/pykeen/issues/new?assignees=&labels=&template=feature_request.md&title=)
and provide the therein requested information.

In case you don't have a GitHub account, you can use [this template](./ISSUE_TEMPLATE/feature_request.md)
and send it via e-mail to the [project team.](./project_team.md)


## Pull request

Requests for direct code changes are handled as
[GitHub Pull Requests](https://help.github.com/articles/about-pull-requests/) and can be
[submitted here](https://github.com/pykeen/pykeen/compare).

Before submitting pull requests, please make sure that these address existing bug reports and/or feature requests. If this
is not the case, please make sure to first create these as described in
[Issues and Feature Requests](#issues-and-feature-requests).

When submitting pull requests, we encourage you to comply with these points:
 - Follow all instructions within the [pull request template](./pull_request_template.md)
 - Make sure that you create or adjust all documentation that is affected by your code changes
 - Create test scripts for non-trivial code changes, if these are not covered by existing tests.
 See also [CI & unit tests](#CI-&-unit-tests)


### Continuous integration & unit tests

To ensure code quality while developing new features this repository uses unit tests. The goal is to ensure the
correctness of all operations as well as the compatibility of the supported platforms when merging new code into the
master. In order to avoid overloading testing resources, the PyKEEN repository has a special syntax to invoke testing
**only when needed**.

This can be done in two ways.
1. If you want to test a specific commit, the commit message has to include the string "Trigger CI" (case insensitive) 
2. With the help of our @PyKEEN-bot, by simply commenting in the Pull Request-thread with a message containing 
"@PyKEEN-bot" and "test". (This invocation is limited to administrators and collaborators of this repository)

Last but not least, pushing to master will always trigger unit tests, unless you added the text "skip ci" to
your commit message. Read The Docs will only create a build once a commit is pushed/merged to master.


##### Technical information

The reason the bot has to create a new commit to trigger unit tests has to do with the inner working of Github Actions,
which consider a Pull Request to be part of the current master branch. A Github Action is strictly linked to the branch
that triggered the event, which in the case of Pull Request comments always will be the current master branch.
Accordingly, invoking unit tests straight from a Pull Request comment event would only allow to invoke unit tests for
the current master branch. Therefore, the @PyKEEN-bot comes into play, as the bot is triggered by the
Pull Request comment and creates a new empty commit to the branch linked to the Pull Request, which in return invokes
unit tests that are correctly linked to the actual branch that should be tested. As the commit is empty there will be 
no issues with merging in case you forgot to pull while working on a specific branch.


## Support

In case you just want to ask a question or need support for this software, you can reach out to the project team
via e-mail to the [project team.](./project_team.md)


## Code of Conduct

This project and everyone contributing to it is governed by
[our Code of Conduct](./CODE_OF_CONDUCT.md).
By participating, you are expected to comply with this code. Please report unacceptable behavior to the
[project team.](./project_team.md)
