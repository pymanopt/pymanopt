# Contributing

[Fork and clone the repository][fork] by going to
https://github.com/pymanopt/pymanopt/fork, and cloning the repository with

    git clone git@github.com:<your-username>/pymanopt.git

Set up a local development environment, installing both the runtime and
development dependencies by running

    pip install -e ".[all]"

Verify that all existing tests pass by executing the test suite with:

    pytest -v --cov-report= --cov=src/pymanopt tests

Note that we run the [flake8][flake8] utility on every python file in the
package to verify coding style consistency during our integration tests.
As such, failure to comply to the [style guide][style] will result in a failing
ci build.
To prevent adding commits which fail to adhere to the PEP8
guidelines, we include a [pre-commit][pre-commit] config, which immediately
invokes flake8 on all files staged for commit when running `git commit`.
To enable the hook, simply run `pre-commit install` after installing
`pre-commit` either manually via `pip` or as part of `dev` variant, e.g.

    pip install -e ".[dev]"

Push a feature branch to your fork and [submit a pull request][pr].
Refer to [this guide][commits] on how to write good commit messages.

## Sign-off

By making a contribution (pull requesting or committing) to the Pymanopt
project you certify that

- you have the right to submit it to Pymanopt.

- you created the contribution/modification; or you based it on previous work
  that, to the best of your knowledge, is covered by a compatible open source
  license; or someone who did one of the former provided you with this
  contribution/modification and you are submitting it without changes.

- you understand and agree that your contribution/modification to this
  project is public and that a record of it (including all information you
  submit with it, including copyright notices and your sign-off) is
  maintained indefinitely and may be redistributed consistent with Pymanopt's
  3-clause BSD license or the open source license(s) involved.

To make your certification explicit we borrow the "sign-off" procedure
from the Linux kernel project, i.e., each commit message should contain
a line saying

    Signed-off-by: Name Surname <name.surname@example.org>

using your real name and email address.
Running the git-commit command with the -s option automatically adds this line.

[fork]: https://help.github.com/articles/cloning-a-repository/
[flake8]: http://flake8.pycqa.org/en/latest/
[pre-commit]: https://pre-commit.com/
[style]: https://www.python.org/dev/peps/pep-0008/
[pr]: https://github.com/pymanopt/pymanopt/compare
[commits]: http://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html
