# Contributing

[Fork and clone the repository][fork]:

    git clone git@github.com:your-username/pymanopt.git

Verify that all existing tests pass by either running

    python setup.py test

or executing the test suite via [nose][nose]:

    nosetests

Note that we also run the [pep8][pep8] utility on every python file in the
package to verify coding style consistency. As such, failure to comply to the
[style guide][style] will result in a failing travis-ci build.

Push a feature branch to your fork and [submit a pull request][pr]. Refer to
[this guide][commits] on how to write good commit messages.

[fork]: https://help.github.com/articles/cloning-a-repository/
[nose]: https://nose.readthedocs.org/en/latest/
[pep8]: http://pep8.readthedocs.org/en/latest/
[style]: https://www.python.org/dev/peps/pep-0008/
[pr]: https://github.com/j-towns/pymanopt/compare
[commits]: http://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html
