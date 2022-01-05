import argparse


def _parse_arguments(name, backends):
    parser = argparse.ArgumentParser(name)
    parser.add_argument(
        "-b",
        "--backend",
        help="backend to run the test on",
        choices=backends,
        default=backends[0],
    )
    parser.add_argument("-q", "--quiet", action="store_true")
    return vars(parser.parse_args())


class ExampleRunner:
    def __init__(self, run_function, name, backends):
        self._arguments = _parse_arguments(name, backends)
        self._run_function = run_function
        self._name = name

    def run(self):
        backend = self._arguments["backend"]
        quiet = self._arguments["quiet"]
        if not quiet:
            print(self._name)
            print("-" * len(self._name))
            print(f"Using '{backend}' backend")
            print()
        self._run_function(backend=backend, quiet=quiet)
