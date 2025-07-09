import argparse
import sys

from ._methods import (
    _TestSystem,
)


def _autoconfigure(envfile: str = None, shfile: str = None):
    from casm.bset import autoconfigure

    print("Begin autoconfigure...")
    results = autoconfigure(apply_results=False, return_results=True, verbose=True)

    if results["vars"] is None:
        print("autoconfigure: FAILED")
        for failed in results["failed"]:
            print()
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("Tried:")
            print("##")
            if "CASM_PREFIX" not in failed["vars"]:
                print("export CASM_PREFIX=$(python -m libcasm.casmglobal --prefix)")
            for k, v in failed["vars"].items():
                if v is not None:
                    print(f'export {k}="{v}"')
            print("##")
            print()
            print(f'Error: {failed["what"]}')
    else:
        print("autoconfigure: SUCCESS")
        print()
        print("Successful configuration:")
        print()
        print("##")
        if "CASM_PREFIX" not in results["vars"]:
            print("export CASM_PREFIX=$(python -m libcasm.casmglobal --prefix)")
        for k, v in results["vars"].items():
            if v is not None:
                print(f'export {k}="{v}"')
        print("##")

        if envfile:
            with open(envfile, "a") as f:
                if "CASM_PREFIX" not in results["vars"]:
                    import io
                    from contextlib import redirect_stdout

                    from libcasm.casmglobal.__main__ import main as cgmain

                    g = io.StringIO()
                    with redirect_stdout(g):
                        cgmain(argv=["casmglobal", "--prefix"])
                    casm_prefix = g.getvalue().strip()
                    f.write(f"CASM_PREFIX={casm_prefix}\n")
                for k, v in results["vars"].items():
                    if v is not None:
                        f.write(f"{k}={v}\n")

            print()
            print("appended variables to:", envfile)

        if shfile:
            with open(shfile, "w") as f:
                if "CASM_PREFIX" not in results["vars"]:
                    f.write(
                        "export CASM_PREFIX=$(python -m libcasm.casmglobal --prefix)\n"
                    )
                for k, v in results["vars"].items():
                    if v is not None:
                        f.write(f'export {k}="{v}"\n')

            print()
            print(f"wrote: {shfile}")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--autoconfigure",
    action="store_true",
    help="Run autoconfigure to find environment variables for compiling Clexulator.",
)
parser.add_argument(
    "--test",
    action="store_true",
    help="Test configuration of environment variables for compiling Clexulator.",
)
parser.add_argument(
    "--envfile",
    type=str,
    help="If provided, successful variables are appended to the given file.",
)
parser.add_argument(
    "--shfile",
    type=str,
    help="If provided, a shell script with given name is written that can be used to "
    "set the successful variables.",
)
parser.add_argument(
    "--cxxflags",
    type=str,
    help=(
        "If using --test, specifies the C++ compiler flags that are tested "
        '(default="-O3 -Wall -fPIC -std=c++17 ").'
    ),
    default="-O3 -Wall -fPIC -std=c++17 ",
)
parser.add_argument(
    "--soflags",
    type=str,
    help=(
        "If using --test, specifies the shared object compilation flags that are "
        'tested (default="-shared ").'
    ),
    default="-shared ",
)
parser.add_argument(
    "--prefix",
    type=str,
    help=(
        "If using --test, specifies the CASM prefix that is tested "
        "(default=$(python -m libcasm.casmglobal --prefix))."
    ),
)

args = parser.parse_args(args=sys.argv[1:])

if not sys.argv[1:]:
    parser.print_help()
    exit()

if args.autoconfigure:
    _autoconfigure(envfile=args.envfile, shfile=args.shfile)
elif args.test:
    if args.prefix:
        print("prefix:", args.prefix)
    else:
        try:
            from libcasm.casmglobal.__main__ import main as cgmain
        except ImportError:
            raise ImportError("libcasm is not installed")

        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            cgmain(argv=["casmglobal", "--prefix"])
        casm_prefix = f.getvalue().strip()
        print("prefix:", casm_prefix)
    print("cxxflags:", args.cxxflags)
    print("soflags:", args.soflags)
    print()

    verbose = True
    test_vars = dict(
        CASM_CXXFLAGS=args.cxxflags,
        CASM_SOFLAGS=args.soflags,
    )

    with _TestSystem() as test_system:

        try:
            test_system.try_vars(test_vars, verbose=verbose)
            print("SUCCESS")
            sys.exit(0)
        except Exception:
            print("FAILED")
            sys.exit(1)
        test_system.reset()
else:
    parser.print_help()
    exit()
