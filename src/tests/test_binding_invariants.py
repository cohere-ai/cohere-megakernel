"""Shared harness and source-level invariants for the nanobind binding tests.

The invariant checks read src/bindings/*.cpp rather than exercising the
extension, because the properties they protect are per-call-site conventions
whose violation is invisible at runtime until something wedges under load.
Checking the source catches the mistake at the moment it is introduced.

Also holds the small test runner the binding suites share. These suites run as
plain scripts rather than under pytest so they can be invoked on a machine with
only the build environment present.

Imported by test_kv_bindings.py, test_launch_bindings.py and
test_token_callback.py; not a test module in its own right.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import traceback
from collections.abc import Callable

# ``python src/tests/*.py`` puts src/tests on sys.path[0]; production modules
# live one directory up in src/.
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import native  # noqa: E402


class InvariantViolation(AssertionError):
    """A binding source file broke a convention."""


def bindings_dir() -> str:
    return os.path.join(_SRC_DIR, "bindings")


# Every construct that can start a new binding definition, in one alternation so
# a single ordered scan can both name the entry points and delimit their chunks.
#
# ``def_rw`` / ``def_ro`` take a pointer-to-member and can only read or write a
# field, so they are matched purely to terminate the preceding chunk. ``def_prop_*``
# and ``def_static`` are NOT: they take arbitrary callables, and several of them
# (DecodeService.kv_pressure, Profiler.device_ptr) do take a lock or touch the
# device. Skipping them, as an earlier revision did, silently exempted every
# property from the GIL audit.
_BINDING_TOKEN_RE = re.compile(
    r"""
      (?P<cls>nb::class_<[^>]*>\(\s*m\s*,\s*"(?P<clsname>[^"]+)")
    | (?P<field>\.def_r[wo]\()
    | (?P<define>\.def(?:_prop_r[wo]|_static)?\()
    """,
    re.VERBOSE,
)
# A named definition is ``.def("name", ...)``. ``.def(nb::init<...>())`` binds a
# constructor with no name and no body of ours, so it is skipped.
_DEFINITION_NAME_RE = re.compile(r'\s*"([^"]+)"')


def entry_point_chunks(source: str) -> list[tuple[str, str]]:
    """Split a binding TU into (entry point name, source chunk) pairs.

    Handles both module-level ``m.def("x", ...)`` and class methods
    ``nb::class_<T>(m, "T").def("x", ...)``. Method names come back qualified as
    ``T.x`` so that, say, ``KvHandle.__init__`` (which calls into native code and
    must release the GIL) is a different entry from ``KvHandleDesc.__init__``
    (which only assigns POD fields).

    Chunking is by position: a chunk runs from its ``.def(`` to the next binding
    token of any kind, so anything found inside it really does belong to that
    entry point.
    """

    tokens = list(_BINDING_TOKEN_RE.finditer(source))
    chunks: list[tuple[str, str]] = []
    current_class: str | None = None
    for index, token in enumerate(tokens):
        end = tokens[index + 1].start() if index + 1 < len(tokens) else len(source)
        if token.lastgroup is not None and token.group("cls") is not None:
            current_class = token.group("clsname")
            continue
        if token.group("field") is not None:
            continue
        named = _DEFINITION_NAME_RE.match(source, token.end())
        if named is None:
            continue  # .def(nb::init<...>()) and friends
        # `m.def(` is module scope; a bare `.def(` continues the last class.
        module_level = source[: token.start()].rstrip().endswith("m")
        name = named.group(1)
        if not module_level and current_class is not None:
            name = f"{current_class}.{name}"
        chunks.append((name, source[token.start() : end]))
    return chunks


def assert_entry_points_release_gil(
    *, filename: str, exempt: dict[str, str]
) -> int:
    """Every ``m.def`` entry point in ``filename`` must release the GIL.

    ``exempt`` maps entry point name to the reason it does not need the guard;
    requiring a reason (rather than a bare set) keeps the exemption list from
    quietly becoming a dumping ground.

    Returns the number of entry points checked, so callers can assert the
    parser actually found something.
    """

    path = os.path.join(bindings_dir(), filename)
    with open(path, encoding="utf-8") as handle:
        source = handle.read()

    chunks = entry_point_chunks(source)
    if not chunks:
        raise InvariantViolation(f"found no .def( entry points in {path}")

    missing = [
        name
        for name, chunk in chunks
        if name not in exempt and "gil_scoped_release" not in chunk
    ]
    if missing:
        raise InvariantViolation(
            f"{filename}: these entry points call into native code without "
            f"releasing the GIL: {missing}. Add `nb::gil_scoped_release "
            f"release;` around the native call, or add the name to the "
            f"`exempt` mapping in the test with a reason."
        )

    stale = sorted(set(exempt) - {name for name, _ in chunks})
    if stale:
        raise InvariantViolation(
            f"{filename}: exempt names that no longer exist: {stale}. "
            f"Remove them so the list stays meaningful."
        )
    return len(chunks)


def default_library_path() -> str:
    root = os.path.dirname(_SRC_DIR)
    return os.path.join(root, "build", "libmk_release.so")


def run_test_main(
    *, tests: list[Callable[[str], None]], argv: list[str], description: str
) -> int:
    """Run a binding suite as a script. Returns a process exit code.

    Binds the process to the requested build first, since that is what a real
    entry point does and what ``native.ext()`` requires. Each test is still
    handed the resolved library path, because several of them assert on it.

    Failures are reported and the run continues, so one broken binding does not
    hide the state of the rest.
    """

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--lib-path", default=None, help="path to libmk_release.so")
    parser.add_argument(
        "--filter", default=None, help="run only tests whose name contains this"
    )
    args = parser.parse_args(argv)

    library_path = os.path.abspath(args.lib_path or default_library_path())
    if not os.path.isfile(library_path):
        print(
            f"libmk_release.so not found at {library_path}; build with "
            "`cmake --build build`",
            file=sys.stderr,
        )
        return 2
    native.bind(library_path=library_path, debug=False)

    selected = [
        test for test in tests if args.filter is None or args.filter in test.__name__
    ]
    if not selected:
        print(f"no test matches {args.filter!r}", file=sys.stderr)
        return 2

    failures = 0
    for test in selected:
        try:
            test(library_path)
        except BaseException:  # noqa: BLE001 - the harness reports everything.
            failures += 1
            print(f"FAIL {test.__name__}", flush=True)
            traceback.print_exc()
        else:
            print(f"ok   {test.__name__}", flush=True)

    print(f"\n{len(selected) - failures}/{len(selected)} passed")
    return 1 if failures else 0
