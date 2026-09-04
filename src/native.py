"""Binds this process to one megakernel build and hands out the extension.

Everything behind this ABI is process-global (the
physical KV block pool, the prefix cache, the decode-service registry). Loading
two different builds into one interpreter would give each its own copy of that
state and silently corrupt the KV pool, so a second distinct path is a hard
error rather than a warning.

That invariant is the reason the library path stops here. Exactly one call to
:func:`bind` happens per process, at the CLI entry point that owns ``--lib``;
every other module reaches the ABI through :func:`ext` and never sees a path.
Several builds can still coexist on disk for A/B comparison -- the choice is
made once, at startup, instead of being carried through every call site.

``mk_ext.abi3.so`` links ``libmk_release.so`` with ``RUNPATH=$ORIGIN``, so the
two files must stay side by side; copying only one of them elsewhere fails at
import time, not at build time. ``bind`` is given the ``libmk_release.so`` path
because that is what ``--lib`` has always named, and loads its sibling
extension.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
from types import ModuleType

# Must match the CMake target name in CMakeLists.txt and the NB_MODULE() name
# in src/bindings/mk_ext.cpp.
_EXTENSION_MODULE_NAME = "mk_ext"
# STABLE_ABI builds carry the `.abi3.` infix instead of a per-interpreter tag.
_EXTENSION_FILENAME = "mk_ext.abi3.so"

_BIND_LOCK = threading.RLock()
_BOUND_MODULE: ModuleType | None = None
_BOUND_LIBRARY_PATH: str | None = None
_BOUND_DEBUG: bool | None = None


class NativeExtensionError(RuntimeError):
    """The mk_ext extension could not be located, loaded, or rebound."""


def extension_path_for(library_path: str) -> str:
    """Return the ``mk_ext.abi3.so`` that belongs to ``library_path``."""

    build_dir = os.path.dirname(os.path.realpath(library_path))
    return os.path.join(build_dir, _EXTENSION_FILENAME)


def _validate_library_path(library_path: str) -> str:
    """Reject anything that is not an existing absolute library path.

    No default search path is offered on purpose: silently picking up some
    other build would create a second, incompatible set of the process-global
    KV pools.
    """

    if not isinstance(library_path, str) or not library_path:
        raise ValueError("library_path must be a non-empty string")
    if not os.path.isabs(library_path):
        raise ValueError("library_path must be absolute")
    resolved = os.path.realpath(library_path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"release library does not exist: {resolved}")
    return resolved


def _import_extension(library_path: str) -> ModuleType:
    extension_path = extension_path_for(library_path)
    if not os.path.isfile(extension_path):
        raise NativeExtensionError(
            f"{_EXTENSION_FILENAME} not found next to {library_path!r}; "
            "build it with: cmake -S . -B build -G Ninja "
            "-DPython_EXECUTABLE=$(which python) && cmake --build build"
        )

    spec = importlib.util.spec_from_file_location(
        _EXTENSION_MODULE_NAME, extension_path
    )
    if spec is None or spec.loader is None:
        raise NativeExtensionError(
            f"failed to build an import spec for {extension_path!r}"
        )
    module = importlib.util.module_from_spec(spec)
    # Register before exec_module: CPython expects an extension module to be
    # findable in sys.modules while its initialization runs.
    sys.modules[_EXTENSION_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_EXTENSION_MODULE_NAME, None)
        raise
    return module


def bind(*, library_path: str, debug: bool) -> ModuleType:
    """Bind this process to one build and return the ``mk_ext`` extension.

    Call this once, from the entry point that owns the ``--lib`` flag, before
    anything touches the ABI. Repeat calls with identical arguments are no-ops
    so that a test suite can bind per test; a differing library path or debug
    flag is an error (see below).

    ``debug`` is pushed into the C++ half here, as part of the same
    indivisible step that loads it. That ordering is not cosmetic: C++
    allocates the watchdog snapshot buffers at decode-service geometry setup,
    and only when debug is already on. A flip afterwards leaves the debug path
    with nothing to read, which is why a rebind that changes the flag raises
    instead of silently degrading a ``--debug`` run.
    """

    global _BOUND_MODULE, _BOUND_LIBRARY_PATH, _BOUND_DEBUG

    resolved_library = _validate_library_path(library_path)
    requested_debug = bool(debug)
    with _BIND_LOCK:
        if _BOUND_MODULE is not None:
            if _BOUND_LIBRARY_PATH != resolved_library:
                raise NativeExtensionError(
                    "one process cannot load two different megakernel builds; "
                    "everything behind this ABI is process-global "
                    f"(bound={_BOUND_LIBRARY_PATH!r}, "
                    f"requested={resolved_library!r})"
                )
            if _BOUND_DEBUG != requested_debug:
                raise NativeExtensionError(
                    "the native debug flag cannot change after the build is "
                    "bound; the C++ watchdog sizes its snapshot buffers from "
                    f"the value seen at bind time (bound={_BOUND_DEBUG}, "
                    f"requested={requested_debug})"
                )
            return _BOUND_MODULE

        module = _import_extension(resolved_library)
        module.launch.set_debug(enabled=requested_debug)
        _BOUND_MODULE = module
        _BOUND_LIBRARY_PATH = resolved_library
        _BOUND_DEBUG = requested_debug
        return module


def ext() -> ModuleType:
    """Return the bound ``mk_ext`` extension.

    This is the only way the rest of the codebase reaches native code. It is
    on the decode hot path (once per KV operation), so it deliberately does no
    validation: the module is already dlopen'd and mapped by the time anyone
    calls this, and re-checking the path on disk could not un-load it.
    """

    module = _BOUND_MODULE
    if module is None:
        raise NativeExtensionError(
            "no megakernel build is bound to this process; call "
            "native.bind(library_path=..., debug=...) from the entry point "
            "first"
        )
    return module


def bound_library_path() -> str | None:
    """Return the library path this process is bound to, or None if unbound."""

    return _BOUND_LIBRARY_PATH


def debug_enabled() -> bool:
    """Return the host-side debug switch, or False before any bind.

    This gates the cuda-gdb toolchain only -- device pointer maps under
    ``cwd/dump/`` and the C++ watchdog's park-on-the-hung-stream behaviour. It
    never changes decode behaviour or the compiled kernel, so a ``--debug`` run
    stays representative.
    """

    return bool(_BOUND_DEBUG)
