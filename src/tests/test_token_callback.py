"""Tests for the nanobind token-callback trampoline (src/bindings/launch.cpp).

This callback is the one place where the decode thread calls *into* Python, so
the failure modes are different from the rest of the bindings: a raised
exception has no C ABI channel to travel through, and the GIL has to be
acquired rather than released. These tests drive the real C entry point via
`launch._benchmark_token_callback`, not the Python object, because calling the
wrapper from Python would bypass the trampoline entirely.

    python src/tests/test_token_callback.py
"""

from __future__ import annotations

import contextlib
import ctypes
import functools
import os
import re
import sys
import threading

# ``python src/tests/*.py`` puts src/tests on sys.path[0]; production modules
# live one directory up in src/.
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import native  # noqa: E402
import test_binding_invariants  # noqa: E402

LAUNCH_CPP = os.path.join(test_binding_invariants.bindings_dir(), "launch.cpp")


def with_launch(test):
    """Adapt a test that wants the bound `launch` module to the runner's ABI.

    The shared runner hands every test a library path; resolving it here keeps
    that plumbing out of sixteen test bodies.
    """

    @functools.wraps(test)
    def wrapper(library_path: str) -> None:
        test(native.ext().launch)

    return wrapper


def with_launch_and_unraisable(test):
    @functools.wraps(test)
    def wrapper(library_path: str) -> None:
        with capture_unraisable() as captured:
            test(native.ext().launch, captured)

    return wrapper


def source_only(test):
    """For checks that read the binding source and need no built extension."""

    @functools.wraps(test)
    def wrapper(_library_path: str) -> None:
        test()

    return wrapper


def drive(launch, callback, *, batch: int, iterations: int = 1) -> None:
    """Invoke the callback through the real C ABI path, GIL released."""
    launch._benchmark_token_callback(
        callback_ptr=callback.function_ptr,
        context_ptr=callback.context_ptr,
        batch=batch,
        iterations=iterations,
    )


@with_launch
def test_receives_sampled_and_emitted(launch):
    seen = []
    callback = launch.TokenCallback(lambda s, e: seen.append((s, e)))
    drive(launch, callback, batch=4)

    assert len(seen) == 1
    sampled, emitted = seen[0]
    # The harness fills sampled with 12345 and marks every row emitted.
    assert list(sampled) == [12345] * 4
    assert list(emitted) == [1] * 4


@with_launch
def test_arguments_are_plain_int_tuples(launch):
    """The callback body indexes these per row; they must be cheap Python ints.

    Guards against a refactor to ndarray/memoryview views, which would be
    zero-copy but make every element access box a scalar -- slower for the
    small batches this runs at, and it would silently change `token`'s type
    where it reaches request.emit().
    """
    captured = []
    callback = launch.TokenCallback(lambda s, e: captured.append((s, e)))
    drive(launch, callback, batch=3)

    sampled, emitted = captured[0]
    for sequence in (sampled, emitted):
        assert type(sequence) is tuple
        assert all(type(value) is int for value in sequence)


@contextlib.contextmanager
def capture_unraisable():
    """Collect what PyErr_WriteUnraisable reports.

    That is the channel the trampoline uses to surface a failing callback,
    since the C ABI only carries a status code. The default hook prints to
    stderr, which a script runner cannot assert on.
    """
    captured = []
    previous = sys.unraisablehook
    sys.unraisablehook = captured.append
    try:
        yield captured
    finally:
        sys.unraisablehook = previous
        # Each entry holds a traceback, which holds the test frame, which
        # holds the TokenCallback local. Left alone this keeps the wrapper
        # alive to interpreter shutdown and nanobind reports it as a leak.
        captured.clear()


@with_launch_and_unraisable
def test_exception_is_caught_and_reported(launch, unraisable):
    """A raising callback must not unwind through the C ABI."""

    def explode(sampled, emitted):
        raise ValueError("kaboom-sentinel")

    callback = launch.TokenCallback(explode)
    drive(launch, callback, batch=2)  # Must not crash the process.

    assert "kaboom-sentinel" in callback.last_error
    # The traceback must reach the unraisable hook: the C signature can only
    # carry a non-zero return, so this is the sole channel for the cause.
    assert len(unraisable) == 1
    assert unraisable[0].exc_type is ValueError
    assert "kaboom-sentinel" in str(unraisable[0].exc_value)


@with_launch_and_unraisable
def test_error_indicator_is_clean_after_exception(launch, unraisable):
    """A leaked error indicator would surface as a spurious failure later.

    PyErr_WriteUnraisable is supposed to consume it; if it did not, the next
    unrelated Python call on this thread would raise.
    """
    callback = launch.TokenCallback(lambda s, e: 1 / 0)
    drive(launch, callback, batch=2)

    assert int("41") + 1 == 42  # Any Python work would blow up otherwise.
    assert "ZeroDivisionError" in callback.last_error
    assert len(unraisable) == 1


@with_launch_and_unraisable
def test_failure_does_not_wedge_later_invocations(launch, unraisable):
    """One bad step must not poison the callback for the rest of the run."""
    calls = []

    def flaky(sampled, emitted):
        calls.append(len(sampled))
        if len(calls) == 1:
            raise RuntimeError("first-step-only")

    callback = launch.TokenCallback(flaky)
    drive(launch, callback, batch=2, iterations=5)

    assert calls == [2] * 5
    assert len(unraisable) == 1


@with_launch
def test_last_error_empty_until_failure(launch):
    callback = launch.TokenCallback(lambda s, e: None)
    assert callback.last_error == ""
    drive(launch, callback, batch=2, iterations=8)
    assert callback.last_error == ""


@with_launch
def test_return_value_is_ignored(launch):
    """Only raising aborts decode; a truthy return must not be read as failure."""
    callback = launch.TokenCallback(lambda s, e: "not a status code")
    drive(launch, callback, batch=2, iterations=4)
    assert callback.last_error == ""


@with_launch
def test_pointers_are_stable_across_reads(launch):
    """The descriptor caches these once; they must not change per access."""
    callback = launch.TokenCallback(lambda s, e: None)
    assert callback.function_ptr == callback.function_ptr
    assert callback.context_ptr == callback.context_ptr
    assert callback.function_ptr != 0
    assert callback.context_ptr != 0


@with_launch
def test_context_ptr_distinguishes_instances(launch):
    """Two sessions must not demux into each other's slots."""
    first_calls, second_calls = [], []
    first = launch.TokenCallback(lambda s, e: first_calls.append(s))
    second = launch.TokenCallback(lambda s, e: second_calls.append(s))

    assert first.context_ptr != second.context_ptr
    # Same C entry point, dispatch is purely by context.
    assert first.function_ptr == second.function_ptr

    drive(launch, first, batch=1)
    assert len(first_calls) == 1 and len(second_calls) == 0


@with_launch
def test_matches_ctypes_observable_behaviour(launch):
    """TokenCallback must observe the same values a plain C callback does.

    A ctypes CFUNCTYPE serves as the reference implementation. Both are driven
    through the identical C entry point, so any difference is the binding
    layer's doing.
    """
    ctypes_seen = []
    signature = ctypes.CFUNCTYPE(
        ctypes.c_int, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32)

    @signature
    def legacy(_context, sampled, emitted, batch):
        ctypes_seen.append(
            ([int(sampled[i]) for i in range(batch)],
             [int(emitted[i]) for i in range(batch)]))
        return 0

    launch._benchmark_token_callback(
        callback_ptr=ctypes.cast(legacy, ctypes.c_void_p).value,
        context_ptr=0, batch=5, iterations=1)

    nb_seen = []
    callback = launch.TokenCallback(
        lambda s, e: nb_seen.append((list(s), list(e))))
    drive(launch, callback, batch=5)

    assert nb_seen == ctypes_seen


@with_launch
def test_callable_is_kept_alive_by_the_wrapper(launch):
    """The wrapper owns its callable; the caller keeping no reference is fine."""
    marker = []
    callback = launch.TokenCallback(lambda s, e: marker.append(len(s)))
    # The lambda has no other referent now.
    drive(launch, callback, batch=3, iterations=2)
    assert marker == [3, 3]


@with_launch
def test_reentrant_python_call_does_not_deadlock(launch):
    """The trampoline acquires the GIL; the body must be free to use Python.

    A gil_scoped_acquire that was mismatched, or a lock held across the call,
    would hang here rather than fail.
    """
    done = threading.Event()

    def body(sampled, emitted):
        # Force real interpreter work, including an allocation and a thread
        # state check.
        sum(int(value) for value in sampled)
        done.set()

    callback = launch.TokenCallback(body)
    worker = threading.Thread(
        target=drive, args=(launch, callback), kwargs={"batch": 4, "iterations": 64})
    worker.start()
    worker.join(timeout=30)
    assert not worker.is_alive(), "token callback deadlocked"
    assert done.is_set()


@with_launch
def test_concurrent_callbacks_from_multiple_threads(launch):
    """Two decode services can run at once; the GIL handoff must be safe."""
    counts = [0, 0]

    def make(index):
        def body(sampled, emitted):
            counts[index] += 1
        return launch.TokenCallback(body)

    callbacks = [make(0), make(1)]
    threads = [
        threading.Thread(target=drive, args=(launch, cb),
                         kwargs={"batch": 4, "iterations": 500})
        for cb in callbacks
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
        assert not thread.is_alive(), "concurrent token callbacks deadlocked"
    assert counts == [500, 500]


@source_only
def test_trampoline_acquires_gil_in_source():
    """Source-level guard: invoke() runs on a thread that holds no GIL.

    A runtime check is not possible here (PyGILState_Check is absent from the
    stable ABI), and omitting the acquire would not fail deterministically --
    it would corrupt the interpreter under load.
    """
    with open(LAUNCH_CPP, encoding="utf-8") as handle:
        source = handle.read()

    body = re.search(
        r"static int invoke\(.*?\n    \}", source, re.DOTALL)
    assert body is not None, "could not locate TokenCallback::invoke"
    assert "nb::gil_scoped_acquire" in body.group(0)
    assert "noexcept" in source[body.start():body.start() + 200]


@source_only
def test_invoke_catches_every_exception_type_in_source():
    """A missing catch-all would let an exception cross the C ABI."""
    with open(LAUNCH_CPP, encoding="utf-8") as handle:
        source = handle.read()
    body = re.search(r"static int invoke\(.*?\n    \}", source, re.DOTALL)
    assert body is not None
    assert "catch (...)" in body.group(0)


TESTS = [
    test_receives_sampled_and_emitted,
    test_arguments_are_plain_int_tuples,
    test_exception_is_caught_and_reported,
    test_error_indicator_is_clean_after_exception,
    test_failure_does_not_wedge_later_invocations,
    test_last_error_empty_until_failure,
    test_return_value_is_ignored,
    test_pointers_are_stable_across_reads,
    test_context_ptr_distinguishes_instances,
    test_matches_ctypes_observable_behaviour,
    test_callable_is_kept_alive_by_the_wrapper,
    test_reentrant_python_call_does_not_deadlock,
    test_concurrent_callbacks_from_multiple_threads,
    test_trampoline_acquires_gil_in_source,
    test_invoke_catches_every_exception_type_in_source,
]


if __name__ == "__main__":
    raise SystemExit(
        test_binding_invariants.run_test_main(
            tests=TESTS, argv=sys.argv[1:], description=__doc__ or ""
        )
    )
