"""Standalone regression tests for the nanobind paged-KV bindings.

Covers ``src/bindings/kv.cpp`` and the state and validation that ``kv_pool``
adds on top of it: handle lifecycle, prefill/decode stepping, rebind, the
prefix cache, error mapping, and the GIL contract.

Needs a CUDA device but NOT model weights or a checkpoint, so it runs in a few
seconds and is the fast gate for KV-layer changes. It allocates a real shared
KV arena, so it must run in its own process (the arena is process-global and
sized once).

    python src/tests/test_kv_bindings.py --lib-path $PWD/build/libmk_release.so

Exits non-zero on the first failure.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import threading
import time
from collections.abc import Callable

import torch

# ``python src/tests/*.py`` puts src/tests on sys.path[0]; production modules
# live one directory up in src/.
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import kv.pool as kv_pool  # noqa: E402
import native  # noqa: E402
import test_binding_invariants  # noqa: E402

# Small geometry: the point is ABI behaviour, not throughput. page_block must
# match what the native pool is initialised with on the first handle.
NUM_LAYERS = 2
BATCH_SIZE = 4
PAGE_BLOCK = 16
N_KV_HEADS = 2
HEAD_DIM = 32
MAX_SEQ = 512
# Keep the arena tiny; this test shares a GPU with whatever else is running.
FRAC_VRAM = 0.01
SW_SIZE = 0  # eviction off: block accounting stays predictable.
SW_PATTERN = 4


class TestFailure(AssertionError):
    """A single check failed."""


def check(condition: bool, message: str) -> None:
    if not condition:
        raise TestFailure(message)


def check_equal(actual: object, expected: object, message: str) -> None:
    if actual != expected:
        raise TestFailure(f"{message}: expected {expected!r}, got {actual!r}")


def check_raises(
    exception_type: type[BaseException], operation: Callable[[], object], message: str
) -> BaseException:
    try:
        operation()
    except exception_type as exc:
        return exc
    except BaseException as exc:  # noqa: BLE001 - report the wrong type precisely.
        raise TestFailure(
            f"{message}: expected {exception_type.__name__}, "
            f"got {type(exc).__name__}: {exc}"
        ) from exc
    raise TestFailure(f"{message}: expected {exception_type.__name__}, nothing raised")


def prefix_key(seed: int) -> bytes:
    """A well-formed 32-byte key. Content is arbitrary but must be stable."""
    return hashlib.sha256(f"mk-kv-test-{seed}".encode()).digest()


# ── Fixtures ────────────────────────────────────────────────────────────────


def make_cache(batch_size: int = BATCH_SIZE):
    return kv_pool.allocate_nmc_paged_kv_cache(
        num_layers=NUM_LAYERS,
        batch_size=batch_size,
        max_seq=MAX_SEQ,
        page_block=PAGE_BLOCK,
        n_kv_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        dtype=torch.bfloat16,
        device=torch.device("cuda:0"),
        sw_size=SW_SIZE,
        sw_pattern=SW_PATTERN,
        frac_vram_utilization=FRAC_VRAM,
        required_max_seq=None,
    )


# ── Tests ───────────────────────────────────────────────────────────────────


def test_bind_identity(library_path: str) -> None:
    """The process binds to one build, once, and refuses to change its mind."""

    module = native.ext()
    check(hasattr(module, "kv"), "mk_ext must expose a `kv` submodule")
    check(
        native.bind(library_path=library_path, debug=False) is module,
        "rebinding the same path must return the same module object",
    )
    check_equal(
        native.bound_library_path(),
        os.path.realpath(library_path),
        "bound_library_path must report the resolved path",
    )
    # Two different builds in one process would give each its own copy of the
    # process-global KV pool. That must fail loudly, not silently. The stand-in
    # only has to exist -- the singleton check must fire before anything tries
    # to dlopen it.
    with tempfile.TemporaryDirectory() as other_build:
        other_library = os.path.join(other_build, "libmk_release.so")
        with open(other_library, "wb"):
            pass
        check_raises(
            native.NativeExtensionError,
            lambda: native.bind(library_path=other_library, debug=False),
            "binding a second distinct build must be rejected",
        )

    # A late debug flip would leave the C++ watchdog with unallocated snapshot
    # buffers, so it is an error rather than a silent downgrade.
    check_raises(
        native.NativeExtensionError,
        lambda: native.bind(library_path=library_path, debug=True),
        "changing the debug flag after bind must be rejected",
    )
    check_equal(native.debug_enabled(), False, "debug must stay as first bound")

    check_raises(
        FileNotFoundError,
        lambda: native.bind(
            library_path="/nonexistent/build/libmk_release.so", debug=False
        ),
        "a nonexistent library path must be rejected",
    )
    check_raises(
        ValueError,
        lambda: native.bind(library_path="build/libmk_release.so", debug=False),
        "a relative library path must be rejected",
    )


def test_native_types(_library_path: str) -> None:
    """The KV types are real classes owned by C++, reachable without aliases."""

    kv = native.ext().kv
    error_type = kv.KvAbiError
    check(
        isinstance(error_type, type) and issubclass(error_type, RuntimeError),
        "KvAbiError must be a real RuntimeError subclass, not a proxy "
        "(an `except` clause requires an actual exception type)",
    )
    try:
        raise error_type("synthetic")
    except kv.KvAbiError:
        pass
    for name in ("KvHandle", "KvHandleDesc", "KvStats", "PrefixCacheStats"):
        check(isinstance(getattr(kv, name), type), f"{name} must resolve to a class")


def test_handle_desc_roundtrip(_library_path: str) -> None:
    """The descriptor is a real C++ struct; Python writes its actual fields."""

    kv = native.ext().kv
    fields = {
        "page_table": 0x1111,
        "cache_seqlens": 0x2222,
        "row_active": 0x3333,
        "num_layers": 1,
        "BS": 2,
        "max_pages_per_seq": 3,
        "num_phys_pages": 4,
        "page_block_size": 5,
        "start_pos": 6,
        "sw_size": 7,
        "sw_pattern": 8,
    }
    desc = kv.KvHandleDesc(**fields)
    for name, value in fields.items():
        check_equal(getattr(desc, name), value, f"KvHandleDesc.{name} keyword init")
    # Distinct values per field: a transposed keyword-to-field mapping in the
    # C++ constructor would show up here as a swapped pair.
    desc.BS = 99
    check_equal(desc.BS, 99, "KvHandleDesc fields must be writable")
    check_equal(desc.num_layers, 1, "writing one field must not disturb another")


def test_prefill_and_decode_stepping(_library_path: str) -> None:
    """Page tables and seqlens track prepare/step calls on the device.

    Note the deliberate asymmetry between the two calls, which is easy to get
    backwards: ``prepare_prefill`` maps per-row using each row's own prompt
    length, whereas ``step_prefill_chunk`` maps the chunk's block range on
    EVERY row regardless of length (kv/cache.cpp, kv_ensure_blocks_for_row
    loop). Both behaviours are asserted separately below.
    """

    cache = make_cache()
    try:
        prompt_lengths = [64, 32, 16, 48]
        cache.prepare_prefill_lengths(prompt_lengths)
        torch.cuda.synchronize()

        seqlens = cache.cache_seqlens.cpu().tolist()
        check_equal(seqlens, prompt_lengths, "cache_seqlens after prepare_prefill")

        # prepare_prefill is per-row: exactly ceil(len / page_block) pages, and
        # the remainder of the row still reads as unmapped (-1).
        page_table = cache.page_table.cpu()
        for layer in range(NUM_LAYERS):
            for row, length in enumerate(prompt_lengths):
                expected_pages = (length + PAGE_BLOCK - 1) // PAGE_BLOCK
                mapped = int((page_table[layer, row] >= 0).sum())
                check_equal(
                    mapped,
                    expected_pages,
                    f"prepare_prefill pages for layer={layer} row={row} len={length}",
                )

        # step_prefill_chunk is uniform across rows: after covering [0, 64) every
        # row holds 4 pages, including the row whose prompt was only 16 long.
        chunk_end = max(prompt_lengths)
        for layer in range(NUM_LAYERS):
            cache.step_prefill_chunk(layer, 0, chunk_end)
        torch.cuda.synchronize()
        chunk_pages = (chunk_end + PAGE_BLOCK - 1) // PAGE_BLOCK
        page_table = cache.page_table.cpu()
        for layer in range(NUM_LAYERS):
            for row in range(BATCH_SIZE):
                check_equal(
                    int((page_table[layer, row] >= 0).sum()),
                    chunk_pages,
                    f"step_prefill_chunk pages for layer={layer} row={row}",
                )
        check_equal(
            cache.cache_seqlens.cpu().tolist(),
            prompt_lengths,
            "step_prefill_chunk must not move cache_seqlens",
        )

        # A decode step writes AT `positions[b]`, so cache_seqlens lands on the
        # position itself, not position + 1. (Attention reads cache_seqlens + 1;
        # do not "fix" this to match that.) Positions are deliberately past the
        # chunk that was just mapped so row 0 must pull in a fresh block.
        before = cache.allocated_blocks()
        positions = [length + 3 for length in prompt_lengths]
        cache.step_decode_positions(positions, [1] * BATCH_SIZE)
        torch.cuda.synchronize()
        check_equal(
            cache.cache_seqlens.cpu().tolist(),
            positions,
            "cache_seqlens after one decode step",
        )
        check(
            cache.allocated_blocks() >= before,
            "decode stepping must never release blocks",
        )
        page_table = cache.page_table.cpu()
        for layer in range(NUM_LAYERS):
            check_equal(
                int((page_table[layer, 0] >= 0).sum()),
                positions[0] // PAGE_BLOCK + 1,
                f"row 0 must gain the block holding position {positions[0]} "
                f"in layer {layer}",
            )

        # An inactive row must be recorded as such and left alone.
        cache.step_decode_positions(positions, [1, 0, 1, 1])
        torch.cuda.synchronize()
        check_equal(
            cache.row_active.cpu().tolist(),
            [1, 0, 1, 1],
            "row_active must mirror the active mask",
        )

        stats = cache.stats()
        check_equal(
            stats.total,
            stats.allocated + stats.free,
            "KvStats total must equal allocated + free",
        )
        check(stats.peak_allocated >= stats.allocated, "peak must bound allocated")
        check(
            cache.free_blocks() == stats.free,
            "free_blocks() and KvStats.free must agree",
        )
    finally:
        cache.close()


def test_masked_variants(_library_path: str) -> None:
    """Masked prefill touches only the rows selected by `active`."""

    cache = make_cache()
    try:
        cache.prepare_prefill_lengths([32] * BATCH_SIZE)
        for layer in range(NUM_LAYERS):
            cache.step_prefill_chunk(layer, 0, 32)
        torch.cuda.synchronize()
        baseline = cache.cache_seqlens.cpu().tolist()

        # Re-admit row 1 only, at a different length.
        active = [0, 1, 0, 0]
        cache.prepare_prefill_lengths_masked([0, 80, 0, 0], active)
        for layer in range(NUM_LAYERS):
            cache.step_prefill_chunk_masked(layer, 0, 80, active)
        torch.cuda.synchronize()

        updated = cache.cache_seqlens.cpu().tolist()
        check_equal(updated[1], 80, "masked prefill must update the active row")
        for row in (0, 2, 3):
            check_equal(
                updated[row],
                baseline[row],
                f"masked prefill must leave inactive row {row} untouched",
            )
    finally:
        cache.close()


def test_set_cache_seqlens_and_free_row(_library_path: str) -> None:
    cache = make_cache()
    try:
        cache.prepare_prefill_lengths([48] * BATCH_SIZE)
        for layer in range(NUM_LAYERS):
            cache.step_prefill_chunk(layer, 0, 48)
        torch.cuda.synchronize()

        cache.set_cache_seqlens([10, 20, 30, 40])
        torch.cuda.synchronize()
        check_equal(
            cache.cache_seqlens.cpu().tolist(),
            [10, 20, 30, 40],
            "set_cache_seqlens must write through to the device tensor",
        )

        before = cache.allocated_blocks()
        cache.free_row(0)
        torch.cuda.synchronize()
        after = cache.allocated_blocks()
        check(after < before, f"free_row must release blocks ({before} -> {after})")
        page_table = cache.page_table.cpu()
        for layer in range(NUM_LAYERS):
            check_equal(
                int((page_table[layer, 0] >= 0).sum()),
                0,
                f"freed row must be fully unmapped in layer {layer}",
            )
    finally:
        cache.close()


def test_rebind(_library_path: str) -> None:
    """Rebind moves live rows into a new, differently sized handle."""

    kv = native.ext().kv
    cache = make_cache(batch_size=BATCH_SIZE)
    new_handle = None
    try:
        cache.prepare_prefill_lengths([64, 32, 16, 48])
        for layer in range(NUM_LAYERS):
            cache.step_prefill_chunk(layer, 0, 64)
        torch.cuda.synchronize()
        allocated_before = cache.allocated_blocks()

        new_batch = 2
        device = cache.page_table.device
        new_page_table = torch.full(
            (NUM_LAYERS, new_batch, cache.max_pages), -1, device=device, dtype=torch.int32
        )
        new_cache_seqlens = torch.zeros((new_batch,), device=device, dtype=torch.int32)
        new_row_active = torch.ones((new_batch,), device=device, dtype=torch.int32)

        # Destination row 0 takes old row 2, destination row 1 starts empty.
        new_handle = cache.rebind(
            [2, -1], new_page_table, new_cache_seqlens, new_row_active
        )
        check(
            isinstance(new_handle, kv.KvHandle),
            "rebind must return a live KvHandle",
        )
        torch.cuda.synchronize()
        check_equal(
            int(new_cache_seqlens.cpu()[0]),
            16,
            "rebound destination row must inherit the source row's length",
        )
        check_equal(
            int(new_cache_seqlens.cpu()[1]), 0, "unmapped destination row must be empty"
        )
        # Ownership transferred, so the rows dropped by the rebind are released
        # only once the old handle goes away. close() drops the last reference,
        # which is what runs the C++ destructor.
        cache.close()
        remaining = new_handle.stats().allocated
        check(
            0 < remaining < allocated_before,
            f"rebind must retain a strict subset of blocks "
            f"({allocated_before} -> {remaining})",
        )
    finally:
        new_handle = None
        cache.close()


def test_descriptor_shares_handle_ownership(_library_path: str) -> None:
    """A descriptor owns its KV handle, so decode cannot read freed blocks.

    This is the invariant that replaced ``desc.kv_handle = <address>``. With a
    raw address, closing the Python cache freed the blocks while the native
    decode loop was still reading them; with a shared_ptr, the blocks survive
    until every holder is gone. The second half matters just as much: nothing
    reclaims them until the descriptor lets go too, which is why
    session._Geometry.release_kv clears both.
    """

    kv = native.ext().kv
    kv.prefix_cache_clear()
    desc = native.ext().launch.NmcDecodeServiceDesc()
    check_equal(desc.kv_handle, None, "a fresh descriptor must hold no handle")

    cache = make_cache()
    desc.kv_handle = cache.handle
    check(
        desc.kv_handle is cache.handle,
        "the descriptor must hand back the same handle object, not a copy",
    )
    cache.prepare_prefill_lengths([PAGE_BLOCK] * BATCH_SIZE)
    for layer in range(NUM_LAYERS):
        cache.step_prefill_chunk(layer, 0, PAGE_BLOCK)
    torch.cuda.synchronize()
    allocated = kv.prefix_cache_stats().active_blocks
    check(allocated > 0, "the prefill must have allocated blocks")

    cache.close()
    check_equal(
        kv.prefix_cache_stats().active_blocks,
        allocated,
        "closing the cache while a descriptor holds the handle must NOT free "
        "the blocks the native side is still pointed at",
    )

    desc.kv_handle = None
    check_equal(
        kv.prefix_cache_stats().active_blocks,
        0,
        "the last holder releasing must return every block to the pool",
    )


def test_prefix_cache(_library_path: str) -> None:
    """publish -> can_attach -> attach round trip, plus clear()."""

    kv = native.ext().kv
    kv.prefix_cache_clear()
    check_equal(kv.prefix_cache_size(), 0, "cache must start empty")

    cache = make_cache()
    try:
        cache.prepare_prefill_lengths([PAGE_BLOCK * 2] * BATCH_SIZE)
        for layer in range(NUM_LAYERS):
            cache.step_prefill_chunk(layer, 0, PAGE_BLOCK * 2)
        torch.cuda.synchronize()

        key = prefix_key(0)
        check(
            not cache.prefix_can_attach(key, 0, 0),
            "can_attach must be False before anything is published",
        )
        published = cache.prefix_publish(key=key, row=0, logical_block=0)
        check(published, "publishing a fully-written logical block must succeed")
        check_equal(
            kv.prefix_cache_size(),
            1,
            "cache size must reflect the published entry",
        )
        check(
            cache.prefix_can_attach(key, 0, 0),
            "can_attach must be True after publish",
        )
        attached = cache.prefix_attach(
            key=key, row=1, logical_block=0, required_from_block=0
        )
        check(attached, "attaching a published block into another row must succeed")

        stats = kv.prefix_cache_stats()
        check_equal(stats.entries, 1, "PrefixCacheStats.entries")
        check(stats.total_blocks > 0, "PrefixCacheStats.total_blocks must be positive")

        # A key that was never published is a miss (False), not an error.
        check(
            not cache.prefix_can_attach(prefix_key(999), 0, 0),
            "an unknown key must be a miss, not an exception",
        )

        kv.prefix_cache_clear()
        check_equal(kv.prefix_cache_size(), 0, "clear() must empty it")
    finally:
        cache.close()
        kv.prefix_cache_clear()


def test_prefix_key_validation(_library_path: str) -> None:
    """Wrong-length keys are caller error, never a silent cache miss.

    The rejection happens natively: the binding measures the blob and
    kv/cache.cpp requires SHA-256-sized material, so the error arrives as
    KvAbiError carrying the C++ message.
    """

    kv = native.ext().kv
    cache = make_cache()
    try:
        for bad_key, label in (
            (b"", "empty"),
            (b"\x00" * 31, "31 bytes"),
            (b"\x00" * 33, "33 bytes"),
        ):
            check_raises(
                kv.KvAbiError,
                lambda k=bad_key: cache.prefix_can_attach(k, 0, 0),
                f"a {label} prefix key must be rejected",
            )
        check_raises(
            TypeError,
            lambda: cache.prefix_can_attach("not bytes", 0, 0),
            "a str prefix key must be rejected",
        )
        # NUL bytes must survive: keys are raw SHA-256 material, so a binding
        # that treated them as C strings would truncate and collide here.
        nul_key = b"\x00" * 16 + b"\xff" * 16
        other_key = b"\x00" * 16 + b"\x01" * 16
        kv.prefix_cache_clear()
        cache.prepare_prefill_lengths([PAGE_BLOCK] * BATCH_SIZE)
        for layer in range(NUM_LAYERS):
            cache.step_prefill_chunk(layer, 0, PAGE_BLOCK)
        torch.cuda.synchronize()
        cache.prefix_publish(key=nul_key, row=0, logical_block=0)
        check(
            not cache.prefix_can_attach(other_key, 0, 0),
            "keys that share a leading NUL run must not collide",
        )
    finally:
        cache.close()
        kv.prefix_cache_clear()


def test_error_mapping(_library_path: str) -> None:
    """Native failures become typed Python exceptions, not return codes."""

    kv = native.ext().kv
    # KvHandle's constructor throws rather than returning a null handle, and
    # the message travels on the exception -- there is no thread-local error
    # slot to consult and no return code to forget to check.
    exc = check_raises(
        kv.KvAbiError,
        lambda: kv.KvHandle(kv.KvHandleDesc()),
        "constructing a handle without a page table must raise",
    )
    check(
        "page_table" in str(exc),
        f"the exception must say what was wrong, got {str(exc)!r}",
    )

    cache = make_cache()
    try:
        # A native-side range check also arrives as KvAbiError, carrying the
        # C++ message.
        row_exc = check_raises(
            kv.KvAbiError,
            lambda: cache.free_row(BATCH_SIZE + 5),
            "an out-of-range row must raise from C++",
        )
        check(
            "out of range" in str(row_exc),
            f"the exception must carry the C++ message, got {str(row_exc)!r}",
        )
        # Length mismatches are caught by the binding layer, which sees both
        # the sequence length and the handle's batch size. It raises
        # ValueError; without it C++ would read out of bounds.
        length_exc = check_raises(
            ValueError,
            lambda: cache.prepare_prefill_lengths([1, 2]),
            "a short prompt_lengths sequence must be rejected",
        )
        check(
            "one per batch row" in str(length_exc),
            f"the exception must name the batch-row requirement, got {str(length_exc)!r}",
        )
        check_raises(
            ValueError,
            lambda: cache.step_decode_positions([1] * (BATCH_SIZE + 1), [1] * BATCH_SIZE),
            "an over-long positions sequence must be rejected",
        )
        # nanobind range-checks every int32 argument, including the elements of
        # a host array, and reports a signature mismatch as TypeError.
        check_raises(
            TypeError,
            lambda: cache.prepare_prefill_lengths([2**40] * BATCH_SIZE),
            "values that do not fit in int32 must be rejected",
        )
        check_raises(
            kv.KvAbiError,
            lambda: cache.step_prefill_chunk(0, 10, 5),
            "pos_end < pos_start must be rejected",
        )
    finally:
        cache.close()

    # After close(), the handle is gone and every method must refuse.
    check_raises(
        ValueError,
        cache.stats,
        "operations on a closed cache must be rejected",
    )


def test_tensor_validation(_library_path: str) -> None:
    """create_kv_handle rejects tensors the native side cannot use."""

    device = torch.device("cuda:0")
    good_pt = torch.full((NUM_LAYERS, BATCH_SIZE, 8), -1, device=device, dtype=torch.int32)
    good_sl = torch.zeros((BATCH_SIZE,), device=device, dtype=torch.int32)
    good_ra = torch.ones((BATCH_SIZE,), device=device, dtype=torch.int32)

    def create(page_table, seqlens, row_active):
        return kv_pool.create_kv_handle(
            page_table, seqlens, row_active, 64, PAGE_BLOCK, 0,
            SW_SIZE, SW_PATTERN,
        )

    check_raises(
        ValueError,
        lambda: create(good_pt.cpu(), good_sl, good_ra),
        "a CPU page_table must be rejected",
    )
    check_raises(
        ValueError,
        lambda: create(good_pt.to(torch.int64), good_sl, good_ra),
        "a non-int32 page_table must be rejected",
    )
    check_raises(
        ValueError,
        lambda: create(good_pt[:, :, ::2], good_sl, good_ra),
        "a non-contiguous page_table must be rejected",
    )
    check_raises(
        ValueError,
        lambda: create(good_pt, torch.zeros((BATCH_SIZE + 1,), device=device, dtype=torch.int32), good_ra),
        "a cache_seqlens length mismatch must be rejected",
    )


def _ticks_during(call: Callable[[], None]) -> int:
    """Run ``call`` and report how far a competing Python thread got."""

    stop = threading.Event()
    ticks = [0]
    started = threading.Event()

    def spin() -> None:
        started.set()
        while not stop.is_set():
            ticks[0] += 1

    worker = threading.Thread(target=spin, name="gil-probe", daemon=True)
    worker.start()
    started.wait(timeout=5.0)
    try:
        call()
    finally:
        stop.set()
        worker.join(timeout=5.0)
    check(not worker.is_alive(), "the probe thread must exit")
    return ticks[0]


def test_gil_release_mechanism(_library_path: str) -> None:
    """``nb::gil_scoped_release`` really does yield the GIL in this build.

    nanobind holds the GIL across a call unless the binding releases it. A
    binding that forgets the guard stalls every other Python thread for the
    duration of the native call, and with the decode-service token callback in
    play it is worse than a stall: a C++ thread that wants the GIL while a
    Python thread waits on kv/cache.cpp's process-global ``g_prefix_mu``
    deadlocks outright.

    The direct check, ``PyGILState_Check()``, is not part of the limited API
    and so is unavailable in our STABLE_ABI build. Instead we compare a
    matched pair of native sleeps: the released one must let a competing
    Python thread run far more than the held one. Self-calibrating, so it
    needs no absolute tick threshold and does not care how fast the host is.
    """

    module = native.ext()
    duration = 0.3
    held = _ticks_during(lambda: module._gil_sleep_held(duration))
    released = _ticks_during(lambda: module._gil_sleep_released(duration))

    check(
        released > held * 10,
        "nb::gil_scoped_release did not yield the GIL: a competing thread got "
        f"{released} ticks with it released vs {held} while held "
        "(expected a large multiple)",
    )


def test_kv_bindings_all_release_the_gil(_library_path: str) -> None:
    """Every KV binding wraps its native call in ``nb::gil_scoped_release``.

    A source-level invariant rather than a runtime one on purpose: the guard is
    a per-call-site convention in src/bindings/kv.cpp, and the failure it
    guards against (someone adds an entry point and forgets it) is invisible at
    runtime until the decode thread wedges under load. Checking the source
    catches it at the moment it is introduced.
    """

    count = test_binding_invariants.assert_entry_points_release_gil(
        filename="kv.cpp",
        exempt={
            "KvHandleDesc.__init__": "assigns POD fields; no native call",
            "KvStats.__repr__": "formats a POD copy; no native call",
            "KvHandle.batch_size": "reads an int from host state",
            "KvHandle.num_layers": "reads an int from host state",
            "KvHandle.max_pages_per_seq": "reads an int from host state",
        },
    )
    check(count > 0, "parser found no entry points in kv.cpp")


def test_concurrent_access(_library_path: str) -> None:
    """Concurrent readers must not deadlock or corrupt the global pool."""

    kv = native.ext().kv
    cache = make_cache()
    errors: list[BaseException] = []
    barrier = threading.Barrier(4)

    def hammer() -> None:
        try:
            barrier.wait(timeout=10.0)
            for _ in range(500):
                cache.stats()
                cache.free_blocks()
                kv.prefix_cache_stats()
        except BaseException as exc:  # noqa: BLE001 - reported by the main thread.
            errors.append(exc)

    threads = [
        threading.Thread(target=hammer, name=f"kv-hammer-{i}", daemon=True)
        for i in range(4)
    ]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30.0)
            check(
                not thread.is_alive(),
                f"{thread.name} did not finish in 30s; likely a GIL/mutex deadlock",
            )
        check_equal(errors, [], "concurrent KV access raised")
    finally:
        cache.close()


TESTS: tuple[Callable[[str], None], ...] = (
    test_bind_identity,
    test_native_types,
    test_handle_desc_roundtrip,
    test_prefill_and_decode_stepping,
    test_masked_variants,
    test_set_cache_seqlens_and_free_row,
    test_rebind,
    test_descriptor_shares_handle_ownership,
    test_prefix_cache,
    test_prefix_key_validation,
    test_error_mapping,
    test_tensor_validation,
    test_gil_release_mechanism,
    test_kv_bindings_all_release_the_gil,
    test_concurrent_access,
)


if __name__ == "__main__":
    raise SystemExit(
        test_binding_invariants.run_test_main(
            tests=TESTS, argv=sys.argv[1:], description=__doc__ or ""
        )
    )
