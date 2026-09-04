// Entry point of the nanobind extension. Python binds it once per process
// through src/native.py and reaches it as `native.ext()`.
//
// Layout: one submodule per ABI area, one translation unit per submodule, so
// that editing bindings never recompiles src/decode/runtime.cu and so that no binding TU
// ever includes decode/megakernel.cuh / kittens.cuh.
//
//   mk_ext.kv  -- src/bindings/kv.cpp, the paged-KV / prefix-cache ABI
//
// TWO THINGS THAT WILL BITE WHOEVER ADDS THE NEXT SUBMODULE:
//
//  1. GIL. nanobind holds the GIL across a call unless you release it. Any
//     blocking entry point -- DecodeService::run(), DecodeService::wait_paused(),
//     JitKernel::compile() (runs nvcc), and anything that takes the
//     process-global mutex in kv/cache.cpp -- must release it explicitly or
//     the decode thread will stall, or deadlock, the whole server.
//     test_binding_invariants.py audits this at the source level.
//
//  2. Global state. Everything here reaches into the process-global pools in
//     kv/cache.cpp / decode/runtime.cu. Exactly one copy of libmk_release.so
//     may be loaded per process; see the comment on the mk_ext target in
//     CMakeLists.txt.

#include <nanobind/nanobind.h>

#include <chrono>
#include <thread>

#include "bindings.h"

namespace nb = nanobind;

NB_MODULE(mk_ext, m) {
    m.doc() = "Native megakernel bindings (nanobind).";

    nb::module_ kv = m.def_submodule(
        "kv", "Paged-KV pool, page tables, and prefix cache (src/kv/cache.h).");
    mk_bindings::bind_kv(kv);

    // Registered at the top level, before the binders run, so that the
    // `launch` and `service` submodules both raise this one class. Its
    // reference is owned by the module namespace from here on.
    const nb::exception<mk_bindings::MkAbiError> mk_abi_error(
        m, "MkAbiError", PyExc_RuntimeError);
    (void)mk_abi_error;

    nb::module_ launch = m.def_submodule(
        "launch", "Megakernel launch descriptors, JIT and profiler (src/decode/abi.h).");
    mk_bindings::bind_launch(launch);

    nb::module_ service = m.def_submodule(
        "service", "Long-lived decode service for continuous batching (src/decode/abi.h).");
    mk_bindings::bind_service(service);

    // ── GIL probes (test-only) ──────────────────────────────────────────────
    // A matched pair used by src/test_kv_bindings.py to prove that
    // nb::gil_scoped_release actually yields the GIL in THIS build. They exist
    // because the obvious check, PyGILState_Check(), is not part of the
    // limited API and therefore unavailable in our STABLE_ABI build -- see
    // CMakeLists.txt. Comparing how far a Python thread gets during each of
    // these two sleeps is self-calibrating and needs no absolute threshold.
    //
    // Keep them: if a future nanobind/Python upgrade ever broke GIL release,
    // the failure mode is a hung decode thread, which is miserable to debug
    // from first principles.
    m.def(
        "_gil_sleep_released",
        [](double seconds) {
            nb::gil_scoped_release release;
            std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
        },
        nb::arg("seconds"),
        "Sleep with the GIL released. Test scaffolding; do not call in anger.");
    m.def(
        "_gil_sleep_held",
        [](double seconds) {
            std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
        },
        nb::arg("seconds"),
        "Sleep while holding the GIL. Test scaffolding; do not call in anger.");
}
