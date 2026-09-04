#pragma once

// Shared declarations for the nanobind extension. Each ABI area lives in its
// own translation unit so that editing one does not recompile the others, and
// so that none of them ever pulls in decode/megakernel.cuh / kittens.cuh.

#include <nanobind/nanobind.h>

#include "decode/abi.h"

namespace mk_bindings {

// ── Error reporting ─────────────────────────────────────────────────────────
//
// mk::MkError is what the native side throws; mk_ext.cpp registers it as this
// Python class. The alias lets the bindings raise it directly for failures they
// detect themselves (bad arguments, a closed service), so Python sees one
// exception type either way.
using MkAbiError = mk::MkError;

// ── Per-area binders ────────────────────────────────────────────────────────

// Populates the `mk_ext.kv` submodule with the paged-KV and prefix-cache ABI
// declared in src/kv/cache.h.
void bind_kv(nanobind::module_& m);

// Populates the `mk_ext.launch` submodule with the megakernel launch/runtime
// descriptors and the JIT, profiler and one-shot decode entry points declared
// in src/decode/abi.h. Registers all four descriptor types, including
// NmcDecodeServiceDesc, because they embed one another and must be registered
// together.
void bind_launch(nanobind::module_& m);

// Populates the `mk_ext.service` submodule with the long-lived decode service
// (continuous batching) entry points declared in src/decode/abi.h.
void bind_service(nanobind::module_& m);

}  // namespace mk_bindings
