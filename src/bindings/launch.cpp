// nanobind bindings for the megakernel launch / runtime descriptors declared
// in src/decode/abi.h. Reached from Python as `mk_ext.launch`; src/decode/schedule.py
// builds the descriptors and drives the JIT and profiler entry points.
//

#include "bindings.h"

#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <new>  // placement new in descriptor_init; clangd's unused-include heuristic misses this
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "decode/abi.h"

namespace nb = nanobind;

namespace mk_bindings {
namespace {

// Raw view of a layout-locked struct's memory, for debugging a suspected ABI
// mismatch against the JIT'd kernel. Read-only: mutate through the fields.
//
// Restricted to trivially copyable descriptors. The bytes of a descriptor
// holding a shared_ptr or vector are a libstdc++ implementation detail, and
// exposing them invites a memcpy of a refcounted member. The static_assert
// enforces this as members are added.
template <typename Desc>
nb::bytes descriptor_raw_bytes(const Desc& desc) {
    static_assert(
        std::is_trivially_copyable_v<Desc>,
        "_raw_bytes may only be exposed for trivially copyable descriptors; "
        "see the layout-lock note at the top of decode/abi.h");
    return nb::bytes(reinterpret_cast<const char*>(&desc), sizeof(Desc));
}

// Value-initialise, so every field starts at zero. See the note on the
// zero-initialisation contract above.
template <typename Desc>
void descriptor_init(Desc* self) {
    new (self) Desc{};
}

// Pack a host array into a Python tuple of ints. A tuple because the callback
// only reads it, and it is the cheaper object to allocate. Built through the C
// API so the storage is sized once; this runs once per decode step.
template <typename T>
nb::object pack_int_tuple(const T* data, size_t count) {
    PyObject* tuple = PyTuple_New(static_cast<Py_ssize_t>(count));
    if (tuple == nullptr) throw nb::python_error();
    for (size_t i = 0; i < count; ++i) {
        PyObject* item = PyLong_FromLongLong(static_cast<long long>(data[i]));
        if (item == nullptr) {
            Py_DECREF(tuple);
            throw nb::python_error();
        }
        // Steals `item`. The SET_ITEM macro is unavailable: this extension is
        // built against the stable ABI.
        PyTuple_SetItem(tuple, static_cast<Py_ssize_t>(i), item);
    }
    return nb::steal(tuple);
}

// Adapts a Python callable to mk::NmcRuntimeTokenCallback.
//
// The descriptor carries a C function pointer plus an opaque context: `invoke`
// is the entry point and `context` is the address of one of these objects.
// Python must keep the object alive for as long as the descriptor is in use,
// which for the decode service is its whole lifetime.
//
// A failing callback aborts the decode loop, and the C signature has room only
// for a non-zero return. The traceback is printed here and the message kept in
// last_error() so the session layer can report the cause.
class TokenCallback {
public:
    explicit TokenCallback(nb::callable fn) : fn_(std::move(fn)) {}

    // Called from the decode thread WITHOUT the GIL, across a C ABI boundary,
    // so no exception may escape.
    static int invoke(void* context, const long long* sampled,
                      const int32_t* emitted, int32_t batch) noexcept {
        auto* self = static_cast<TokenCallback*>(context);
        nb::gil_scoped_acquire gil;
        try {
            const auto count = static_cast<size_t>(batch < 0 ? 0 : batch);
            self->fn_(pack_int_tuple(sampled, count),
                      pack_int_tuple(emitted, count));
            return 0;
        } catch (nb::python_error& error) {
            self->last_error_ = error.what();
            error.restore();
            PyErr_WriteUnraisable(self->fn_.ptr());
            return 1;
        } catch (const std::exception& error) {
            self->last_error_ = error.what();
            return 1;
        } catch (...) {
            self->last_error_ = "unknown C++ exception in token callback";
            return 1;
        }
    }

    uint64_t function_ptr() const {
        return reinterpret_cast<uint64_t>(&TokenCallback::invoke);
    }
    uint64_t context_ptr() { return reinterpret_cast<uint64_t>(this); }
    const std::string& last_error() const { return last_error_; }

private:
    nb::callable fn_;
    std::string last_error_;
};

}  // namespace

// Derives the Python attribute name from the C++ member, so the two can never
// drift. Expands to a chained `.def_rw(...)`, so it requires a `Desc` type
// alias to be in scope and must follow another call in the chain.
#define RW(field) .def_rw(#field, &Desc::field)

// kv_handle owns its target, so its setter must accept None.
//
// Descriptors outlive the geometry they describe: session keeps one per
// batch size for the whole session. Without a way to clear this field, any
// descriptor that had been used would pin its KV blocks until the session
// ended. Assigning None is how a geometry hands them back; see
// session._Geometry.release_kv.
#define RW_KV_HANDLE()                                                        \
    .def_prop_rw(                                                             \
        "kv_handle",                                                          \
        [](const Desc& self) { return self.kv_handle; },                      \
        [](Desc& self, std::shared_ptr<kv_pool::KvHandle> handle) {             \
            self.kv_handle = std::move(handle);                               \
        },                                                                    \
        nb::for_setter(nb::arg("value").none()),                              \
        "The KV geometry this descriptor decodes into. Owned, not borrowed: "  \
        "the native side cannot read freed KV blocks. Assign None to release.")

// A std::vector field, whose getter returns BY VALUE.
//
// def_rw would hand back a reference, and nanobind's vector caster then builds
// a Python list whose elements point into the vector's storage. For a vector of
// bound structs (zero_regions, schedule_variants) that is a use-after-free one
// ordinary line of Python away:
//
//     region = desc.zero_regions[0]
//     desc.zero_regions = [...]   # reallocates, frees the old buffer
//     region.ptr                  # reads freed memory
//
// Copying every element means a read can never outlive the vector it came from.
// The cost is that reads are copies in both senses: `desc.zero_regions
// .append(x)` and `desc.zero_regions[0].ptr = y` are silent no-ops, so callers
// assign whole lists. test_vector_fields_are_copies_not_views covers both.
//
// Applied to the scalar vectors (eos_token_ids, start_pos_per_row, timing_ms)
// as well, whose elements convert to Python ints and floats by value and are
// already safe. One rule for every vector field is harder to forget than one
// that depends on the element type.
#define RW_VEC(field, doc)                                                    \
    .def_prop_rw(                                                             \
        #field,                                                               \
        [](const Desc& self) { return self.field; },                          \
        [](Desc& self, decltype(Desc::field) value) {                         \
            self.field = std::move(value);                                    \
        },                                                                    \
        doc " Reads return a copy: assign a whole list to change it.")

// Owning field whose setter must accept None, so a long-lived descriptor can
// drop the compiled kernel it names. See RW_KV_HANDLE.
#define RW_JIT_HANDLE()                                                       \
    .def_prop_rw(                                                             \
        "jit_handle",                                                         \
        [](const Desc& self) { return self.jit_handle; },                     \
        [](Desc& self, std::shared_ptr<mk::JitKernel> kernel) {               \
            self.jit_handle = std::move(kernel);                              \
        },                                                                    \
        nb::for_setter(nb::arg("value").none()),                              \
        "The compiled megakernel this descriptor launches. Owned, not "        \
        "borrowed. Assign None to release.")

void bind_launch(nb::module_& m) {
    {
        using Desc = mk::NmcLaunchDesc;
        nb::class_<Desc>(m, "NmcLaunchDesc")
            .def("__init__", &descriptor_init<Desc>)
            .def_prop_ro("_raw_bytes", &descriptor_raw_bytes<Desc>)
            RW(num_sms) RW(stream_u64) RW(inst_buf) RW(num_inst_per_sm)
            RW(max_inst) RW(W_upgate) RW(W_down) RW(W_qkv)
            RW(W_oproj) RW(W_lmhead) RW(W_router) RW(W_moe_upgate)
            RW(W_moe_down) RW(K_pool) RW(V_pool) RW(num_phys_pages)
            RW(page_table) RW(cache_seqlens) RW(row_active) RW(max_pages_per_seq)
            RW(page_block_size) RW(cos_table) RW(sin_table) RW(k_cache)
            RW(v_cache) RW(upgate_scratch) RW(silu_out) RW(q_out)
            RW(o_proj_in) RW(x_resid) RW(o_partial) RW(lse_partial)
            RW(lm_logits) RW(rmsnorm_gamma) RW(x_raw) RW(x_attn)
            RW(x_ffn) RW(router_logits) RW(moe_x) RW(moe_hidden)
            RW(moe_down_out) RW(topk_experts) RW(topk_scores) RW(topk_local_slots)
            RW(route_row_for_token) RW(routed_token_ids) RW(routed_scores) RW(expert_counts)
            RW(expert_offsets) RW(max_routed) RW(moe_up_task_words) RW(moe_down_task_words)
            RW(moe_up_task_count) RW(moe_down_task_count) RW(moe_up_task_head) RW(moe_down_task_head)
            RW(max_moe_tasks) RW(bar_upgate)
            RW(bar_silu) RW(bar_ffn_down) RW(bar_qkv) RW(bar_combine)
            RW(bar_attn) RW(bar_oproj) RW(bar_layer) RW(bar_router)
            RW(bar_topk) RW(bar_route) RW(bar_gather) RW(bar_moe_upgate)
            RW(bar_moe_down) RW(prof_buf) RW(num_layers) RW(BS)
            RW(D) RW(Dff) RW(Hq) RW(Hkv)
            RW(head_dim) RW(num_splits) RW(timing) RW(attn_queue_words)
            RW(attn_queue_heads) RW(attn_queue_len)
            RW(attn_num_splits);
    }

    {
        // Bound so Python can build the `zero_regions` vectors. Constructible
        // with both fields because that is how the whole list is built in one
        // comprehension; the descriptors' vector fields are assigned whole.
        using Desc = mk::NmcZeroRegion;
        nb::class_<Desc>(m, "NmcZeroRegion")
            .def("__init__", &descriptor_init<Desc>)
            .def("__init__",
                 [](Desc* self, uint64_t ptr, uint64_t bytes) {
                     new (self) Desc{ptr, bytes};
                 },
                 nb::arg("ptr"), nb::arg("bytes"))
            RW(ptr) RW(bytes);
    }

    {
        using Desc = mk::NmcRuntimeScheduleVariant;
        nb::class_<Desc>(m, "NmcRuntimeScheduleVariant")
            .def("__init__", &descriptor_init<Desc>)
            RW(inst_buf) RW(num_inst_per_sm) RW_JIT_HANDLE() RW(max_inst)
            RW(bucket_upper);
    }

    {
        // `launch` is bound by reference (nanobind's def_rw getter uses
        // rv_policy::reference_internal), so `desc.launch.num_sms = x` mutates
        // the embedded struct in place. The test asserts this.
        using Desc = mk::NmcRuntimeGenerateDesc;
        nb::class_<Desc>(m, "NmcRuntimeGenerateDesc")
            .def("__init__", &descriptor_init<Desc>)
            RW(launch) RW(w_ln0) RW(generated_ids)
            RW_VEC(zero_regions, "Device regions zeroed before every step.")
            RW_VEC(eos_token_ids, "Stop tokens; empty disables EOS stopping.")
            RW(generated_lengths) RW(finish_reasons) RW(executed_steps)
            RW_VEC(timing_ms, "Output: per-step kernel ms, sized by the runtime.")
            RW(max_new) RW(start_pos) RW(warmup)
            RW(vocab_size) RW(max_seq_len) RW_JIT_HANDLE()
            RW_VEC(schedule_variants, "Context buckets, ascending by bucket_upper.")
            RW_KV_HANDLE() RW(rms_norm_eps) RW(temperature)
            RW(sampling_seed) RW(timing) RW(token_callback) RW(token_callback_context)
            RW_VEC(start_pos_per_row, "Per-row prompt lengths; empty means uniform.")
            RW(attn_drain) RW(max_attn_splits) RW(min_attn_chunk);
    }

    {
        using Desc = mk::NmcDecodeServiceDesc;
        nb::class_<Desc>(m, "NmcDecodeServiceDesc")
            .def("__init__", &descriptor_init<Desc>)
            RW(launch) RW(w_ln0) RW(generated_ids)
            RW_VEC(zero_regions, "Device regions zeroed before every step.")
            RW_VEC(eos_token_ids, "Stop tokens; empty disables EOS stopping.")
            RW(max_new) RW(vocab_size) RW(max_seq_len)
            RW_KV_HANDLE() RW(rms_norm_eps) RW(d_gen_col) RW(d_temperature)
            RW(d_seed)
            RW_VEC(schedule_variants, "Context buckets, ascending by bucket_upper.")
            RW_JIT_HANDLE() RW(token_callback)
            RW(token_callback_context) RW(attn_drain) RW(max_attn_splits) RW(min_attn_chunk);
    }

    // Compile-time proof that the trampoline is ABI-compatible with the
    // descriptor's function-pointer field. Signature drift in decode/abi.h would
    // otherwise surface as a crash at the first decode step.
    //
    // is_convertible accommodates `invoke` being noexcept, which since C++17 is
    // part of the function type. Dropping noexcept is the only implicit
    // conversion permitted between function pointer types, so drift in the
    // parameter or return types still fails here.
    static_assert(
        std::is_convertible_v<decltype(&TokenCallback::invoke),
                              mk::NmcRuntimeTokenCallback>,
        "TokenCallback::invoke must be ABI-compatible with "
        "NmcRuntimeTokenCallback");

    nb::class_<TokenCallback>(m, "TokenCallback")
        .def(nb::init<nb::callable>(), nb::arg("callback"),
             "Wrap a Python callable as a native token callback. The callable "
             "receives (sampled_tuple, emitted_tuple) and its return value is "
             "ignored; raising aborts the decode loop.")
        .def_prop_ro("function_ptr", &TokenCallback::function_ptr,
                     "Value for NmcRuntime*Desc.token_callback.")
        .def_prop_ro("context_ptr", &TokenCallback::context_ptr,
                     "Value for NmcRuntime*Desc.token_callback_context. Keep "
                     "this object alive for as long as the descriptor is used.")
        .def_prop_ro("last_error", &TokenCallback::last_error,
                     "Message from the most recent failed invocation, or ''.");

    // Test scaffolding. Drives an arbitrary NmcRuntimeTokenCallback from a
    // GIL-free C++ loop, the way the decode thread calls it, and returns the
    // seconds taken for `iterations` invocations. Takes a raw function pointer
    // so any callback implementation can be timed through one call path.
    m.def(
        "_benchmark_token_callback",
        [](uint64_t callback_ptr, uint64_t context_ptr, int batch, int iterations) {
            if (batch <= 0 || iterations <= 0) {
                throw std::invalid_argument("batch and iterations must be positive");
            }
            auto callback =
                reinterpret_cast<mk::NmcRuntimeTokenCallback>(callback_ptr);
            auto* context = reinterpret_cast<void*>(context_ptr);
            std::vector<long long> sampled(static_cast<size_t>(batch), 12345);
            std::vector<int32_t> emitted(static_cast<size_t>(batch), 1);
            double seconds = 0.0;
            {
                nb::gil_scoped_release release;
                const auto started = std::chrono::steady_clock::now();
                for (int i = 0; i < iterations; ++i) {
                    callback(context, sampled.data(), emitted.data(), batch);
                }
                seconds = std::chrono::duration<double>(
                              std::chrono::steady_clock::now() - started)
                              .count();
            }
            return seconds;
        },
        nb::arg("callback_ptr"), nb::arg("context_ptr"), nb::arg("batch"),
        nb::arg("iterations"),
        "Time N token-callback invocations from a GIL-free thread. Test only.");

    // ── GIL policy for everything below ─────────────────────────────────────
    //
    // Every native call runs under `nb::gil_scoped_release`, with two
    // exceptions noted at their definitions. That is deliberately blunter than
    // strictly necessary -- `set_debug` is an atomic store -- but the cost of
    // an unnecessary release is a few dozen nanoseconds, while the cost of a
    // missing one on jit_compile (which shells out to nvcc for tens of
    // seconds) or decode_service_run (which runs for the process lifetime) is
    // a wedged interpreter. A uniform rule also gives the test suite something
    // it can check mechanically.
    //
    // String arguments are taken by `std::string`, not `const char*`. nanobind
    // would hand us a pointer into the Python str's internal buffer, which we
    // must not dereference after dropping the GIL; an owned copy sidesteps it.

    // ── JitKernel ───────────────────────────────────────────────────────────
    nb::class_<mk::JitKernel>(m, "JitKernel",
        "One compiled megakernel variant. Owns the cached .so and its dlopen "
        "handle; the kernel is unloaded when the last holder (a descriptor's "
        "jit_handle, or Python) drops it.")
        .def_static(
            "compile",
            [](int32_t bs, bool enable_profiler, const std::string& repo_root,
               const std::string& config_json) {
                nb::gil_scoped_release release;
                return mk::JitKernel::compile(bs, enable_profiler, repo_root,
                                              config_json);
            },
            nb::arg("bs"), nb::arg("enable_profiler"), nb::arg("repo_root"),
            nb::arg("config_json"),
            "Compile, or fetch from the little_jit cache, the megakernel for "
            "one batch size. Blocks for as long as nvcc takes.")

        .def(
            "decode_launch",
            [](mk::JitKernel& self, const mk::NmcLaunchDesc& desc) {
                // `desc` is a reference into the caller's Python object, which
                // the argument tuple keeps alive for the call, so the pointer
                // stays valid across the GIL release.
                nb::gil_scoped_release release;
                return self.decode_launch(desc);
            },
            nb::arg("desc"),
            "Launch one decode step. Returns CUDA-event milliseconds. The "
            "launch is asynchronous with respect to the device; synchronize "
            "before reading any output tensor.")

        // GIL NOT released: returns a reference to a std::string member.
        .def_prop_ro("artifact_path", &mk::JitKernel::artifact_path,
                     "Filesystem path of the compiled .so.");

    // ── Profiler ────────────────────────────────────────────────────────────
    nb::class_<mk::Profiler>(m, "Profiler",
        "A device profiler buffer. Owns its device allocation, which is freed "
        "when Python drops the object.")
        .def(
            "__init__",
            [](mk::Profiler* self, int32_t num_sms, int32_t max_events) {
                nb::gil_scoped_release release;
                new (self) mk::Profiler(num_sms, max_events);
            },
            nb::arg("num_sms"), nb::arg("max_events"))

        .def_prop_ro(
            "device_ptr",
            [](const mk::Profiler& self) {
                nb::gil_scoped_release release;
                return self.device_ptr();
            },
            "Device address to hand to NmcLaunchDesc.prof_buf.")

        .def(
            "init",
            [](mk::Profiler& self) {
                nb::gil_scoped_release release;
                self.init();
            },
            "Reset the buffer's event cursors.")

        .def(
            "export_to",
            [](mk::Profiler& self, const std::string& filename) {
                nb::gil_scoped_release release;
                self.export_to(filename);
            },
            nb::arg("filename"));

    // ── Diagnostics ─────────────────────────────────────────────────────────
    // GIL NOT released: reads a thread-local std::string and returns. Releasing
    // would be pointless, and worse, it would let another Python thread run
    // between the failing call and this one.
    m.def(
        "last_error", []() { return std::string(mk::last_error()); },
        "Message from the most recent DecodeService.run failure on this "
        "thread. Everything else raises MkAbiError directly, so this is for "
        "diagnostics only.");

    // GIL NOT released: a single relaxed atomic store.
    m.def(
        "set_debug", [](bool enabled) { mk::set_debug(enabled); },
        nb::arg("enabled"),
        "Widen host-side watchdog diagnostics. MUST be called before creating "
        "a decode service: geometry setup only allocates the snapshot buffers "
        "when debug is already on, so flipping it later leaves them missing.");

    // ── Attention drain split policy ────────────────────────────────────────
    m.def(
        "attn_drain_splits",
        [](const std::vector<int32_t>& cache_seqlens, int32_t hkv,
           int32_t num_sms, int32_t page_block, int32_t max_attn_splits,
           int32_t min_attn_chunk, int32_t oversub_k) {
            nb::gil_scoped_release release;
            return mk::attn_drain_splits(cache_seqlens, hkv, num_sms,
                                         page_block, max_attn_splits,
                                         min_attn_chunk, oversub_k);
        },
        nb::arg("cache_seqlens"), nb::arg("hkv"), nb::arg("num_sms"),
        nb::arg("page_block"), nb::arg("max_attn_splits"),
        nb::arg("min_attn_chunk"), nb::arg("oversub_k"),
        "Per-row split counts from the native drain policy.\n\n"
        "This is a test hook: it exposes the C++ policy that actually runs so a "
        "test can compare it with the pure-Python policy in "
        "decode_schedule._attn_drain_splits_by_row.");

    // ── One-shot lockstep decode runtime ────────────────────────────────────
    m.def(
        "runtime_generate",
        [](mk::NmcRuntimeGenerateDesc& desc) {
            nb::gil_scoped_release release;
            mk::runtime_generate(desc);
        },
        nb::arg("desc"),
        "Run the whole native decode loop. Blocks for the entire generation; "
        "the device buffers the descriptor points at must outlive the call.\n\n"
        "Writes `desc.executed_steps` and, when `desc.timing` is set, "
        "`desc.timing_ms` back into the descriptor you passed in.");
}

#undef RW
#undef RW_KV_HANDLE
#undef RW_JIT_HANDLE

}  // namespace mk_bindings
