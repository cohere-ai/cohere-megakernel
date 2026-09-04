// nanobind bindings for the paged-KV / prefix-cache classes in src/kv/cache.h.

#include "bindings.h"

#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "kv/cache.h"

namespace nb = nanobind;

namespace mk_bindings {
namespace {

// Prefix keys are raw SHA-256 material and may contain NUL bytes, so they can
// never round-trip through a C string. Copy into a std::string (which keeps an
// explicit length) before releasing the GIL.
std::string copy_key(const nb::bytes& key) {
    return std::string(key.c_str(), key.size());
}

// Reject a host array whose length disagrees with the handle's batch size.
// The callee indexes [0, BS) unconditionally, so this is the difference
// between a diagnosable error and reading past the vector's buffer.
// std::invalid_argument (-> ValueError) marks a check made by this binding
// layer; KvError (-> KvAbiError) is what the native layer itself raises.
void check_batch_len(const std::vector<int32_t>& values, int32_t batch_size,
                     const char* name) {
    if (values.size() != static_cast<size_t>(batch_size)) {
        throw std::invalid_argument(
            std::string(name) + ": expected " + std::to_string(batch_size) +
            " values, one per batch row, got " +
            std::to_string(values.size()));
    }
}

}  // namespace

void bind_kv(nb::module_& m) {
    // Surfaces in Python as `native.ext().kv.KvAbiError`, and is caught
    // under that name. It subclasses RuntimeError, so `except RuntimeError`
    // also works. Constructing this registers the C++ -> Python translator for
    // kv_pool::KvError and installs the type in `m`; the module namespace owns
    // the reference from here on.
    const nb::exception<kv_pool::KvError> kv_abi_error(m, "KvAbiError",
                                                     PyExc_RuntimeError);
    (void)kv_abi_error;

    // ── Descriptors ─────────────────────────────────────────────────────────
    // The C++ structs from kv/cache.h are bound directly, so Python mutates the
    // actual fields and the two sides cannot disagree about layout.
    nb::class_<kv_pool::KvHandleDesc>(m, "KvHandleDesc")
        .def(nb::init<>())
        // Keyword constructor written with C++20 designated initializers so
        // the compiler checks each name-to-field mapping. Binding these 11
        // same-ish scalars positionally through nb::init<...> would let a
        // reordered field in kv/cache.h silently remap every keyword.
        .def(
            "__init__",
            [](kv_pool::KvHandleDesc* self, uint64_t page_table,
               uint64_t cache_seqlens, uint64_t row_active, int32_t num_layers,
               int32_t BS, int32_t max_pages_per_seq, int32_t num_phys_pages,
               int32_t page_block_size, int32_t start_pos, int32_t sw_size,
               int32_t sw_pattern) {
                new (self) kv_pool::KvHandleDesc{
                    .page_table = page_table,
                    .cache_seqlens = cache_seqlens,
                    .row_active = row_active,
                    .num_layers = num_layers,
                    .BS = BS,
                    .max_pages_per_seq = max_pages_per_seq,
                    .num_phys_pages = num_phys_pages,
                    .page_block_size = page_block_size,
                    .start_pos = start_pos,
                    .sw_size = sw_size,
                    .sw_pattern = sw_pattern,
                };
            },
            nb::arg("page_table"), nb::arg("cache_seqlens"),
            nb::arg("row_active"), nb::arg("num_layers"), nb::arg("BS"),
            nb::arg("max_pages_per_seq"), nb::arg("num_phys_pages"),
            nb::arg("page_block_size"), nb::arg("start_pos"),
            nb::arg("sw_size"), nb::arg("sw_pattern"))
        .def_rw("page_table", &kv_pool::KvHandleDesc::page_table)
        .def_rw("cache_seqlens", &kv_pool::KvHandleDesc::cache_seqlens)
        .def_rw("row_active", &kv_pool::KvHandleDesc::row_active)
        .def_rw("num_layers", &kv_pool::KvHandleDesc::num_layers)
        .def_rw("BS", &kv_pool::KvHandleDesc::BS)
        .def_rw("max_pages_per_seq", &kv_pool::KvHandleDesc::max_pages_per_seq)
        .def_rw("num_phys_pages", &kv_pool::KvHandleDesc::num_phys_pages)
        .def_rw("page_block_size", &kv_pool::KvHandleDesc::page_block_size)
        .def_rw("start_pos", &kv_pool::KvHandleDesc::start_pos)
        .def_rw("sw_size", &kv_pool::KvHandleDesc::sw_size)
        .def_rw("sw_pattern", &kv_pool::KvHandleDesc::sw_pattern);

    nb::class_<kv_pool::KvStats>(m, "KvStats")
        .def(nb::init<>())
        .def_ro("total", &kv_pool::KvStats::total)
        .def_ro("allocated", &kv_pool::KvStats::allocated)
        .def_ro("peak_allocated", &kv_pool::KvStats::peak_allocated)
        .def_ro("free", &kv_pool::KvStats::free)
        .def("__repr__", [](const kv_pool::KvStats& s) {
            return "KvStats(total=" + std::to_string(s.total) +
                   ", allocated=" + std::to_string(s.allocated) +
                   ", peak_allocated=" + std::to_string(s.peak_allocated) +
                   ", free=" + std::to_string(s.free) + ")";
        });

    nb::class_<kv_pool::PrefixCacheStats>(m, "PrefixCacheStats")
        .def(nb::init<>())
        .def_ro("entries", &kv_pool::PrefixCacheStats::entries)
        .def_ro("total_blocks", &kv_pool::PrefixCacheStats::total_blocks)
        .def_ro("active_blocks", &kv_pool::PrefixCacheStats::active_blocks)
        .def_ro("free_blocks", &kv_pool::PrefixCacheStats::free_blocks)
        .def_ro("cached_blocks", &kv_pool::PrefixCacheStats::cached_blocks)
        .def_ro("free_cached_blocks", &kv_pool::PrefixCacheStats::free_cached_blocks)
        .def_ro("evicted_blocks", &kv_pool::PrefixCacheStats::evicted_blocks);

    // ── The handle itself ───────────────────────────────────────────────────
    nb::class_<kv_pool::KvHandle>(m, "KvHandle")
        .def(
            "__init__",
            [](kv_pool::KvHandle* self, const kv_pool::KvHandleDesc& desc) {
                nb::gil_scoped_release release;
                new (self) kv_pool::KvHandle(desc);
            },
            nb::arg("desc"))

        .def_prop_ro("batch_size", &kv_pool::KvHandle::batch_size)
        .def_prop_ro("num_layers", &kv_pool::KvHandle::num_layers)
        .def_prop_ro("max_pages_per_seq", &kv_pool::KvHandle::max_pages_per_seq)

        // ── Prefill / decode stepping ───────────────────────────────────────
        .def(
            "prepare_prefill",
            [](kv_pool::KvHandle& self,
               const std::vector<int32_t>& prompt_lengths, uint64_t stream) {
                check_batch_len(prompt_lengths, self.batch_size(),
                                "prompt_lengths");
                nb::gil_scoped_release release;
                self.prepare_prefill(prompt_lengths.data(), stream);
            },
            nb::arg("prompt_lengths"), nb::arg("stream"))

        .def(
            "prepare_prefill_masked",
            [](kv_pool::KvHandle& self,
               const std::vector<int32_t>& prompt_lengths,
               const std::vector<int32_t>& active, uint64_t stream) {
                check_batch_len(prompt_lengths, self.batch_size(),
                                "prompt_lengths");
                check_batch_len(active, self.batch_size(), "active");
                nb::gil_scoped_release release;
                self.prepare_prefill_masked(prompt_lengths.data(),
                                            active.data(), stream);
            },
            nb::arg("prompt_lengths"), nb::arg("active"), nb::arg("stream"))

        .def(
            "step_prefill_chunk",
            [](kv_pool::KvHandle& self, int32_t layer_idx, int32_t pos_start,
               int32_t pos_end, uint64_t stream) {
                nb::gil_scoped_release release;
                self.step_prefill_chunk(layer_idx, pos_start, pos_end, stream);
            },
            nb::arg("layer_idx"), nb::arg("pos_start"), nb::arg("pos_end"),
            nb::arg("stream"))

        .def(
            "step_prefill_chunk_masked",
            [](kv_pool::KvHandle& self, int32_t layer_idx, int32_t pos_start,
               int32_t pos_end, const std::vector<int32_t>& active,
               uint64_t stream) {
                check_batch_len(active, self.batch_size(), "active");
                nb::gil_scoped_release release;
                self.step_prefill_chunk_masked(layer_idx, pos_start, pos_end,
                                               active.data(), stream);
            },
            nb::arg("layer_idx"), nb::arg("pos_start"), nb::arg("pos_end"),
            nb::arg("active"), nb::arg("stream"))

        .def(
            "step_decode_positions",
            [](kv_pool::KvHandle& self, const std::vector<int32_t>& positions,
               const std::vector<int32_t>& active, uint64_t stream) {
                check_batch_len(positions, self.batch_size(), "positions");
                check_batch_len(active, self.batch_size(), "active");
                nb::gil_scoped_release release;
                self.step_decode_positions(positions.data(), active.data(),
                                           stream);
            },
            nb::arg("positions"), nb::arg("active"), nb::arg("stream"))

        .def(
            "set_cache_seqlens",
            [](kv_pool::KvHandle& self, const std::vector<int32_t>& lengths,
               uint64_t stream) {
                check_batch_len(lengths, self.batch_size(), "lengths");
                nb::gil_scoped_release release;
                self.set_cache_seqlens(lengths.data(), stream);
            },
            nb::arg("lengths"), nb::arg("stream"))

        .def(
            "free_row",
            [](kv_pool::KvHandle& self, int32_t row, uint64_t stream) {
                nb::gil_scoped_release release;
                self.free_row(row, stream);
            },
            nb::arg("row"), nb::arg("stream"))

        // Returns a NEW handle; `self` stays valid until the caller drops it.
        .def(
            "rebind",
            [](kv_pool::KvHandle& self, const std::vector<int32_t>& src_rows,
               int32_t new_batch_size, uint64_t new_page_table,
               uint64_t new_cache_seqlens, uint64_t new_row_active,
               uint64_t stream) {
                // src_rows is indexed by DESTINATION row, so it is sized by
                // new_batch_size rather than this handle's batch size.
                check_batch_len(src_rows, new_batch_size, "src_rows");
                nb::gil_scoped_release release;
                return self.rebind(src_rows.data(), new_batch_size,
                                   new_page_table, new_cache_seqlens,
                                   new_row_active, stream);
            },
            nb::arg("src_rows"), nb::arg("new_batch_size"),
            nb::arg("new_page_table"), nb::arg("new_cache_seqlens"),
            nb::arg("new_row_active"), nb::arg("stream"))

        // ── Pool statistics ─────────────────────────────────────────────────
        .def("stats",
             [](const kv_pool::KvHandle& self) {
                 nb::gil_scoped_release release;
                 return self.stats();
             })

        .def("free_blocks",
             [](const kv_pool::KvHandle& self) {
                 nb::gil_scoped_release release;
                 return self.free_blocks();
             })

        // ── Prefix cache, per-handle half ───────────────────────────────────
        // Keys must be exactly 32 bytes of caller-computed SHA-256 material.
        // The length passed below is measured from the bytes object itself, so
        // it always describes the blob truthfully, and kv/cache.cpp
        // (parse_block_hash) rejects any other length. A wrong-length key is
        // therefore caller error, not a silent cache miss.
        .def(
            "prefix_attach",
            [](kv_pool::KvHandle& self, const nb::bytes& key, int32_t row,
               int32_t logical_block, int32_t required_from_block,
               uint64_t stream) {
                const std::string key_bytes = copy_key(key);
                nb::gil_scoped_release release;
                return self.prefix_attach(
                    key_bytes.data(), static_cast<int32_t>(key_bytes.size()),
                    row, logical_block, required_from_block, stream);
            },
            nb::arg("key"), nb::arg("row"), nb::arg("logical_block"),
            nb::arg("required_from_block"), nb::arg("stream"))

        .def(
            "prefix_attach_batch",
            [](kv_pool::KvHandle& self, const nb::bytes& keys, int32_t key_len,
               const std::vector<int32_t>& rows,
               const std::vector<int32_t>& logical_blocks,
               const std::vector<int32_t>& required_from_blocks,
               uint64_t stream) {
                const size_t count = rows.size();
                if (logical_blocks.size() != count ||
                    required_from_blocks.size() != count) {
                    throw std::invalid_argument(
                        "prefix_attach_batch: rows, logical_blocks and "
                        "required_from_blocks must have equal length");
                }
                // The native side strides `keys` by key_len for `count`
                // entries; a short blob would read past the end of the Python
                // bytes object.
                if (keys.size() != count * static_cast<size_t>(key_len)) {
                    throw std::invalid_argument(
                        "prefix_attach_batch: key blob must hold exactly "
                        "count * key_len bytes");
                }
                const std::string key_bytes = copy_key(keys);
                nb::gil_scoped_release release;
                return self.prefix_attach_batch(
                    key_bytes.data(), key_len, rows.data(),
                    logical_blocks.data(), required_from_blocks.data(),
                    static_cast<int32_t>(count), stream);
            },
            nb::arg("keys"), nb::arg("key_len"), nb::arg("rows"),
            nb::arg("logical_blocks"), nb::arg("required_from_blocks"),
            nb::arg("stream"))

        .def(
            "prefix_can_attach",
            [](const kv_pool::KvHandle& self, const nb::bytes& key,
               int32_t logical_block, int32_t required_from_block) {
                const std::string key_bytes = copy_key(key);
                nb::gil_scoped_release release;
                return self.prefix_can_attach(
                    key_bytes.data(), static_cast<int32_t>(key_bytes.size()),
                    logical_block, required_from_block);
            },
            nb::arg("key"), nb::arg("logical_block"),
            nb::arg("required_from_block"))

        .def(
            "prefix_publish",
            [](kv_pool::KvHandle& self, const nb::bytes& key, int32_t row,
               int32_t logical_block) {
                const std::string key_bytes = copy_key(key);
                nb::gil_scoped_release release;
                return self.prefix_publish(
                    key_bytes.data(), static_cast<int32_t>(key_bytes.size()),
                    row, logical_block);
            },
            nb::arg("key"), nb::arg("row"), nb::arg("logical_block"));

    // ── Prefix cache, process-global half ───────────────────────────────────
    // No handle is involved: one prefix cache serves every handle.
    m.def("prefix_cache_clear", []() {
        nb::gil_scoped_release release;
        kv_pool::prefix_cache_clear();
    });

    m.def("prefix_cache_size", []() {
        nb::gil_scoped_release release;
        return kv_pool::prefix_cache_size();
    });

    m.def("prefix_cache_stats", []() {
        nb::gil_scoped_release release;
        return kv_pool::prefix_cache_stats();
    });
}

}  // namespace mk_bindings
