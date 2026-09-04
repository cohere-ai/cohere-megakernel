#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>

// Paged-KV cache for the release runtime.
//
// DESIGN CREDIT: the paging scheme here follows vLLM. Fixed-size KV blocks, a
// per-row block table indexed by logical block, a reference-counted global
// block pool, and a prefix cache keyed by a hash of block content are all from
// PagedAttention (Kwon et al., "Efficient Memory Management for Large Language
// Model Serving with PagedAttention", SOSP 2023) and its vLLM v1 realisation in
// vllm/v1/core/kv_cache_utils.py and vllm/v1/core/block_pool.py. The structure
// names in kv/cache.cpp deliberately echo those files.
//
// A KvHandle owns row-to-physical-page references, while the physical block
// pool and prefix cache are process-global. Global pool/cache bookkeeping is
// serialized internally; callers must serialize operations on each handle and
// mutate rows only while its decode service is paused. Every handle in one
// process must use the same caller-owned physical K/V arena.
//
// Device pointers uint64_t, on purpose. Page tables, seqlen vectors and
// the K/V pools are Torch tensors owned by Python; they arrive here as plain
// addresses so this header stays free of any CUDA or torch type and remains
// compilable by a plain C++ compiler (src/bindings/*.cpp are not built by
// nvcc). They are borrowed and must outlive the handle.
namespace kv_pool {

struct KvHandleDesc {
    uint64_t page_table;
    uint64_t cache_seqlens;
    uint64_t row_active;
    int32_t num_layers;
    int32_t BS;
    int32_t max_pages_per_seq;
    int32_t num_phys_pages;
    int32_t page_block_size;
    int32_t start_pos;
    // Total tokens in the sliding-attention window. Zero disables eviction.
    int32_t sw_size;
    // MoE/attention hybrid period: dense/full at (layer % sw_pattern) == 0,
    // sliding otherwise. Derived from model ``layer_types`` (NMC: period 4).
    // Do NOT pass ``prefix_dense_sliding_window_pattern`` here -- that config
    // flag only controls force-prefix RoPE on early dense layers.
    int32_t sw_pattern;
};

struct KvStats {
    int32_t total;
    int32_t allocated;
    int32_t peak_allocated;
    int32_t free;
};

// Host-only KV metadata snapshot for crash/watchdog dumps.
//
// NEVER touches CUDA: after a wedged kernel, any device memcpy can block
// forever. The page_table / cache_seqlens / row_active fields are copies of the
// host mirrors (last values flushed or about to be flushed by the step calls).
// Device-side corruption that never made it back to the host will NOT appear
// here — call that out in the dump note.
//
// Output buffers may be null to query geometry+stats only. When non-null,
// page_table_cap / BS must be large enough or the call throws after still
// writing the geometry fields (so the caller can allocate and retry).
struct KvHostSnapshot {
    int32_t num_layers;
    int32_t BS;
    int32_t max_pages_per_seq;
    int32_t num_phys_pages;
    int32_t page_block_size;
    int32_t sw_size;
    int32_t sw_pattern;
    // Number of int32 entries in page_table_host (== layers * BS * max_pages).
    int32_t page_table_elems;
    KvStats stats;
    // Optional outputs (null = skip that copy).
    int32_t* page_table_out;
    int32_t page_table_cap;  // capacity of page_table_out in ints
    int32_t* cache_seqlens_out;
    int32_t* row_active_out;
};

struct PrefixCacheStats {
    int32_t entries;
    int32_t total_blocks;
    int32_t active_blocks;
    int32_t free_blocks;
    int32_t cached_blocks;
    int32_t free_cached_blocks;
    int32_t evicted_blocks;
};

// Every failure below is reported by throwing this. src/bindings/kv.cpp
// surfaces it in Python as `KvAbiError`, a RuntimeError subclass.
class KvError : public std::runtime_error {
 public:
    using std::runtime_error::runtime_error;
};

// One page-table geometry over the process-global block pool.
//
// Not copyable or movable: the address of a live handle is handed to the CUDA
// runtime through NmcDecodeServiceDesc::kv_handle, so it must not change.
class KvHandle {
 public:
    // Reads the caller-owned device tables named by `desc`. Throws KvError if
    // the geometry is inconsistent or the pool cannot be initialised.
    explicit KvHandle(const KvHandleDesc& desc);
    // Releases this handle's block references only, never the caller's tensors.
    ~KvHandle();

    KvHandle(const KvHandle&) = delete;
    KvHandle& operator=(const KvHandle&) = delete;
    KvHandle(KvHandle&&) = delete;
    KvHandle& operator=(KvHandle&&) = delete;

    // ── Prefill ─────────────────────────────────────────────────────────────
    // `prompt_lengths` and `active` are host arrays of length BS(). These
    // methods read exactly BS entries from a bare pointer, so the caller
    // guarantees the length; the Python binding enforces it against BS()
    // (src/bindings/kv.cpp `check_batch_len`).
    void prepare_prefill(const int32_t* prompt_lengths, uint64_t stream_u64);
    void prepare_prefill_masked(const int32_t* prompt_lengths,
                                const int32_t* active, uint64_t stream_u64);
    void step_prefill_chunk(int32_t layer_idx, int32_t pos_start,
                            int32_t pos_end, uint64_t stream_u64);
    void step_prefill_chunk_masked(int32_t layer_idx, int32_t pos_start,
                                   int32_t pos_end, const int32_t* active,
                                   uint64_t stream_u64);

    // ── Decode ──────────────────────────────────────────────────────────────
    // Allocates/evicts blocks and asynchronously updates the borrowed device
    // tensors on `stream_u64`. `positions`/`active` are host int32[BS].
    void step_decode_positions(const int32_t* positions, const int32_t* active,
                               uint64_t stream_u64);
    void set_cache_seqlens(const int32_t* lengths, uint64_t stream_u64);
    void free_row(int32_t row, uint64_t stream_u64);

    // Retain selected block references in a newly allocated geometry. `this`
    // remains valid on success; destroy it only after the caller has committed
    // its service to the returned handle. Destination row `d` takes source row
    // `src_rows[d]`, where -1 leaves the row empty; `src_rows` is host
    // int32[new_batch_size].
    std::unique_ptr<KvHandle> rebind(const int32_t* src_rows,
                                     int32_t new_batch_size,
                                     uint64_t new_page_table,
                                     uint64_t new_cache_seqlens,
                                     uint64_t new_row_active,
                                     uint64_t stream_u64);

    // ── Introspection ───────────────────────────────────────────────────────
    KvStats stats() const;
    // Free physical blocks in the shared process-wide pool. Exposed on the
    // handle because that is how callers reach the pool; KvStats::free reports
    // the same number.
    int32_t free_blocks() const;
    // Host-only snapshot for watchdog / FATAL dumps. See KvHostSnapshot.
    // Touches no CUDA, so it is safe to call after a wedged kernel.
    void host_snapshot(KvHostSnapshot& io) const;

    int32_t batch_size() const;
    int32_t num_layers() const;
    int32_t max_pages_per_seq() const;

    // ── Prefix cache (per-handle half) ──────────────────────────────────────
    // Keys must be exactly 32 bytes of caller-computed SHA-256 material. The
    // cache adds no model namespace; callers must include one in their keys or
    // clear the cache before changing the weights behind an arena.
    //
    // These return false for a genuine cache miss / no-op and throw KvError for
    // invalid input, so a miss can never be confused with an error.
    bool prefix_attach(const char* key, int32_t key_len, int32_t row,
                       int32_t logical_block, int32_t required_from_block,
                       uint64_t stream_u64);
    // Validates all entries before mutating any row, then uploads the page
    // table once. Duplicate (row, logical_block) destinations are invalid.
    // `keys` is a tightly packed [count, key_len] byte array; the three int32
    // arrays each have `count` entries. Returns false if ANY entry is
    // unavailable, having changed nothing.
    bool prefix_attach_batch(const char* keys, int32_t key_len,
                             const int32_t* rows, const int32_t* logical_blocks,
                             const int32_t* required_from_blocks, int32_t count,
                             uint64_t stream_u64);
    bool prefix_can_attach(const char* key, int32_t key_len,
                           int32_t logical_block,
                           int32_t required_from_block) const;
    bool prefix_publish(const char* key, int32_t key_len, int32_t row,
                        int32_t logical_block);

    // Host mirror of the page table, defined in kv/cache.cpp. Opaque here so
    // this header does not leak <vector> and the pool internals.
    struct State;

 private:
    // Adopts an already-populated State. Used only by rebind(), which builds
    // the new geometry by copying the source handle's. (The public constructor
    // reads the device tables instead.)
    explicit KvHandle(std::unique_ptr<State> state);

    std::unique_ptr<State> s_;
};

// ── Prefix cache (process-global half) ──────────────────────────────────────
// No handle is involved: one prefix cache serves every handle in the process.
void prefix_cache_clear();
int32_t prefix_cache_size();
PrefixCacheStats prefix_cache_stats();

}  // namespace kv_pool
