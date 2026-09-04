// Paged KV metadata manager. Owns no model weights or device tensors.
//
// The block pool, block hashing and prefix cache below follow vLLM's design;
// see the DESIGN CREDIT note in kv/cache.h. KVCacheBlock, BlockHash and the
// acquire / retain / release refcount discipline correspond to the types of the
// same names in vllm/v1/core/kv_cache_utils.py and vllm/v1/core/block_pool.py.
//
// Sliding-window eviction (kv_evict_below and the prefill-retention rule beside
// it) is specific to this model's hybrid full/sliding layer pattern.
#include "kv/cache.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define MK_CUDA_CHECK(call) do {                                            \
    cudaError_t error__ = (call);                                           \
    if (error__ != cudaSuccess) {                                           \
        throw KvError(std::string("CUDA failure: ") +                       \
                      cudaGetErrorString(error__));                          \
    }                                                                        \
} while (0)

namespace kv_pool {
namespace {
// Prefix keys are SHA-256 digests. This protocol value is fixed by the Python
// prefix-key generator; change both sides together if the hashing scheme moves.
constexpr int32_t kPrefixHashBytes = 32;
}

// Host mirrors of the borrowed device tables, plus the per-handle bookkeeping
// the pool cannot hold. Named `KvHandle::State` so kv/cache.h can forward
// declare it; the short alias keeps the free helpers below readable.
struct KvHandle::State {
    std::vector<int> page_table_host;
    std::vector<int> cache_seqlens_host;
    std::vector<int> row_active_host;
    std::vector<int> sliding_evict_until;
    size_t page_table_elems = 0;
    int num_layers = 0;
    int BS = 0;
    int max_pages_per_seq = 0;
    int num_phys_pages = 0;
    int page_block_size = 0;
    int sw_size = 0;
    int sw_pattern = 0;
    uint64_t page_table = 0;
    uint64_t cache_seqlens = 0;
    uint64_t row_active = 0;
    int peak_allocated = 0;
};

using RuntimeKvState = KvHandle::State;

struct BlockHash {
    std::array<uint8_t, kPrefixHashBytes> bytes{};

    bool operator==(const BlockHash& other) const {
        return bytes == other.bytes;
    }
};

struct BlockHashHasher {
    size_t operator()(const BlockHash& h) const {
        // FNV-1a over the fixed SHA-256 bytes. This is only the host hash-table
        // hash; collision correctness is still guarded by BlockHash equality.
        size_t v = 1469598103934665603ull;
        for (uint8_t b : h.bytes) {
            v ^= (size_t)b;
            v *= 1099511628211ull;
        }
        return v;
    }
};

static BlockHash parse_block_hash(const char* key, int32_t key_len) {
    if (!key || key_len != kPrefixHashBytes) {
        throw KvError("prefix cache keys must be SHA-256-sized");
    }
    BlockHash h;
    std::memcpy(h.bytes.data(), key, h.bytes.size());
    return h;
}

struct KVCacheBlock {
    int block_id = -1;
    BlockHash block_hash{};
    bool has_hash = false;
    int ref_cnt = 0;
    KVCacheBlock* prev_free_block = nullptr;
    KVCacheBlock* next_free_block = nullptr;
    bool in_free_queue = false;
};

struct PrefixCacheEntry {
    // One entry covers a token block hash across all layers.  Sliding-window
    // layers may not retain older logical blocks, so a per-layer page can be
    // absent and is represented as -1.
    std::vector<int> block_ids;
};

static std::mutex g_prefix_mu;
static std::unordered_map<BlockHash, PrefixCacheEntry, BlockHashHasher> g_prefix_cache;

struct KVBlockPool {
    std::vector<KVCacheBlock> blocks;
    KVCacheBlock* free_head = nullptr;
    KVCacheBlock* free_tail = nullptr;
    int free_count = 0;
    int peak_active = 0;
    int evicted_blocks = 0;

    void clear() {
        blocks.clear();
        free_head = nullptr;
        free_tail = nullptr;
        free_count = 0;
        peak_active = 0;
        evicted_blocks = 0;
    }

    void init_once(int num_blocks) {
        if (!blocks.empty()) {
            if ((int)blocks.size() != num_blocks) {
                throw KvError("KV block pool size changed; restart the engine to resize the arena");
            }
            return;
        }
        blocks.resize(num_blocks);
        for (int i = 0; i < num_blocks; ++i) {
            blocks[i].block_id = i;
            push_free_tail(&blocks[i]);
        }
    }

    KVCacheBlock& at(int block_id) {
        if (block_id < 0 || block_id >= (int)blocks.size()) {
            throw KvError("KV block id out of range");
        }
        return blocks[block_id];
    }

    void push_free_tail(KVCacheBlock* block) {
        if (!block || block->in_free_queue || block->ref_cnt != 0) return;
        block->prev_free_block = free_tail;
        block->next_free_block = nullptr;
        if (free_tail) free_tail->next_free_block = block;
        else free_head = block;
        free_tail = block;
        block->in_free_queue = true;
        ++free_count;
    }

    void remove_from_free_queue(KVCacheBlock* block) {
        if (!block || !block->in_free_queue) return;
        if (block->prev_free_block) block->prev_free_block->next_free_block = block->next_free_block;
        else free_head = block->next_free_block;
        if (block->next_free_block) block->next_free_block->prev_free_block = block->prev_free_block;
        else free_tail = block->prev_free_block;
        block->prev_free_block = nullptr;
        block->next_free_block = nullptr;
        block->in_free_queue = false;
        --free_count;
    }

    KVCacheBlock* pop_free_head() {
        KVCacheBlock* block = free_head;
        if (!block) return nullptr;
        remove_from_free_queue(block);
        return block;
    }

    int active_count() const {
        return (int)blocks.size() - free_count;
    }

    void update_peak() {
        peak_active = std::max(peak_active, active_count());
    }
};

static KVBlockPool g_kv_block_pool;

static void prefix_cache_remove_block_nolock(KVCacheBlock& block) {
    if (!block.has_hash) return;
    auto it = g_prefix_cache.find(block.block_hash);
    if (it != g_prefix_cache.end()) {
        bool has_any = false;
        for (int& id : it->second.block_ids) {
            if (id == block.block_id) id = -1;
            if (id >= 0) has_any = true;
        }
        if (!has_any) g_prefix_cache.erase(it);
    }
    block.has_hash = false;
    block.block_hash = BlockHash{};
    ++g_kv_block_pool.evicted_blocks;
}

static KVCacheBlock& kv_acquire_block_nolock() {
    KVCacheBlock* block = g_kv_block_pool.pop_free_head();
    if (!block) {
        throw KvError("KV cache OOM while allocating blocks");
    }
    if (block->has_hash) {
        prefix_cache_remove_block_nolock(*block);
    }
    block->ref_cnt = 1;
    g_kv_block_pool.update_peak();
    return *block;
}

static void kv_retain_block_nolock(int block_id) {
    KVCacheBlock& block = g_kv_block_pool.at(block_id);
    if (block.ref_cnt == 0) {
        g_kv_block_pool.remove_from_free_queue(&block);
    }
    ++block.ref_cnt;
    g_kv_block_pool.update_peak();
}

static void kv_release_block_nolock(int block_id) {
    KVCacheBlock& block = g_kv_block_pool.at(block_id);
    if (block.ref_cnt <= 0) {
        throw KvError("KV block refcount underflow");
    }
    --block.ref_cnt;
    if (block.ref_cnt == 0) {
        g_kv_block_pool.push_free_tail(&block);
    }
}

// NMC hybrid attention: a layer is full/dense when ``(layer % sw_pattern) == 0``
// (layers 0, P, 2P, ...) and sliding otherwise. ``sw_pattern`` is the
// MoE/attention period derived from ``layer_types``. The config's
// ``prefix_dense_sliding_window_pattern`` is a different quantity that only
// forces RoPE on early dense layers, so it must not be substituted here.
static inline bool kv_layer_is_sliding(const RuntimeKvState& kv, int layer) {
    return (layer % kv.sw_pattern) != 0;
}

static bool prefix_entry_can_attach(
    const RuntimeKvState& kv,
    const PrefixCacheEntry& entry,
    int logical_block,
    int required_from_block) {
    if ((int)entry.block_ids.size() != kv.num_layers) {
        throw KvError("prefix cache entry has incompatible layer count");
    }
    for (int li = 0; li < kv.num_layers; ++li) {
        const int block_id = entry.block_ids[li];
        if (block_id >= 0) {
            const KVCacheBlock& block = g_kv_block_pool.blocks.at(block_id);
            if (block.has_hash) continue;
        }
        // Sliding layers may lack blocks wholly below the SWA window; dense
        // layers must have every logical block present for a valid attach.
        if (!kv_layer_is_sliding(kv, li) || logical_block >= required_from_block) {
            return false;
        }
    }
    return true;
}

static size_t checked_product(size_t left, size_t right, const char* label) {
    if (right != 0 && left > std::numeric_limits<size_t>::max() / right) {
        throw KvError(std::string(label) + " size overflow");
    }
    return left * right;
}

static size_t checked_page_table_elems(
    int num_layers,
    int batch_size,
    int max_pages_per_seq) {
    size_t elements = checked_product(
        static_cast<size_t>(num_layers),
        static_cast<size_t>(batch_size),
        "page table");
    elements = checked_product(
        elements,
        static_cast<size_t>(max_pages_per_seq),
        "page table");
    if (elements > std::numeric_limits<size_t>::max() / sizeof(int)) {
        throw KvError("page table byte size overflow");
    }
    return elements;
}

static void runtime_kv_init_from_ptrs(
    RuntimeKvState& kv,
    uint64_t page_table,
    uint64_t cache_seqlens,
    uint64_t row_active,
    int num_layers,
    int BS,
    int max_pages_per_seq,
    int num_phys_pages,
    int page_block_size,
    int start_pos,
    int sw_size,
    int sw_pattern) {
    if (page_table == 0) {
        throw KvError("KV page_table pointer must be nonzero");
    }
    if (num_layers <= 0 || BS <= 0 || max_pages_per_seq <= 0 ||
        num_phys_pages <= 0 || page_block_size <= 0) {
        throw KvError("KV dimensions and page counts must be positive");
    }
    if (start_pos < 0 || sw_size < 0 || sw_pattern <= 0) {
        throw KvError(
            "KV start_pos/sw_size must be non-negative and sw_pattern positive");
    }
    const int64_t max_tokens =
        static_cast<int64_t>(max_pages_per_seq) * page_block_size;
    if (start_pos > max_tokens) {
        throw KvError("KV start_pos exceeds addressable context");
    }
    kv.page_table = page_table;
    kv.cache_seqlens = cache_seqlens;
    kv.row_active = row_active;
    kv.num_layers = num_layers;
    kv.BS = BS;
    kv.max_pages_per_seq = max_pages_per_seq;
    kv.num_phys_pages = num_phys_pages;
    kv.page_block_size = page_block_size;
    kv.sw_size = sw_size;
    kv.sw_pattern = sw_pattern;
    kv.page_table_elems = checked_page_table_elems(
        num_layers,
        BS,
        max_pages_per_seq);
    kv.page_table_host.resize(kv.page_table_elems);
    kv.cache_seqlens_host.resize(BS);
    kv.row_active_host.assign(BS, 1);
    kv.sliding_evict_until.assign(
        checked_product(
            static_cast<size_t>(num_layers),
            static_cast<size_t>(BS),
            "sliding eviction table"),
        0);
    MK_CUDA_CHECK(cudaMemcpy(
        kv.page_table_host.data(), (void*)page_table,
        kv.page_table_elems * sizeof(int), cudaMemcpyDeviceToHost));
    if (cache_seqlens) {
        MK_CUDA_CHECK(cudaMemcpy(
            kv.cache_seqlens_host.data(), (void*)cache_seqlens,
            BS * sizeof(int), cudaMemcpyDeviceToHost));
    } else {
        std::fill(kv.cache_seqlens_host.begin(), kv.cache_seqlens_host.end(), start_pos);
    }
    for (int length : kv.cache_seqlens_host) {
        if (length < 0 || static_cast<int64_t>(length) > max_tokens) {
            throw KvError(
                "initial KV cache length is outside the addressable context");
        }
    }
    if (row_active) {
        MK_CUDA_CHECK(cudaMemcpy(
            kv.row_active_host.data(), (void*)row_active,
            BS * sizeof(int), cudaMemcpyDeviceToHost));
    }
    for (int v : kv.page_table_host) {
        if (v < -1 || v >= num_phys_pages) {
            throw KvError(
                "initial KV page table contains an invalid block id");
        }
    }
    {
        std::lock_guard<std::mutex> lock(g_prefix_mu);
        g_kv_block_pool.init_once(num_phys_pages);
        for (int v : kv.page_table_host) {
            if (v >= 0) kv_retain_block_nolock(v);
        }
        kv.peak_allocated = g_kv_block_pool.active_count();
    }
}

static int kv_allocated_blocks(const RuntimeKvState& kv) {
    (void)kv;
    std::lock_guard<std::mutex> lock(g_prefix_mu);
    return g_kv_block_pool.active_count();
}

static void kv_update_peak(RuntimeKvState& kv) {
    kv.peak_allocated = std::max(kv.peak_allocated, kv_allocated_blocks(kv));
}

static int& kv_pt(RuntimeKvState& kv, int layer, int batch, int block) {
    const size_t row =
        static_cast<size_t>(layer) * static_cast<size_t>(kv.BS) +
        static_cast<size_t>(batch);
    return kv.page_table_host[
        row * static_cast<size_t>(kv.max_pages_per_seq) +
        static_cast<size_t>(block)];
}

static int& kv_evict_until(RuntimeKvState& kv, int layer, int batch) {
    return kv.sliding_evict_until[
        static_cast<size_t>(layer) * static_cast<size_t>(kv.BS) +
        static_cast<size_t>(batch)];
}

static int kv_div_ceil_nonnegative(int value, int divisor) {
    return value == 0 ? 0 : 1 + (value - 1) / divisor;
}

static int64_t kv_max_tokens(const RuntimeKvState& kv) {
    return static_cast<int64_t>(kv.max_pages_per_seq) * kv.page_block_size;
}

static void kv_validate_length(
    const RuntimeKvState& kv,
    int length,
    const char* label) {
    if (length < 0 || static_cast<int64_t>(length) > kv_max_tokens(kv)) {
        throw KvError(
            std::string(label) + " is outside the addressable KV context");
    }
}

static void kv_release_all_blocks_nolock(RuntimeKvState& kv) {
    // Reverse logical order so later blocks reach the free-queue tail before
    // earlier, more reusable prefix blocks.
    for (int b = 0; b < kv.BS; ++b) {
        for (int li = kv.num_layers - 1; li >= 0; --li) {
            for (int blk = kv.max_pages_per_seq - 1; blk >= 0; --blk) {
                int& slot = kv_pt(kv, li, b, blk);
                if (slot >= 0) {
                    kv_release_block_nolock(slot);
                    slot = -1;
                }
            }
        }
    }
}

static bool kv_ensure_blocks_for_row(
    RuntimeKvState& kv,
    int layer,
    int batch,
    int blk_start,
    int blk_end) {
    if (blk_end <= blk_start) return false;
    if (blk_start < 0 || blk_end > kv.max_pages_per_seq) {
        std::ostringstream os;
        os << "prefill row " << batch << " layer " << layer
           << " needs logical blocks [" << blk_start << ", " << blk_end
           << ") but max_pages_per_seq=" << kv.max_pages_per_seq;
        throw KvError(os.str());
    }
    bool changed = false;
    for (int blk = blk_start; blk < blk_end; ++blk) {
        int& slot = kv_pt(kv, layer, batch, blk);
        if (slot != -1) continue;
        {
            std::lock_guard<std::mutex> lock(g_prefix_mu);
            KVCacheBlock& block = kv_acquire_block_nolock();
            slot = block.block_id;
        }
        changed = true;
    }
    if (changed) kv_update_peak(kv);
    return changed;
}

static bool kv_evict_below(RuntimeKvState& kv, int layer, int batch, int keep_from) {
    if (keep_from <= 0) return false;
    // Dense/full layers keep the entire history; never SWA-evict them.
    if (!kv_layer_is_sliding(kv, layer)) return false;
    keep_from = std::min(keep_from, kv.max_pages_per_seq);
    int& evict_until = kv_evict_until(kv, layer, batch);
    bool changed = false;
    for (int blk = evict_until; blk < keep_from; ++blk) {
        int& slot = kv_pt(kv, layer, batch, blk);
        if (slot != -1) {
            {
                std::lock_guard<std::mutex> lock(g_prefix_mu);
                kv_release_block_nolock(slot);
            }
            slot = -1;
            changed = true;
        }
    }
    evict_until = std::max(evict_until, keep_from);
    return changed;
}


static int kv_prefill_swa_keep_from(const RuntimeKvState& kv, int pos_start) {
    // Must track FA3 Hopper local-tile N for bf16 headdim 128. If FA tile sizes
    // change, revisit — under-retention reintroduces the paged-KV fault above.
    constexpr int kFlashAttnLocalBlockN = 128;
    if (kv.sw_size <= 0 || pos_start < kv.sw_size) {
        return 0;
    }
    // FA window_size left radius is (sw_size - 1); oldest attended absolute
    // index for the first token of this chunk is pos_start - (sw_size - 1).
    const int oldest_attended = pos_start - (kv.sw_size - 1);
    const int tile_start =
        (oldest_attended / kFlashAttnLocalBlockN) * kFlashAttnLocalBlockN;
    return tile_start / kv.page_block_size;
}

static void kv_flush_page_table(RuntimeKvState& kv, cudaStream_t stream) {
    MK_CUDA_CHECK(cudaMemcpyAsync(
        (void*)kv.page_table, kv.page_table_host.data(),
        kv.page_table_elems * sizeof(int), cudaMemcpyHostToDevice, stream));
}

static void kv_flush_cache_seqlens(RuntimeKvState& kv, cudaStream_t stream) {
    if (!kv.cache_seqlens) return;
    MK_CUDA_CHECK(cudaMemcpyAsync(
        (void*)kv.cache_seqlens, kv.cache_seqlens_host.data(),
        kv.BS * sizeof(int), cudaMemcpyHostToDevice, stream));
}

static void kv_step_decode_positions(
    RuntimeKvState& kv,
    const int* positions,
    const int* active,
    cudaStream_t stream) {
    for (int b = 0; b < kv.BS; ++b) {
        if (active != nullptr && active[b] == 0) continue;
        const int pos = positions ? positions[b] : kv.cache_seqlens_host[b];
        if (pos < 0 || static_cast<int64_t>(pos) >= kv_max_tokens(kv)) {
            throw KvError(
                "decode position is outside the addressable KV context");
        }
    }
    bool changed = false;
    for (int b = 0; b < kv.BS; ++b) {
        const bool row_active = active == nullptr || active[b] != 0;
        kv.row_active_host[b] = row_active ? 1 : 0;
        if (!row_active) continue;
        const int pos = positions ? positions[b] : kv.cache_seqlens_host[b];
        kv.cache_seqlens_host[b] = pos;
        const int write_blk = pos / kv.page_block_size;
        if (write_blk < 0 || write_blk >= kv.max_pages_per_seq) {
            std::ostringstream os;
            os << "decode row " << b << " position " << pos
               << " needs logical block " << write_blk
               << " but max_pages_per_seq=" << kv.max_pages_per_seq;
            throw KvError(os.str());
        }
        if (kv.sw_size > 0 && pos >= kv.sw_size) {
            int keep_from = (pos - kv.sw_size) / kv.page_block_size;
            for (int li = 0; li < kv.num_layers; ++li) {
                changed = kv_evict_below(kv, li, b, keep_from) || changed;
            }
        }
        for (int li = 0; li < kv.num_layers; ++li) {
            changed = kv_ensure_blocks_for_row(kv, li, b, write_blk, write_blk + 1) || changed;
        }
    }
    if (changed) kv_flush_page_table(kv, stream);
    kv_flush_cache_seqlens(kv, stream);
    if (kv.row_active) {
        MK_CUDA_CHECK(cudaMemcpyAsync(
            (void*)kv.row_active, kv.row_active_host.data(),
            kv.BS * sizeof(int), cudaMemcpyHostToDevice, stream));
    }
}

KvHandle::KvHandle(const KvHandleDesc& desc)
    : s_(std::make_unique<State>()) {
    if (desc.page_table == 0) {
        throw KvError("KvHandle requires a page_table pointer");
    }
    runtime_kv_init_from_ptrs(
        *s_,
        desc.page_table,
        desc.cache_seqlens,
        desc.row_active,
        desc.num_layers,
        desc.BS,
        desc.max_pages_per_seq,
        desc.num_phys_pages,
        desc.page_block_size,
        desc.start_pos,
        desc.sw_size,
        desc.sw_pattern);
}

KvHandle::~KvHandle() {
    std::lock_guard<std::mutex> lock(g_prefix_mu);
    kv_release_all_blocks_nolock(*s_);
}

int32_t KvHandle::batch_size() const { return s_->BS; }
int32_t KvHandle::num_layers() const { return s_->num_layers; }
int32_t KvHandle::max_pages_per_seq() const { return s_->max_pages_per_seq; }

void prefix_cache_clear() {
    std::lock_guard<std::mutex> lock(g_prefix_mu);
    g_prefix_cache.clear();
    for (KVCacheBlock& block : g_kv_block_pool.blocks) {
        block.has_hash = false;
        block.block_hash = BlockHash{};
    }
}

int32_t prefix_cache_size() {
    std::lock_guard<std::mutex> lock(g_prefix_mu);
    return (int32_t)g_prefix_cache.size();
}

PrefixCacheStats prefix_cache_stats() {
    PrefixCacheStats out{};
    std::lock_guard<std::mutex> lock(g_prefix_mu);
    int cached = 0;
    int free_cached = 0;
    for (const KVCacheBlock& block : g_kv_block_pool.blocks) {
        if (block.has_hash) {
            ++cached;
            if (block.in_free_queue) ++free_cached;
        }
    }
    out.entries = (int)g_prefix_cache.size();
    out.total_blocks = (int)g_kv_block_pool.blocks.size();
    out.active_blocks = g_kv_block_pool.active_count();
    out.free_blocks = g_kv_block_pool.free_count;
    out.cached_blocks = cached;
    out.free_cached_blocks = free_cached;
    out.evicted_blocks = g_kv_block_pool.evicted_blocks;
    return out;
}

bool KvHandle::prefix_attach(
    const char* key,
    int32_t key_len,
    int32_t row,
    int32_t logical_block,
    int32_t required_from_block,
    uint64_t stream_u64) {
    const int32_t rows[1] = {row};
    const int32_t logical_blocks[1] = {logical_block};
    const int32_t required_from_blocks[1] = {required_from_block};
    return prefix_attach_batch(
        key, key_len, rows, logical_blocks, required_from_blocks, 1,
        stream_u64);
}

bool KvHandle::prefix_attach_batch(
    const char* keys,
    int32_t key_len,
    const int32_t* rows,
    const int32_t* logical_blocks,
    const int32_t* required_from_blocks,
    int32_t count,
    uint64_t stream_u64) {
    RuntimeKvState* kv = s_.get();
    if (!keys || !rows || !logical_blocks || !required_from_blocks ||
        key_len <= 0 || count <= 0) {
        throw KvError(
            "prefix_attach_batch requires non-empty key and index arrays");
    }

    std::vector<std::vector<int>> cached_block_ids;
    cached_block_ids.reserve(count);
    {
        std::lock_guard<std::mutex> lock(g_prefix_mu);
        std::unordered_set<uint64_t> destinations;
        destinations.reserve(static_cast<size_t>(count));

        for (int32_t i = 0; i < count; ++i) {
            const int row = rows[i];
            const int logical_block = logical_blocks[i];
            if (row < 0 || row >= kv->BS || logical_block < 0 ||
                logical_block >= kv->max_pages_per_seq) {
                throw KvError("prefix_attach_batch row/block out of range");
            }
            const uint64_t destination =
                (static_cast<uint64_t>(static_cast<uint32_t>(row)) << 32) |
                static_cast<uint32_t>(logical_block);
            if (!destinations.insert(destination).second) {
                throw KvError(
                    "prefix_attach_batch contains a duplicate destination");
            }
            const int required_from_block = std::max(
                0, std::min(required_from_blocks[i], kv->max_pages_per_seq));
            const char* key = keys + static_cast<size_t>(i) * key_len;
            const BlockHash hash = parse_block_hash(key, key_len);
            auto it = g_prefix_cache.find(hash);
            if (it == g_prefix_cache.end() ||
                !prefix_entry_can_attach(
                    *kv, it->second, logical_block, required_from_block)) {
                return false;
            }
            cached_block_ids.push_back(it->second.block_ids);
        }

        for (int32_t i = 0; i < count; ++i) {
            for (int li = 0; li < kv->num_layers; ++li) {
                const int cached_block = cached_block_ids[i][li];
                if (cached_block < 0) {
                    continue;
                }
                int& slot = kv_pt(
                    *kv, li, rows[i], logical_blocks[i]);
                if (slot == cached_block) {
                    continue;
                }
                if (slot >= 0) {
                    kv_release_block_nolock(slot);
                }
                kv_retain_block_nolock(cached_block);
                slot = cached_block;
            }
        }
    }
    kv_flush_page_table(*kv, reinterpret_cast<cudaStream_t>(stream_u64));
    return true;
}

bool KvHandle::prefix_can_attach(
    const char* key,
    int32_t key_len,
    int32_t logical_block,
    int32_t required_from_block) const {
    RuntimeKvState* kv = s_.get();
    if (!key || key_len <= 0) {
        throw KvError("prefix_can_attach requires a key");
    }
    if (logical_block < 0 || logical_block >= kv->max_pages_per_seq) {
        throw KvError("prefix_can_attach block out of range");
    }
    required_from_block = std::max(0, std::min(required_from_block, kv->max_pages_per_seq));
    BlockHash h = parse_block_hash(key, key_len);
    std::lock_guard<std::mutex> lock(g_prefix_mu);
    auto it = g_prefix_cache.find(h);
    if (it == g_prefix_cache.end()) return false;
    return prefix_entry_can_attach(*kv, it->second, logical_block, required_from_block);
}

bool KvHandle::prefix_publish(
    const char* key,
    int32_t key_len,
    int32_t row,
    int32_t logical_block) {
    RuntimeKvState* kv = s_.get();
    if (!key || key_len <= 0) {
        throw KvError("prefix_publish requires a key");
    }
    if (row < 0 || row >= kv->BS || logical_block < 0 ||
        logical_block >= kv->max_pages_per_seq) {
        throw KvError("prefix_publish row/block out of range");
    }
    BlockHash h = parse_block_hash(key, key_len);
    std::lock_guard<std::mutex> lock(g_prefix_mu);
    if (g_prefix_cache.find(h) != g_prefix_cache.end()) return false;
    PrefixCacheEntry entry;
    entry.block_ids.resize(kv->num_layers);
    bool has_any_page = false;
    for (int li = 0; li < kv->num_layers; ++li) {
        int block_id = kv_pt(*kv, li, row, logical_block);
        entry.block_ids[li] = block_id;
        if (block_id >= 0) {
            KVCacheBlock& block = g_kv_block_pool.at(block_id);
            if (block.has_hash && !(block.block_hash == h)) {
                throw KvError("cannot publish a KV block under two different prefix hashes");
            }
            has_any_page = true;
        }
    }
    if (!has_any_page) return false;
    const auto [entry_it, inserted] =
        g_prefix_cache.emplace(h, std::move(entry));
    if (!inserted) return false;

    for (int block_id : entry_it->second.block_ids) {
        if (block_id < 0) continue;
        KVCacheBlock& block = g_kv_block_pool.at(block_id);
        block.block_hash = h;
        block.has_hash = true;
    }
    return true;
}

void KvHandle::prepare_prefill(
    const int32_t* prompt_lengths,
    uint64_t stream_u64) {
    RuntimeKvState* kv = s_.get();
    if (!prompt_lengths) {
        throw KvError("prepare_prefill requires prompt_lengths");
    }
    // Validate the entire batch before changing host metadata or allocating
    // blocks, so malformed input cannot leave a partially prepared handle.
    for (int b = 0; b < kv->BS; ++b) {
        kv_validate_length(*kv, prompt_lengths[b], "prefill length");
    }
    bool changed = false;
    for (int b = 0; b < kv->BS; ++b) {
        const int row_seq = prompt_lengths[b];
        kv->cache_seqlens_host[b] = row_seq;
        const int blk_hi = kv_div_ceil_nonnegative(
            row_seq,
            kv->page_block_size);
        if (blk_hi <= 0) continue;
        for (int li = 0; li < kv->num_layers; ++li) {
            const bool sliding = kv_layer_is_sliding(*kv, li);
            const int blk_lo = (
                sliding && kv->sw_size > 0 && row_seq > kv->sw_size)
                ? (row_seq - kv->sw_size) / kv->page_block_size
                : 0;
            changed = kv_ensure_blocks_for_row(*kv, li, b, blk_lo, blk_hi) || changed;
        }
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    if (changed) kv_flush_page_table(*kv, stream);
    kv_flush_cache_seqlens(*kv, stream);
}

void KvHandle::step_prefill_chunk(
    int32_t layer_idx,
    int32_t pos_start,
    int32_t pos_end,
    uint64_t stream_u64) {
    RuntimeKvState* kv = s_.get();
    if (layer_idx < 0 || layer_idx >= kv->num_layers) {
        throw KvError("step_prefill_chunk layer out of range");
    }
    kv_validate_length(*kv, pos_start, "prefill chunk start");
    kv_validate_length(*kv, pos_end, "prefill chunk end");
    if (pos_end < pos_start) {
        throw KvError("prefill chunk end precedes its start");
    }
    if (pos_end == pos_start) return;
    bool changed = false;
    if (kv->sw_size > 0 && pos_start >= kv->sw_size) {
        // FA3 tile-aligned cutoff — see kv_prefill_swa_keep_from. Tighter
        // decode-style (pos - sw_size) eviction is unsafe for FlashAttention
        // chunked prefill.
        const int keep_from = kv_prefill_swa_keep_from(*kv, pos_start);
        for (int b = 0; b < kv->BS; ++b) {
            changed = kv_evict_below(*kv, layer_idx, b, keep_from) || changed;
        }
    }
    const int write_blk_start = pos_start / kv->page_block_size;
    const int write_blk_end = kv_div_ceil_nonnegative(
        pos_end,
        kv->page_block_size);
    for (int b = 0; b < kv->BS; ++b) {
        changed = kv_ensure_blocks_for_row(
            *kv, layer_idx, b, write_blk_start, write_blk_end) || changed;
    }
    if (changed) {
        kv_flush_page_table(*kv, reinterpret_cast<cudaStream_t>(stream_u64));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Row-targeted (masked) prefill for continuous batching.
//
void KvHandle::prepare_prefill_masked(
    const int32_t* prompt_lengths,
    const int32_t* active,
    uint64_t stream_u64) {
    RuntimeKvState* kv = s_.get();
    if (!prompt_lengths || !active) {
        throw KvError(
            "prepare_prefill_masked requires prompt_lengths and active");
    }
    for (int b = 0; b < kv->BS; ++b) {
        if (active[b] != 0) {
            kv_validate_length(*kv, prompt_lengths[b], "masked prefill length");
        }
    }
    bool changed = false;
    for (int b = 0; b < kv->BS; ++b) {
        if (active[b] == 0) continue;  // leave live rows untouched
        const int row_seq = prompt_lengths[b];
        kv->cache_seqlens_host[b] = row_seq;
        const int blk_hi = kv_div_ceil_nonnegative(
            row_seq,
            kv->page_block_size);
        if (blk_hi <= 0) continue;
        for (int li = 0; li < kv->num_layers; ++li) {
            const bool sliding = kv_layer_is_sliding(*kv, li);
            const int blk_lo = (
                sliding && kv->sw_size > 0 && row_seq > kv->sw_size)
                ? (row_seq - kv->sw_size) / kv->page_block_size
                : 0;
            changed = kv_ensure_blocks_for_row(*kv, li, b, blk_lo, blk_hi) || changed;
        }
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    if (changed) kv_flush_page_table(*kv, stream);
    kv_flush_cache_seqlens(*kv, stream);
}

void KvHandle::step_prefill_chunk_masked(
    int32_t layer_idx,
    int32_t pos_start,
    int32_t pos_end,
    const int32_t* active,
    uint64_t stream_u64) {
    RuntimeKvState* kv = s_.get();
    if (!active) {
        throw KvError("step_prefill_chunk_masked requires active");
    }
    if (layer_idx < 0 || layer_idx >= kv->num_layers) {
        throw KvError("step_prefill_chunk_masked layer out of range");
    }
    kv_validate_length(*kv, pos_start, "masked prefill chunk start");
    kv_validate_length(*kv, pos_end, "masked prefill chunk end");
    if (pos_end < pos_start) {
        throw KvError("masked prefill chunk end precedes its start");
    }
    if (pos_end == pos_start) return;
    bool changed = false;
    if (kv->sw_size > 0 && pos_start >= kv->sw_size) {
        // FA3 tile-aligned cutoff — see kv_prefill_swa_keep_from.
        const int keep_from = kv_prefill_swa_keep_from(*kv, pos_start);
        for (int b = 0; b < kv->BS; ++b) {
            if (active[b] == 0) continue;
            changed = kv_evict_below(*kv, layer_idx, b, keep_from) || changed;
        }
    }
    const int write_blk_start = pos_start / kv->page_block_size;
    const int write_blk_end = kv_div_ceil_nonnegative(
        pos_end,
        kv->page_block_size);
    for (int b = 0; b < kv->BS; ++b) {
        if (active[b] == 0) continue;
        changed = kv_ensure_blocks_for_row(
            *kv, layer_idx, b, write_blk_start, write_blk_end) || changed;
    }
    if (changed) {
        kv_flush_page_table(*kv, reinterpret_cast<cudaStream_t>(stream_u64));
    }
}

// Free one row's KV blocks (used when evicting a finished sequence so its slot
// can be reused by a new request). Releases block refs and clears the row's
// page-table entries + evict state; cache_seqlens/row_active reset to 0.
void KvHandle::free_row(
    int32_t row,
    uint64_t stream_u64) {
    RuntimeKvState* kv = s_.get();
    if (row < 0 || row >= kv->BS) throw KvError("free_row row out of range");
    {
        std::lock_guard<std::mutex> lock(g_prefix_mu);
        for (int li = kv->num_layers - 1; li >= 0; --li) {
            for (int blk = kv->max_pages_per_seq - 1; blk >= 0; --blk) {
                int& slot = kv_pt(*kv, li, row, blk);
                if (slot >= 0) {
                    kv_release_block_nolock(slot);
                    slot = -1;
                }
            }
            kv_evict_until(*kv, li, row) = 0;
        }
    }
    kv->cache_seqlens_host[row] = 0;
    kv->row_active_host[row] = 0;
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    kv_flush_page_table(*kv, stream);
    kv_flush_cache_seqlens(*kv, stream);
    if (kv->row_active) {
        MK_CUDA_CHECK(cudaMemcpyAsync(
            (void*)kv->row_active, kv->row_active_host.data(),
            kv->BS * sizeof(int), cudaMemcpyHostToDevice, stream));
    }
}

KvHandle::KvHandle(std::unique_ptr<State> state) : s_(std::move(state)) {}

std::unique_ptr<KvHandle> KvHandle::rebind(
    const int32_t* src_rows,   // host int32[new_bs]; -1 means "empty dst row"
    int32_t new_bs,
    uint64_t new_page_table,
    uint64_t new_cache_seqlens,
    uint64_t new_row_active,
    uint64_t stream_u64) {
    RuntimeKvState* old_kv = s_.get();
    if (!src_rows) throw KvError("rebind requires src_rows");
    if (new_bs < 1) throw KvError("rebind new_bs must be >= 1");
    if (!new_page_table || !new_cache_seqlens || !new_row_active) {
        throw KvError("rebind requires all destination device tensors");
    }
    std::unordered_set<int> seen_rows;
    for (int d = 0; d < new_bs; ++d) {
        const int source_row = src_rows[d];
        if (source_row < -1 || source_row >= old_kv->BS) {
            throw KvError("rebind src row out of range");
        }
        if (source_row >= 0 && !seen_rows.insert(source_row).second) {
            throw KvError("rebind cannot duplicate a source row");
        }
    }
    // Own the new state through a KvHandle from here on: every path below can
    // throw after blocks have been retained, and ~KvHandle is what releases
    // them, so no hand-rolled rollback is needed.
    std::unique_ptr<KvHandle> handle(
        new KvHandle(std::make_unique<State>()));
    RuntimeKvState* nk = handle->s_.get();
    // Copy invariant geometry from the old handle.
    nk->page_table = new_page_table;
    nk->cache_seqlens = new_cache_seqlens;
    nk->row_active = new_row_active;
    nk->num_layers = old_kv->num_layers;
    nk->BS = new_bs;
    nk->max_pages_per_seq = old_kv->max_pages_per_seq;
    nk->num_phys_pages = old_kv->num_phys_pages;
    nk->page_block_size = old_kv->page_block_size;
    nk->sw_size = old_kv->sw_size;
    nk->sw_pattern = old_kv->sw_pattern;
    nk->peak_allocated = old_kv->peak_allocated;
    nk->page_table_elems = checked_page_table_elems(
        nk->num_layers,
        new_bs,
        nk->max_pages_per_seq);
    nk->page_table_host.assign(nk->page_table_elems, -1);
    nk->cache_seqlens_host.assign(new_bs, 0);
    nk->row_active_host.assign(new_bs, 0);
    nk->sliding_evict_until.assign(
        checked_product(
            static_cast<size_t>(nk->num_layers),
            static_cast<size_t>(new_bs),
            "rebound sliding eviction table"),
        0);
    {
        std::lock_guard<std::mutex> lock(g_prefix_mu);
        for (int d = 0; d < new_bs; ++d) {
            const int s = src_rows[d];
            if (s < 0) continue;  // empty dst row
            nk->cache_seqlens_host[d] = old_kv->cache_seqlens_host[s];
            nk->row_active_host[d] = old_kv->row_active_host[s];
            for (int li = 0; li < nk->num_layers; ++li) {
                kv_evict_until(*nk, li, d) =
                    kv_evict_until(*old_kv, li, s);
                for (int blk = 0; blk < nk->max_pages_per_seq; ++blk) {
                    const int slot = kv_pt(*old_kv, li, s, blk);
                    if (slot >= 0) kv_retain_block_nolock(slot);
                    kv_pt(*nk, li, d, blk) = slot;
                }
            }
        }
    }
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_u64);
    kv_flush_page_table(*nk, stream);
    kv_flush_cache_seqlens(*nk, stream);
    if (nk->row_active) {
        MK_CUDA_CHECK(cudaMemcpyAsync(
            (void*)nk->row_active, nk->row_active_host.data(),
            new_bs * sizeof(int), cudaMemcpyHostToDevice, stream));
    }
    MK_CUDA_CHECK(cudaStreamSynchronize(stream));
    return handle;
}

void KvHandle::set_cache_seqlens(
    const int32_t* lengths,
    uint64_t stream_u64) {
    RuntimeKvState* kv = s_.get();
    if (!lengths) throw KvError("set_cache_seqlens requires lengths");
    for (int b = 0; b < kv->BS; ++b) {
        kv_validate_length(*kv, lengths[b], "cache sequence length");
    }
    for (int b = 0; b < kv->BS; ++b) kv->cache_seqlens_host[b] = lengths[b];
    kv_flush_cache_seqlens(*kv, reinterpret_cast<cudaStream_t>(stream_u64));
}

void KvHandle::step_decode_positions(
    const int32_t* positions,
    const int32_t* active,
    uint64_t stream_u64) {
    if (!positions) throw KvError("step_decode_positions requires positions");
    kv_step_decode_positions(
        *s_,
        positions,
        active,
        reinterpret_cast<cudaStream_t>(stream_u64));
}

KvStats KvHandle::stats() const {
    KvStats out{};
    std::lock_guard<std::mutex> lock(g_prefix_mu);
    out.total = s_->num_phys_pages;
    out.allocated = g_kv_block_pool.active_count();
    out.peak_allocated = s_->peak_allocated;
    out.free = g_kv_block_pool.free_count;
    return out;
}

// Lightweight free-block probe for the NMC decode service's per-step KV-OOM
// pre-flight (see decode/runtime.cu). The free count is a property of the shared
// process-wide pool, not of this handle; it is a method because callers only
// ever reach the pool through a handle and KvStats::free reports the same
// number.
int32_t KvHandle::free_blocks() const {
    std::lock_guard<std::mutex> lock(g_prefix_mu);
    return g_kv_block_pool.free_count;
}

void KvHandle::host_snapshot(KvHostSnapshot& io) const {
    // Host-only: no CUDA, no device touch. Safe after a wedged kernel.
    RuntimeKvState* kv = s_.get();
    io.num_layers = kv->num_layers;
    io.BS = kv->BS;
    io.max_pages_per_seq = kv->max_pages_per_seq;
    io.num_phys_pages = kv->num_phys_pages;
    io.page_block_size = kv->page_block_size;
    io.sw_size = kv->sw_size;
    io.sw_pattern = kv->sw_pattern;
    if (kv->page_table_elems >
        static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        throw KvError("page_table_elems exceeds int32");
    }
    io.page_table_elems = static_cast<int32_t>(kv->page_table_elems);
    {
        std::lock_guard<std::mutex> lock(g_prefix_mu);
        io.stats.total = kv->num_phys_pages;
        io.stats.allocated = g_kv_block_pool.active_count();
        io.stats.peak_allocated = kv->peak_allocated;
        io.stats.free = g_kv_block_pool.free_count;
    }
    if (io.page_table_out != nullptr) {
        if (io.page_table_cap < io.page_table_elems) {
            throw KvError("host_snapshot page_table_out capacity too small");
        }
        if (kv->page_table_host.size() != kv->page_table_elems) {
            throw KvError(
                "page_table_host size disagrees with page_table_elems");
        }
        std::memcpy(
            io.page_table_out,
            kv->page_table_host.data(),
            kv->page_table_elems * sizeof(int32_t));
    }
    if (io.cache_seqlens_out != nullptr) {
        if ((int)kv->cache_seqlens_host.size() != kv->BS) {
            throw KvError("cache_seqlens_host size disagrees with BS");
        }
        std::memcpy(
            io.cache_seqlens_out,
            kv->cache_seqlens_host.data(),
            static_cast<size_t>(kv->BS) * sizeof(int32_t));
    }
    if (io.row_active_out != nullptr) {
        if ((int)kv->row_active_host.size() != kv->BS) {
            throw KvError("row_active_host size disagrees with BS");
        }
        std::memcpy(
            io.row_active_out,
            kv->row_active_host.data(),
            static_cast<size_t>(kv->BS) * sizeof(int32_t));
    }
}

}  // namespace kv_pool
