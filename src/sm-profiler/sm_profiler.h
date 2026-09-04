/*
 * SM-level Profiler Library
 *
 * Context-based API optimized for minimal overhead:
 *   - Buffer parsed once at init, not per-call
 *   - Handle-based range events (no active-event lookup table)
 *   - Slim 24-byte event struct (redundant fields derived at export)
 *   - __ldg for read-only header access
 *
 * Typical b2b overhead: ~50-100ns (down from 300-400ns).
 *
 * Adapted from https://github.com/leepoly/sm-profiler; see README.md in this
 * directory for the changes.
 */

 #pragma once

 #include <stdint.h>
 #include <stddef.h>
 #include <cuda_runtime.h>

 #ifdef __cplusplus
 extern "C" {
 #endif

 /* Opaque handle to profiler buffer */
 typedef struct sm_profiler_buffer* sm_profiler_buffer_t;


 /*
  * Create a profiler buffer
  *
  * @param num_blocks: Number of CUDA blocks expected
  * @param num_groups: Number of groups (warps) per block
  * @param max_events_per_group: Maximum events per group (warp)
  * @param enabled: Whether profiling is enabled (0 = disabled, minimal buffer)
  * @return: Handle to profiler buffer, or NULL on error
  */
 sm_profiler_buffer_t sm_profiler_create_buffer(
     uint32_t num_blocks,
     uint32_t num_groups,
     uint32_t max_events_per_group,
     int enabled
 );

 void sm_profiler_destroy_buffer(sm_profiler_buffer_t buffer);
 uint64_t* sm_profiler_get_device_ptr(sm_profiler_buffer_t buffer);

 /*
  * Initialize buffer on host side (call before kernel launch).
  * Writes header, clears counters and event data.
  */
 void sm_profiler_init_buffer(sm_profiler_buffer_t buffer);

 int sm_profiler_register_event(
     sm_profiler_buffer_t buffer,
     uint32_t event_no,
     const char* name
 );

 int sm_profiler_export_to_file(
     sm_profiler_buffer_t buffer,
     const char* filename
 );

/*
 * Export ranges as Chrome/Perfetto complete ("X") events. Event names are
 * emitted exactly as registered and all groups for a block share one block
 * track. The regular exporter remains available for detailed per-group traces.
 */
int sm_profiler_export_to_file_compact(
    sm_profiler_buffer_t buffer,
    const char* filename
);

 void sm_profiler_get_info(
     sm_profiler_buffer_t buffer,
     uint32_t* out_num_blocks,
     uint32_t* out_num_groups,
     uint32_t* out_max_events
 );

  /*
  * Compute the required device buffer size in bytes for the given parameters.
  * Use this to pre-allocate memory (e.g. via PyTorch) before calling
  * sm_profiler_create_buffer_external().
  */
  size_t sm_profiler_calc_buffer_size(
    uint32_t num_blocks,
    uint32_t num_groups,
    uint32_t max_events_per_group,
    int enabled
);

/*
 * Create a profiler buffer backed by externally-allocated device memory.
 * The caller owns the device memory and is responsible for freeing it.
 * Use sm_profiler_destroy_buffer() to free the handle (device memory
 * is NOT freed).
 *
 * @param external_device_ptr: Caller-owned device pointer (>= calc_buffer_size bytes)
 * @param buffer_size: Size of external_device_ptr allocation in bytes
 */
sm_profiler_buffer_t sm_profiler_create_buffer_external(
    uint64_t* external_device_ptr,
    size_t buffer_size,
    uint32_t num_blocks,
    uint32_t num_groups,
    uint32_t max_events_per_group,
    int enabled
);

 #ifdef __cplusplus
 }
 #endif

 /* ============================================================================
  * Device-side API
  *
  * Usage:
  *   SmProfilerCtx pctx = {};
  *   if (lane_id == 0) pctx = sm_profiler_init_ctx(prof_buf);
  *
  *   uint32_t h = sm_profiler_start(pctx, EVT_LOAD);
  *   // ... work ...
  *   sm_profiler_end(pctx, h);
  *
  * Only one thread per warp (typically lane 0) should init and use the
  * context. Zero-initialized contexts (enabled=0) safely no-op.
  * ============================================================================ */

 #ifdef __CUDACC__

 #define SM_PROFILER_EVENT_FLAG_INSTANT (1u << 31)
 #define SM_PROFILER_EVENT_NO_MASK      0x7FFFFFFFu
 #define SM_PROFILER_INVALID_HANDLE     (~0u)

 /*
  * 24-byte device event (was 48).
  * block_id, group_idx, event_id are derived from buffer position at export.
  */
 struct alignas(8) SmProfilerDeviceEvent {
     uint64_t st_timestamp_ns;
     uint64_t en_timestamp_ns;
     uint32_t event_no;        /* bits 0-30 = type id; bit 31 = instant flag */
     uint32_t sm_id;
 };

 /* ---- Helpers ---- */

 __device__ __forceinline__ uint32_t sm_profiler_get_block_idx() {
     return (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
 }

 __device__ __forceinline__ uint32_t sm_profiler_get_thread_idx_in_block() {
     return (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
 }

 __device__ __forceinline__ uint32_t sm_profiler_get_lane_id() {
     uint32_t lane_id;
     asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane_id));
     return lane_id;
 }

 __device__ __forceinline__ uint32_t sm_profiler_get_warp_id() {
     return sm_profiler_get_thread_idx_in_block() / 32;
 }

 __device__ __forceinline__ uint32_t sm_profiler_get_smid() {
     uint32_t smid;
     asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
     return smid;
 }

 __device__ __forceinline__ uint64_t sm_profiler_get_timestamp() {
     uint64_t ret;
     asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ret));
     return ret;
 }

 /* ---- Per-warp context: parsed once, reused across all profiler calls ---- */

 struct SmProfilerCtx {
     SmProfilerDeviceEvent* events;   /* this group's event array */
     uint32_t* counter;               /* this group's event counter */
     uint32_t max_events;
     uint8_t sm_id;
     bool enabled{false};
 };

 __device__ __forceinline__ uint64_t sm_profiler_load_u64_evict_last(uint64_t* ptr) {
    uint64_t ret;
    asm volatile("ld.global.L1::evict_last.u64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
 }
 __device__ __forceinline__ uint32_t sm_profiler_load_u32_evict_last(uint32_t* ptr) {
    uint32_t ret;
    asm volatile("ld.global.L1::evict_last.u32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
 }
 /*
  * Buffer layout (uint64-aligned):
  *   [header:  2 x uint64]
  *   [counters: num_blocks * num_groups x uint32, ceil-padded to uint64]
  *   [events:  num_blocks * num_groups * max_events x SmProfilerDeviceEvent]
  */

 __device__ __forceinline__
SmProfilerCtx sm_profiler_init_ctx_for_block(uint64_t* buffer, uint32_t block_idx) {
     SmProfilerCtx ctx{};
     if (!buffer) return ctx;

     uint64_t header0 = sm_profiler_load_u64_evict_last(buffer);
     uint64_t header1 = sm_profiler_load_u64_evict_last(buffer + 1);
     uint32_t num_blocks = (uint32_t)(header0 >> 32);
     uint32_t num_groups = (uint32_t)(header0 & 0xFFFFFFFF);
     ctx.max_events = (uint32_t)(header1 >> 32);
     ctx.enabled    = (uint32_t)(header1 & 0xFFFFFFFF);

     if (!ctx.enabled) return ctx;

     ctx.sm_id = sm_profiler_get_smid();

     uint32_t group_idx = sm_profiler_get_warp_id();
    if (block_idx >= num_blocks || group_idx >= num_groups) {
        ctx.enabled = false;
        return ctx;
    }
     uint32_t linear    = block_idx * num_groups + group_idx;

     ctx.counter = (uint32_t*)(buffer + 2) + linear;

     uint32_t counters_u64 = ((uint32_t)(num_blocks * num_groups) * (uint32_t)sizeof(uint32_t) + 7u) / 8u;
     char* event_base = (char*)(buffer + 2 + counters_u64);
     ctx.events = (SmProfilerDeviceEvent*)(
         event_base + (size_t)linear * ctx.max_events * sizeof(SmProfilerDeviceEvent));

     return ctx;
 }

__device__ __forceinline__
SmProfilerCtx sm_profiler_init_ctx(uint64_t* buffer) {
    return sm_profiler_init_ctx_for_block(buffer, sm_profiler_get_block_idx());
}

__device__ __forceinline__
SmProfilerCtx sm_profiler_init_ctx_by_smid(uint64_t* buffer) {
    return sm_profiler_init_ctx_for_block(buffer, sm_profiler_get_smid());
}

 /*
  * Start a range event.
  * Returns a handle to pass to sm_profiler_end().
  * Returns SM_PROFILER_INVALID_HANDLE when disabled or full.
  */
 __device__ __forceinline__
 uint32_t sm_profiler_start(SmProfilerCtx& ctx, uint32_t event_no) {
     if (!ctx.enabled) return SM_PROFILER_INVALID_HANDLE;

     uint32_t id = sm_profiler_load_u32_evict_last(ctx.counter);
     if (id >= ctx.max_events) return SM_PROFILER_INVALID_HANDLE;
     *ctx.counter = id + 1;
    

     SmProfilerDeviceEvent* ev = ctx.events + id;
     ev->event_no = event_no;
     ev->sm_id    = ctx.sm_id;
     asm volatile("" ::: "memory");
     ev->st_timestamp_ns = sm_profiler_get_timestamp();
     return id;
 }

 /*
  * End a range event.  Pass the handle returned by sm_profiler_start().
  *
  * Timestamp is read unconditionally FIRST so that after __syncthreads()
  * all warps sample globaltimer on the very first eligible cycle,
  * minimising post-barrier scheduling skew.
  */
 __device__ __forceinline__
 void sm_profiler_end(SmProfilerCtx& ctx, uint32_t handle) {
     uint64_t ts = sm_profiler_get_timestamp();
     if (handle == SM_PROFILER_INVALID_HANDLE) return;
     ctx.events[handle].en_timestamp_ns = ts;
 }

 /*
  * Atomically end one range event and start the next, sharing a single
  * timestamp for both.  Eliminates the visible gap between back-to-back
  * events and halves the profiler overhead at transition points.
  *
  * Returns a handle for the new event (pass to sm_profiler_end / swap).
  */
 __device__ __forceinline__
 uint32_t sm_profiler_swap(SmProfilerCtx& ctx,
                           uint32_t       end_handle,
                           uint32_t       new_event_no) {
     uint64_t ts = sm_profiler_get_timestamp();

     if (end_handle != SM_PROFILER_INVALID_HANDLE)
         ctx.events[end_handle].en_timestamp_ns = ts;

     if (!ctx.enabled) return SM_PROFILER_INVALID_HANDLE;

     uint32_t id = sm_profiler_load_u32_evict_last(ctx.counter);
     if (id >= ctx.max_events) return SM_PROFILER_INVALID_HANDLE;
     *ctx.counter = id + 1;
     
     SmProfilerDeviceEvent* ev = ctx.events + id;
     ev->event_no = new_event_no;
     ev->sm_id    = ctx.sm_id;
     ev->st_timestamp_ns = ts;
     return id;
 }

 /*
  * Record an instant (point) event.
  */
 __device__ __forceinline__
 void sm_profiler_instant(SmProfilerCtx& ctx, uint32_t event_no) {
     if (!ctx.enabled) return;

     uint32_t id = sm_profiler_load_u32_evict_last(ctx.counter);
     if (id >= ctx.max_events) return;
     *ctx.counter = id + 1;
     

     SmProfilerDeviceEvent* ev = ctx.events + id;
     ev->event_no = event_no | SM_PROFILER_EVENT_FLAG_INSTANT;
     ev->sm_id    = ctx.sm_id;
     ev->st_timestamp_ns = sm_profiler_get_timestamp();
 }

 #endif /* __CUDACC__ */
