// NMC release host runtime implementing decode/abi.h. Together with
// kv/cache.cpp, it builds the self-contained libmk_release.so.
#include "decode/launch.cuh"
#include "kv/cache.h"
#include "jit.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <ratio>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "nlohmann/json.hpp"

namespace {

using JitNmcFn = float (*)(const mk::NmcLaunchDesc*);

// Only DecodeService::run still reports through this; everything else throws
// MkError. See the run() comment in decode/abi.h for why that one is different.
thread_local std::string g_nmc_last_error;

// Destructors and C-ABI failure paths must not throw. Still route cudaFree
// through the checked wrapper, then report a cleanup failure without masking
// the primary error or terminating during stack unwinding.
static void cuda_free_checked_noexcept(void* ptr, const char* label) noexcept {
    if (!ptr) return;
    try {
        MK_NMC_CUDA_CHECK(cudaFree(ptr));
    } catch (const std::exception& error) {
        std::fprintf(
            stderr,
            "[mk-release] failed to free %s: %s\n",
            label,
            error.what());
    }
}

static void cuda_free_host_checked_noexcept(void* ptr, const char* label) noexcept {
    if (!ptr) return;
    try {
        MK_NMC_CUDA_CHECK(cudaFreeHost(ptr));
    } catch (const std::exception& error) {
        std::fprintf(
            stderr,
            "[mk-release] failed to free pinned %s: %s\n",
            label,
            error.what());
    }
}

using json = nlohmann::json;

enum class FinishReason : int32_t {
    kNone = 0,
    kEos = 1,
    kLength = 2,
};

// Keep watchdog expiry distinguishable at the C ABI. The Python server must
// terminate the process for this return code: reporting the timeout does not
// cancel the in-flight kernel, so neither this CUDA context nor the server can
// be recovered safely in-process.
class DecodeStepWatchdogExpired final : public std::runtime_error {
public:
    explicit DecodeStepWatchdogExpired(const char* message)
        : std::runtime_error(message) {}
};
constexpr int kDecodeServiceWatchdogExpired = -2;

// The watchdog reports a stalled step to the host; it does not cancel GPU work.
// A genuinely wedged CUDA context can still require process restart.
// TODO(release-config): make these values configurable after production latency
// and recovery policy are established.
constexpr auto kDecodeStepWatchdogTimeout = std::chrono::seconds{1};
constexpr auto kDecodeStepWatchdogPollInterval = std::chrono::microseconds{500};
// Side-stream snapshot must finish quickly or we abandon it and still park on
// the hung decode stream for cuda-gdb. The volatile-load kernel needs a free
// SM when the megakernel occupies the rest of the chip; if it cannot schedule,
// this budget expires and we fall back to copy-engine D2H.
constexpr auto kWatchdogDiagTimeout = std::chrono::seconds{2};
// Budget for the one-off warmup of the diagnostic paths in
// preallocate_watchdog_diag_buffers(). The GPU is supposed to be idle there, so
// this is only a guard against a caller that switched geometry without pausing:
// exceeding it degrades diagnostics instead of hanging the switch forever.
constexpr auto kWatchdogWarmupTimeout = std::chrono::seconds{5};

static void synchronize_stream_with_timeout(
    cudaStream_t stream,
    std::chrono::steady_clock::duration timeout) {
    cudaEvent_t done = nullptr;
    MK_NMC_CUDA_CHECK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
    MK_NMC_CUDA_CHECK(cudaEventRecord(done, stream));
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    for (;;) {
        const cudaError_t status = cudaEventQuery(done);
        if (status == cudaSuccess) {
            MK_NMC_CUDA_CHECK(cudaEventDestroy(done));
            return;
        }
        if (status != cudaErrorNotReady) {
            MK_NMC_CUDA_CHECK(cudaEventDestroy(done));
            MK_NMC_CUDA_CHECK(status);
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            MK_NMC_CUDA_CHECK(cudaEventDestroy(done));
            throw std::runtime_error("CUDA stream wait timed out");
        }
        std::this_thread::sleep_for(kDecodeStepWatchdogPollInterval);
    }
}

static void synchronize_decode_step_with_watchdog(cudaStream_t stream) {
    try {
        synchronize_stream_with_timeout(stream, kDecodeStepWatchdogTimeout);
    } catch (const std::runtime_error&) {
        throw DecodeStepWatchdogExpired(
            "NMC decode-step watchdog expired; GPU kernel did not complete");
    }
}

// Debug mode, set once at startup by the host through
// mk::set_debug() (the --debug CLI flag). It only widens diagnostics;
// it never changes decode behaviour or the compiled kernel. Read on the watchdog
// path, written once before the service starts, so relaxed ordering suffices.
static std::atomic<bool> g_nmc_debug_mode{false};

static bool nmc_debug_mode() {
    return g_nmc_debug_mode.load(std::memory_order_relaxed);
}

// Arm a one-shot, never-disarmed process self-destruct.
//
// A confirmed wedge makes CUDA teardown unsafe. Allow a bounded traceback/log
// flush, then exit without touching CUDA. The detached thread is intentionally
// leaked because there is no safe join point; arming is idempotent.
static void nmc_arm_process_self_destruct(
    std::chrono::steady_clock::duration grace, const char* reason) {
    static std::atomic<bool> armed{false};
    if (armed.exchange(true)) return;
    std::thread([grace, reason]() {
        std::this_thread::sleep_for(grace);
        std::fprintf(stderr,
            "\n[nmc] FATAL: exiting hard -- %s. The CUDA context is wedged, so "
            "teardown cannot complete.\n", reason);
        std::fflush(stderr);
        std::_Exit(99);
    }).detach();
}

// Grace period for the unwind to surface the watchdog error before the
// self-destruct fires. Long enough for a Python traceback and a log flush, short
// enough that a wedged production server restarts promptly.
constexpr auto kWatchdogExitGrace = std::chrono::seconds{5};

// After diagnostics, production arms hard exit and propagates the watchdog
// status; debug parks forever on the live wedge for cuda-gdb. CUDA cannot cancel
// the resident kernel, so neither mode permits in-process recovery.
static void finish_watchdog_wedge(cudaStream_t stream) {
    if (!nmc_debug_mode()) {
        nmc_arm_process_self_destruct(
            kWatchdogExitGrace, "NMC decode-step watchdog expired");
        return;
    }
    std::printf(
        "[nmc] debug mode: parking on the hung stream so cuda-gdb can attach "
        "(this will not return)\n");
    std::fflush(stdout);
    cudaStreamSynchronize(stream);
}

static json json_int_array(const std::vector<int>& v) {
    json out = json::array();
    for (int x : v) out.push_back(x);
    return out;
}

static int json_int(const json& j, const char* key) {
    if (!j.contains(key) || !j.at(key).is_number_integer()) {
        throw std::runtime_error(std::string("NMC JIT config missing integer field ") + key);
    }
    return j.at(key).get<int>();
}

static bool json_bool(const json& j, const char* key, bool fallback) {
    if (!j.contains(key)) return fallback;
    if (!j.at(key).is_boolean()) {
        throw std::runtime_error(std::string("NMC JIT config field must be bool: ") + key);
    }
    return j.at(key).get<bool>();
}

static std::string cfg_hash_suffix(const std::string& config_json) {
    const uint64_t h = static_cast<uint64_t>(std::hash<std::string>{}(config_json));
    std::ostringstream os;
    os << std::hex << h;
    return os.str();
}

static bool is_supported_nmc_batch_size(int bs) {
    return bs == 1 || bs == 2 || bs == 4 || bs == 8;
}

static void emit_op_cfg(std::ostringstream& os, const char* name, const json& op) {
    const int bm = json_int(op, "bm");
    const int bn = json_int(op, "bn");
    const int bk = json_int(op, "bk");
    const int num_cwg = json_int(op, "num_cwg");
    const int stages = json_int(op, "stages");
    const int prefetch_stages = json_int(op, "prefetch_stages");
    const int split_k = json_int(op, "split_k");
    const int m_rows = json_int(op, "m_rows");
    const bool direct_store = json_bool(op, "direct_store", false);
    if (bm <= 0 || bn <= 0 || bk <= 0 || num_cwg <= 0 || stages <= 0 || split_k <= 0) {
        throw std::runtime_error(std::string("NMC JIT op config has non-positive field: ") + name);
    }
    if (prefetch_stages < 0 || prefetch_stages > stages) {
        throw std::runtime_error(
            std::string("NMC JIT op config requires 0 <= PREFETCH_STAGES <= STAGES: ") + name);
    }
    os << "  struct " << name << " {\n";
    os << "    static constexpr int BM=" << bm << ", BN=" << bn << ", BK=" << bk
       << ", NUM_CWG=" << num_cwg << ", STAGES=" << stages
       << ", PREFETCH_STAGES=" << prefetch_stages << ", SPLIT_K=" << split_k << ";\n";
    os << "    static constexpr int M_ROWS=" << m_rows << ";\n";
    os << "    static constexpr bool DIRECT_STORE=" << (direct_store ? "true" : "false") << ";\n";
    os << "  };\n";
}

static std::string make_nmc_jit_source(int bs, bool enable_profiler, const char* config_json_cstr) {
    if (!is_supported_nmc_batch_size(bs)) {
        throw std::runtime_error("unsupported NMC JIT BS");
    }
    if (!config_json_cstr || config_json_cstr[0] == '\0') {
        throw std::runtime_error("NMC JIT config JSON is empty");
    }
    const std::string config_json(config_json_cstr);
    json j = json::parse(config_json);
    if (json_int(j, "batch_size") != bs) {
        throw std::runtime_error("NMC JIT config batch_size does not match bs argument");
    }
    std::ostringstream cfg_os;
    cfg_os << "NmcJitCfg_bs" << bs << "_h" << cfg_hash_suffix(config_json)
           << (enable_profiler ? "_prof" : "_noprof");
    const std::string cfg = cfg_os.str();
    std::ostringstream os;
    os << "#include \"decode/launch.cuh\"\n";
    os << "namespace mk {\n";
    os << "struct " << cfg << " {\n";
    os << "  static constexpr int NUM_WARPGROUPS=" << json_int(j, "num_warpgroups") << ";\n";
    os << "  static constexpr int NUM_WARPS=" << json_int(j, "num_warps") << ";\n";
    os << "  static constexpr int NUM_THREADS=" << json_int(j, "num_threads") << ";\n";
    os << "  static constexpr int INST_RING=" << json_int(j, "inst_ring") << ";\n";
    emit_op_cfg(os, "UpGate", j.at("upgate"));
    emit_op_cfg(os, "Down", j.at("down"));
    emit_op_cfg(os, "QKV", j.at("qkv"));
    emit_op_cfg(os, "OProj", j.at("oproj"));
    emit_op_cfg(os, "LMHead", j.at("lmhead"));
    emit_op_cfg(os, "Router", j.at("router"));
    emit_op_cfg(os, "MoeUpGate", j.at("moe_upgate"));
    emit_op_cfg(os, "MoeDown", j.at("moe_down"));
    os << "  static constexpr bool ENABLE_PROFILER = " << (enable_profiler ? "true" : "false") << ";\n";
    os << "  static constexpr bool PROFILE_BY_SM = false;\n";
    // A/B arm: MoE down epilogue scatter-reduces into x_ffn and skips
    // MOE_COMBINE.
    const bool moe_combine_atomic_tma = json_bool(j, "moe_combine_atomic_tma", false);
    os << "  static constexpr bool MOE_COMBINE_ATOMIC_TMA = "
       << (moe_combine_atomic_tma ? "true" : "false") << ";\n";
    os << "};\n";
    if (enable_profiler) {
        os << "static_assert(" << cfg << "::ENABLE_PROFILER, \"NMC profile JIT must enable profiler\");\n";
    }
    os << "} // namespace mk\n";
    os << "extern \"C\" float mk_nmc_decode_launch_jit("
          "const mk::NmcLaunchDesc* d) {\n";
    os << "  mk::nmc_decode_launch_impl<mk::" << cfg << ">(d);\n";
    os << "  return 0.0f;\n";
    os << "}\n";
    return os.str();
}

static std::vector<std::string> make_nmc_jit_include_paths(const char* repo_root) {
    std::string root = (repo_root && repo_root[0]) ? repo_root : ".";
    return {
        root + "/ext/ThunderKittens/include",
        root + "/ext/ThunderKittens/prototype",
        root + "/src",
        root + "/src/sm-profiler",
    };
}

static std::vector<std::string> make_nmc_jit_hash_files(const char* repo_root) {
    std::string root = (repo_root && repo_root[0]) ? repo_root : ".";
    return {
        root + "/src/decode/megakernel.cuh",
        root + "/src/decode/launch.cuh",
        root + "/src/decode/gemm-n8-wgmma.cuh",
        root + "/src/sm-profiler/sm_profiler.h",
    };
}

// Compiles (or fetches from the little_jit cache) the megakernel .so and
// resolves its entry point. The dlopen'd module lives in the returned handle.
static little_jit::jit_function build_nmc_jit(
    int bs, bool enable_profiler, const char* repo_root, const char* config_json,
    std::string& so_path_out) {
    std::string code = make_nmc_jit_source(bs, enable_profiler, config_json);
    std::vector<std::string> include_paths = make_nmc_jit_include_paths(repo_root);
    std::vector<std::string> hash_files = make_nmc_jit_hash_files(repo_root);
    std::vector<std::string> defines = {"KITTENS_HOPPER"};
    std::vector<std::string> options = {
        "--generate-code=arch=compute_90a,code=[compute_90a,sm_90a]",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-O3",
        "-lineinfo",
        "--use_fast_math",
        "--ptxas-options=-v,-warn-spills",
    };
    std::vector<std::string> flags = {
        "-L/usr/local/cuda/targets/x86_64-linux/lib",
        "-lcuda",
    };
    so_path_out = little_jit::jit_compile(
        code, {}, include_paths, defines, options, flags, hash_files);
    return little_jit::load_function(so_path_out, "mk_nmc_decode_launch_jit");
}

}  // namespace

namespace mk {

// PIMPL'd because little_jit::jit_function is not something a binding TU can
// see; decode/abi.h has to stay compilable without CUDA or little_jit.
struct JitKernel::Impl {
    little_jit::jit_function fn;
    JitNmcFn launch = nullptr;
    std::string so_path;
};

JitKernel::JitKernel(std::unique_ptr<Impl> impl) : p_(std::move(impl)) {}

// Out of line, and after Impl is complete, so ~unique_ptr<Impl> can instantiate.
// That is the one obligation PIMPL imposes.
JitKernel::~JitKernel() = default;

std::shared_ptr<JitKernel> JitKernel::compile(int32_t bs, bool enable_profiler,
                                              const std::string& repo_root,
                                              const std::string& config_json) {
    auto impl = std::make_unique<Impl>();
    impl->fn = build_nmc_jit(bs, enable_profiler, repo_root.c_str(),
                             config_json.c_str(), impl->so_path);
    impl->launch = reinterpret_cast<JitNmcFn>(impl->fn.function);
    if (!impl->launch) throw MkError("NMC JIT launch symbol is null");
    // Not make_shared: the constructor is private.
    return std::shared_ptr<JitKernel>(new JitKernel(std::move(impl)));
}

float JitKernel::decode_launch(const NmcLaunchDesc& d) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(d.stream_u64);
    cudaEvent_t start{}, end{};
    const bool timing = d.timing != 0;
    if (timing) {
        MK_NMC_CUDA_CHECK(cudaEventCreate(&start));
        MK_NMC_CUDA_CHECK(cudaEventCreate(&end));
        MK_NMC_CUDA_CHECK(cudaEventRecord(start, stream));
    }
    p_->launch(&d);
    if (!timing) return 0.0f;
    MK_NMC_CUDA_CHECK(cudaEventRecord(end, stream));
    MK_NMC_CUDA_CHECK(cudaEventSynchronize(end));
    float ms = 0.0f;
    MK_NMC_CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
    MK_NMC_CUDA_CHECK(cudaEventDestroy(start));
    MK_NMC_CUDA_CHECK(cudaEventDestroy(end));
    return ms;
}

const std::string& JitKernel::artifact_path() const { return p_->so_path; }

}  // namespace mk

// ─────────────────────────────────────────────────────────────────────────────
// One-shot NMC host runtime (C++ decode loop).
//
// Python owns setup and allocations; this runs KV updates, resets, seeding,
// launch, and sampling without per-step interpreter overhead. It supports
// single-chunk "all" schedules with context-bucket switching, greedy/Gumbel
// sampling, and descriptor-provided reset regions. Top-p, graph capture, and
// in-loop SM profiling are intentionally unsupported.
//
namespace mk {

namespace {

constexpr int kNmcSeedThreads = 256;
constexpr int kNmcSampleThreads = 256;
// TODO(release-tuning): derive these sampling partition bounds from
// benchmarked H100 occupancy.
constexpr int kNmcMinSamplePartitions = 16;
constexpr int kNmcMaxSamplePartitions = 256;
// TODO(release-config): expose progress cadence through the runner/server log
// configuration when a stable user-facing logging interface is chosen.
constexpr uint64_t kProgressReportIntervalMs = 5'000;
constexpr uint32_t kPhiloxM4x32A = 0xD2511F53u;
constexpr uint32_t kPhiloxM4x32B = 0xCD9E8D57u;
constexpr uint32_t kPhiloxW32A = 0x9E3779B9u;
constexpr uint32_t kPhiloxW32B = 0xBB67AE85u;

// Stateless Philox4x32-10. The counter encodes the exact (decode step, batch
// row, vocab id), so every variate is reproducible from the request seed alone
// and no per-request RNG state is allocated or shared between threads.
__device__ __forceinline__ uint4 nmc_philox4x32_10(uint4 counter, uint2 key) {
    #pragma unroll
    for (int round = 0; round < 10; ++round) {
        const uint hi0 = __umulhi(kPhiloxM4x32A, counter.x);
        const uint lo0 = kPhiloxM4x32A * counter.x;
        const uint hi1 = __umulhi(kPhiloxM4x32B, counter.z);
        const uint lo1 = kPhiloxM4x32B * counter.z;
        counter = make_uint4(
            hi1 ^ counter.y ^ key.x,
            lo1,
            hi0 ^ counter.w ^ key.y,
            lo0);
        key.x += kPhiloxW32A;
        key.y += kPhiloxW32B;
    }
    return counter;
}

// vLLM-style Gumbel-max sampling: perturb logits with -log(-log(u)), then reduce
// block winners. The stateless Philox source is keyed by
// (seed, step, row, vocab id).
__device__ __forceinline__ float nmc_gumbel(
    uint64_t sampling_seed,
    int step,
    int batch_row,
    int vocab_id) {
    const uint2 key = make_uint2(
        static_cast<uint>(sampling_seed),
        static_cast<uint>(sampling_seed >> 32));
    const uint4 counter = make_uint4(
        static_cast<uint>(vocab_id),
        static_cast<uint>(step),
        static_cast<uint>(batch_row),
        0u);
    // Half an ulp avoids both log(0) and log1p(-1), and matches the ordinary
    // open-interval uniform needed by Gumbel-max.
    const float uniform =
        (static_cast<float>(nmc_philox4x32_10(counter, key).x) + 0.5f) *
        2.3283064365386963e-10f;
    return -logf(-log1pf(-uniform));
}

__device__ __forceinline__ float nmc_warp_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, off);
    }
    return v;
}

// Block-wide sum reduction. `smem` must have >= (blockDim.x / 32) floats.
__device__ __forceinline__ float nmc_block_sum(float v, float* smem) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    v = nmc_warp_sum(v);
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    float r = (threadIdx.x < nwarps) ? smem[threadIdx.x] : 0.0f;
    if (warp == 0) {
        r = nmc_warp_sum(r);
        if (lane == 0) smem[0] = r;
    }
    __syncthreads();
    return smem[0];
}

constexpr int kNmcZeroThreads = 256;
constexpr int kNmcZeroMaxWorkersPerRegion = 8;
constexpr uint64_t kNmcZeroTargetBytesPerWorker = 32 * 1024;

// Zero all resettable decode regions in one launch. Regions are separate
// allocations, so a 2D grid gives each region independent workers without a
// global-offset-to-pointer lookup. The final partial uint4 is handled bytewise
// to avoid writing beyond a tensor's allocated extent.
__global__ void nmc_zero_regions_kernel(
    const NmcZeroRegion* regions,
    int num_regions) {
    const int region_idx = blockIdx.x;
    const int worker_idx = blockIdx.y;
    const int tid = threadIdx.x;
    if (region_idx >= num_regions) return;

    const NmcZeroRegion region = regions[region_idx];
    if (region.ptr == 0 || region.bytes == 0) return;
    const uint64_t workers = min(
        uint64_t(kNmcZeroMaxWorkersPerRegion),
        (region.bytes + kNmcZeroTargetBytesPerWorker - 1) /
            kNmcZeroTargetBytesPerWorker);
    if (uint64_t(worker_idx) >= workers) return;

    constexpr uint64_t kVectorBytes = sizeof(uint4);
    const uint64_t full_vectors = region.bytes / kVectorBytes;
    const uint64_t vectors_per_worker =
        (full_vectors + workers - 1) / workers;
    const uint64_t vector_begin = uint64_t(worker_idx) * vectors_per_worker;
    const uint64_t vector_end = min(full_vectors, vector_begin + vectors_per_worker);
    uint4* const vector_dst = reinterpret_cast<uint4*>(region.ptr);
    for (uint64_t v = vector_begin + uint64_t(tid); v < vector_end;
         v += uint64_t(blockDim.x)) {
        vector_dst[v] = make_uint4(0, 0, 0, 0);
    }

    // Only the final worker writes the tail, and only after all full uint4
    // slots. This path is needed for small uint32 barrier arrays.
    if (worker_idx == int(workers) - 1 && tid == 0) {
        unsigned char* const byte_dst =
            reinterpret_cast<unsigned char*>(region.ptr);
        for (uint64_t byte = full_vectors * kVectorBytes;
             byte < region.bytes; ++byte) {
            byte_dst[byte] = 0;
        }
    }
}

// Seed the layer-0 input for one decode token: x_raw = embed[token];
// x_resid = RMSNorm(x_raw, w_ln0, eps). RMSNorm here normalizes by RMS alone
// and subtracts no mean, which is what NMC's weights were trained against.
//
// NMC uses D=2048, so the hot path mirrors add_rmsnorm_op's 128-bit vector
// layout: 256 threads each move one uint4 (eight bf16 values). The scalar path
// remains for ABI robustness if this kernel is ever used with another width.
__global__ void nmc_seed_input_kernel(
    const long long* generated,
    const int* row_active,
    int token_col,
    int max_new,
    const __nv_bfloat16* embed,
    const __nv_bfloat16* w_ln0,
    __nv_bfloat16* x_resid,
    __nv_bfloat16* x_raw,
    int B,
    int D,
    float eps) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    __shared__ float smem[32];
    // Inactive rows: leave activations as the reset zeroed them.
    if (row_active != nullptr && row_active[b] == 0) return;
    const long long tok = generated[(size_t)b * max_new + token_col];
    const __nv_bfloat16* src = embed + (size_t)tok * D;
    __nv_bfloat16* xr = x_raw + (size_t)b * D;
    __nv_bfloat16* xres = x_resid + (size_t)b * D;
    float ss = 0.0f;

    constexpr int kNmcHidden = 2048;
    constexpr int kVecElements = 8;
    if (D == kNmcHidden) {
        constexpr int kVecsPerThread = kNmcHidden / (kNmcSeedThreads * kVecElements);
        static_assert(
            kNmcHidden % (kNmcSeedThreads * kVecElements) == 0,
            "NMC seed RMSNorm vector layout must cover the hidden width");

        float vals[kVecElements * kVecsPerThread];
        #pragma unroll
        for (int i = 0; i < kVecsPerThread; ++i) {
            const int element = (tid + i * kNmcSeedThreads) * kVecElements;
            const uint4 src_vec = *reinterpret_cast<const uint4*>(src + element);
            const unsigned* src_words = reinterpret_cast<const unsigned*>(&src_vec);
            uint4 raw_vec;
            unsigned* raw_words = reinterpret_cast<unsigned*>(&raw_vec);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 v = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162*>(src_words + j));
                vals[i * kVecElements + j * 2] = v.x;
                vals[i * kVecElements + j * 2 + 1] = v.y;
                ss += v.x * v.x + v.y * v.y;
                const __nv_bfloat162 packed = __floats2bfloat162_rn(v.x, v.y);
                raw_words[j] = *reinterpret_cast<const unsigned*>(&packed);
            }
            *reinterpret_cast<uint4*>(xr + element) = raw_vec;
        }
        ss = nmc_block_sum(ss, smem);
        const float inv = rsqrtf(ss * (1.0f / float(kNmcHidden)) + eps);
        #pragma unroll
        for (int i = 0; i < kVecsPerThread; ++i) {
            const int element = (tid + i * kNmcSeedThreads) * kVecElements;
            const uint4 gamma_vec = *reinterpret_cast<const uint4*>(w_ln0 + element);
            const unsigned* gamma_words = reinterpret_cast<const unsigned*>(&gamma_vec);
            uint4 out_vec;
            unsigned* out_words = reinterpret_cast<unsigned*>(&out_vec);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 gamma = __bfloat1622float2(
                    *reinterpret_cast<const __nv_bfloat162*>(gamma_words + j));
                const __nv_bfloat162 packed = __floats2bfloat162_rn(
                    vals[i * kVecElements + j * 2] * inv * gamma.x,
                    vals[i * kVecElements + j * 2 + 1] * inv * gamma.y);
                out_words[j] = *reinterpret_cast<const unsigned*>(&packed);
            }
            *reinterpret_cast<uint4*>(xres + element) = out_vec;
        }
        return;
    }

    for (int i = tid; i < D; i += nt) {
        const float v = __bfloat162float(src[i]);
        xr[i] = src[i];
        ss += v * v;
    }
    ss = nmc_block_sum(ss, smem);
    const float inv = rsqrtf(ss / float(D) + eps);
    for (int i = tid; i < D; i += nt) {
        const float v = __bfloat162float(xr[i]);
        const float g = __bfloat162float(w_ln0[i]);
        xres[i] = __float2bfloat16(v * inv * g);
    }
}

// Greedy argmax over the vocab dimension, partitioned across blockIdx.y.
// Ties resolve to the lower token id.
__global__ void nmc_greedy_partial_kernel(
    const __nv_bfloat16* logits,
    const int* row_active,
    float* partial_vals,
    int* partial_idxs,
    int V,
    int partitions) {
    const int b = blockIdx.x;
    const int part = blockIdx.y;
    const int tid = threadIdx.x;
    extern __shared__ float smem[];
    float* vals = smem;
    int* idxs = reinterpret_cast<int*>(vals + blockDim.x);

    const int begin = (int)(((long long)part * V) / partitions);
    const int end = (int)(((long long)(part + 1) * V) / partitions);
    if (row_active != nullptr && row_active[b] == 0) {
        if (tid == 0) {
            const int out = b * partitions + part;
            partial_vals[out] = -INFINITY;
            partial_idxs[out] = 0;
        }
        return;
    }
    float best = -INFINITY;
    int best_idx = begin;
    for (int i = begin + tid; i < end; i += blockDim.x) {
        const float v = __bfloat162float(logits[(size_t)b * V + i]);
        if (v > best || (v == best && i < best_idx)) {
            best = v;
            best_idx = i;
        }
    }
    vals[tid] = best;
    idxs[tid] = best_idx;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride &&
            (vals[tid + stride] > vals[tid] ||
             (vals[tid + stride] == vals[tid] && idxs[tid + stride] < idxs[tid]))) {
            vals[tid] = vals[tid + stride];
            idxs[tid] = idxs[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        const int out = b * partitions + part;
        partial_vals[out] = vals[0];
        partial_idxs[out] = idxs[0];
    }
}

// Gumbel-max partial reduction over one vocabulary partition. Samples exactly
// from softmax(logits / temperature) with no probability vector materialised.
// See the DESIGN CREDIT note on nmc_gumbel.
__global__ void nmc_gumbel_partial_kernel(
    const __nv_bfloat16* logits,
    const int* row_active,
    float* partial_vals,
    int* partial_idxs,
    int V,
    int partitions,
    float temperature,
    uint64_t sampling_seed,
    int step) {
    const int b = blockIdx.x;
    const int part = blockIdx.y;
    const int tid = threadIdx.x;
    extern __shared__ float smem[];
    float* vals = smem;
    int* idxs = reinterpret_cast<int*>(vals + blockDim.x);

    const int begin = (int)(((long long)part * V) / partitions);
    const int end = (int)(((long long)(part + 1) * V) / partitions);
    if (row_active != nullptr && row_active[b] == 0) {
        if (tid == 0) {
            const int out = b * partitions + part;
            partial_vals[out] = -INFINITY;
            partial_idxs[out] = 0;
        }
        return;
    }
    float best = -INFINITY;
    int best_idx = begin;
    for (int i = begin + tid; i < end; i += blockDim.x) {
        const float logit = __bfloat162float(logits[(size_t)b * V + i]);
        const float score =
            logit / temperature + nmc_gumbel(sampling_seed, step, b, i);
        if (score > best || (score == best && i < best_idx)) {
            best = score;
            best_idx = i;
        }
    }
    vals[tid] = best;
    idxs[tid] = best_idx;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride &&
            (vals[tid + stride] > vals[tid] ||
             (vals[tid + stride] == vals[tid] && idxs[tid + stride] < idxs[tid]))) {
            vals[tid] = vals[tid + stride];
            idxs[tid] = idxs[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        const int out = b * partitions + part;
        partial_vals[out] = vals[0];
        partial_idxs[out] = idxs[0];
    }
}

// Read once per process: this is a diagnostic switch only. Enabling it adds
// CUDA event creation, synchronization, and stdout I/O to EVERY decode step,
// so it must never be used for throughput measurements.
static bool nmc_detailed_timing_enabled() {
    static const bool enabled = []() {
        const char* env_value = std::getenv("NMC_DETAILED_TIMING");
        return env_value != nullptr && std::string(env_value) == "1";
    }();
    return enabled;
}


__global__ void nmc_greedy_final_kernel(
    const float* partial_vals,
    const int* partial_idxs,
    const int* row_active,
    long long* generated,
    int out_col,
    int max_new,
    int partitions) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    extern __shared__ float smem[];
    float* vals = smem;
    int* idxs = reinterpret_cast<int*>(vals + blockDim.x);

    if (row_active != nullptr && row_active[b] == 0) {
        if (tid == 0) {
            generated[(size_t)b * max_new + out_col] =
                generated[(size_t)b * max_new + out_col - 1];
        }
        return;
    }
    float best = -INFINITY;
    int best_idx = 0;
    for (int p = tid; p < partitions; p += blockDim.x) {
        const int off = b * partitions + p;
        const float v = partial_vals[off];
        const int idx = partial_idxs[off];
        if (v > best || (v == best && idx < best_idx)) {
            best = v;
            best_idx = idx;
        }
    }
    vals[tid] = best;
    idxs[tid] = best_idx;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride &&
            (vals[tid + stride] > vals[tid] ||
             (vals[tid + stride] == vals[tid] && idxs[tid + stride] < idxs[tid]))) {
            vals[tid] = vals[tid + stride];
            idxs[tid] = idxs[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        generated[(size_t)b * max_new + out_col] = idxs[0];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Ragged (continuous-batching) kernels.
//
// Per-row gen_col, temperature, and seed let one launch serve sequences at
// different lengths and sampling settings. The long-lived service uses these;
// runtime_generate uses the uniform kernels above.

// Seed layer-0 input for one decode step with a per-row source column.
__global__ void nmc_seed_input_ragged_kernel(
    const long long* generated,
    const int* row_active,
    const int* gen_col,   // device [B]; row b reads generated[b, gen_col[b]].
    int max_new,
    const __nv_bfloat16* embed,
    const __nv_bfloat16* w_ln0,
    __nv_bfloat16* x_resid,
    __nv_bfloat16* x_raw,
    int B,
    int D,
    float eps) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    __shared__ float smem[32];
    if (row_active != nullptr && row_active[b] == 0) return;
    const int col = gen_col[b];
    const long long tok = generated[(size_t)b * max_new + col];
    const __nv_bfloat16* src = embed + (size_t)tok * D;
    __nv_bfloat16* xr = x_raw + (size_t)b * D;
    __nv_bfloat16* xres = x_resid + (size_t)b * D;
    float ss = 0.0f;

    for (int i = tid; i < D; i += nt) {
        const float v = __bfloat162float(src[i]);
        xr[i] = src[i];
        ss += v * v;
    }
    ss = nmc_block_sum(ss, smem);
    const float inv = rsqrtf(ss / float(D) + eps);
    for (int i = tid; i < D; i += nt) {
        const float v = __bfloat162float(xr[i]);
        const float g = __bfloat162float(w_ln0[i]);
        xres[i] = __float2bfloat16(v * inv * g);
    }
}

// Per-row sample partial: each active row picks greedy (temperature==0) or
// Gumbel-max (temperature>0). `step` participates in the Philox counter so a
// row's variate stream advances with its own gen_col.
__global__ void nmc_sample_partial_ragged_kernel(
    const __nv_bfloat16* logits,
    const int* row_active,
    const int* gen_col,
    const float* temperature,   // device [B]
    const unsigned long long* seed,  // device [B]
    float* partial_vals,
    int* partial_idxs,
    int V,
    int partitions) {
    const int b = blockIdx.x;
    const int part = blockIdx.y;
    const int tid = threadIdx.x;
    extern __shared__ float smem[];
    float* vals = smem;
    int* idxs = reinterpret_cast<int*>(vals + blockDim.x);

    const int begin = (int)(((long long)part * V) / partitions);
    const int end = (int)(((long long)(part + 1) * V) / partitions);
    if (row_active != nullptr && row_active[b] == 0) {
        if (tid == 0) {
            const int out = b * partitions + part;
            partial_vals[out] = -INFINITY;
            partial_idxs[out] = 0;
        }
        return;
    }
    const float temp = temperature[b];
    const bool greedy = (temp == 0.0f);
    const unsigned long long row_seed = seed[b];
    const int step = gen_col[b];
    float best = -INFINITY;
    int best_idx = begin;
    for (int i = begin + tid; i < end; i += blockDim.x) {
        const float logit = __bfloat162float(logits[(size_t)b * V + i]);
        const float score = greedy
            ? logit
            : (logit / temp + nmc_gumbel((uint64_t)row_seed, step, b, i));
        if (score > best || (score == best && i < best_idx)) {
            best = score;
            best_idx = i;
        }
    }
    vals[tid] = best;
    idxs[tid] = best_idx;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride &&
            (vals[tid + stride] > vals[tid] ||
             (vals[tid + stride] == vals[tid] && idxs[tid + stride] < idxs[tid]))) {
            vals[tid] = vals[tid + stride];
            idxs[tid] = idxs[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        const int out = b * partitions + part;
        partial_vals[out] = vals[0];
        partial_idxs[out] = idxs[0];
    }
}

// Per-row final reduction: writes the sampled token to generated[b, gen_col[b]+1].
// Inactive rows are left untouched (no previous-token copy — the service never
// reads inactive rows' tails).
__global__ void nmc_sample_final_ragged_kernel(
    const float* partial_vals,
    const int* partial_idxs,
    const int* row_active,
    const int* gen_col,
    long long* generated,
    int max_new,
    int partitions) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    extern __shared__ float smem[];
    float* vals = smem;
    int* idxs = reinterpret_cast<int*>(vals + blockDim.x);

    if (row_active != nullptr && row_active[b] == 0) return;
    float best = -INFINITY;
    int best_idx = 0;
    for (int p = tid; p < partitions; p += blockDim.x) {
        const int off = b * partitions + p;
        const float v = partial_vals[off];
        const int idx = partial_idxs[off];
        if (v > best || (v == best && idx < best_idx)) {
            best = v;
            best_idx = idx;
        }
    }
    vals[tid] = best;
    idxs[tid] = best_idx;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride &&
            (vals[tid + stride] > vals[tid] ||
             (vals[tid + stride] == vals[tid] && idxs[tid + stride] < idxs[tid]))) {
            vals[tid] = vals[tid + stride];
            idxs[tid] = idxs[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        generated[(size_t)b * max_new + gen_col[b] + 1] = idxs[0];
    }
}

// One contiguous uint32 source region for the watchdog side-stream snapshot.
// dst_offset is in uint32 elements into the staging buffer.
struct NmcWatchdogSnapRegion {
    uint64_t src;
    uint32_t n_u32;
    uint32_t dst_offset;
};

// Force L2-visible reads of device counters the hung megakernel is updating
// (atomics / volatile waits). A plain copy-engine D2H on the same addresses may
// not observe the same view; this matches the waiters' volatile loads.
//
// REQUIRES a free SM when the megakernel grid occupies every other SM; otherwise
// this launch never schedules and the side-stream wait times out (caller then
// falls back to copy-engine D2H).
__global__ void nmc_watchdog_volatile_snapshot_kernel(
    const NmcWatchdogSnapRegion* regions,
    int num_regions,
    uint32_t* staging) {
    const int region_idx = (int)blockIdx.x;
    if (region_idx < 0 || region_idx >= num_regions) return;
    const NmcWatchdogSnapRegion reg = regions[region_idx];
    if (reg.src == 0 || reg.n_u32 == 0) return;
    const uint32_t* src = reinterpret_cast<const uint32_t*>(reg.src);
    uint32_t* dst = staging + reg.dst_offset;
    for (uint32_t i = (uint32_t)threadIdx.x; i < reg.n_u32; i += (uint32_t)blockDim.x) {
        dst[i] = *((volatile const uint32_t*)(src + i));
    }
}

// Build the watchdog snapshot region table for one geometry. Pure host math on
// `L` — touches no CUDA API, so it is safe to call both while the GPU is healthy
// (to size the preallocation) and after the megakernel has wedged.
//
// `out_layout` mirrors the table as JSON so the dump can name each slice of the
// staging buffer. Returns the total uint32 element count.
static uint32_t build_watchdog_snap_regions(
    const NmcLaunchDesc& L,
    std::vector<NmcWatchdogSnapRegion>* out_regions,
    json* out_layout) {
    const int layers = std::max(0, L.num_layers);
    const int hkv = std::max(0, L.Hkv);
    // Current release tilings always have ceil(BS/moe_bm)==1, so one
    // fine-grained bar slot per expert. Landmine if a future tiling uses
    // moe_bm < BS with multiple row-blocks per expert.
    const int moe_row_blocks = NMC_NUM_EXPERTS;
    const int moe_bars = layers * moe_row_blocks;

    struct NamedRegion {
        const char* name;
        uint64_t src;
        uint32_t n_u32;
    };
    const NamedRegion named[] = {
        {"bar_ffn_down", L.bar_ffn_down, (uint32_t)layers},
        {"bar_layer", L.bar_layer, (uint32_t)layers},
        {"bar_qkv", L.bar_qkv, (uint32_t)(layers * hkv)},
        {"bar_route", L.bar_route, (uint32_t)layers},
        {"bar_gather", L.bar_gather, (uint32_t)layers},
        {"bar_moe_upgate", L.bar_moe_upgate, (uint32_t)moe_bars},
        {"bar_moe_down", L.bar_moe_down, (uint32_t)moe_bars},
        {"moe_down_task_count", L.moe_down_task_count, (uint32_t)layers},
        {"moe_down_task_head", L.moe_down_task_head, (uint32_t)layers},
        {"moe_up_task_count", L.moe_up_task_count, (uint32_t)layers},
        {"moe_up_task_head", L.moe_up_task_head, (uint32_t)layers},
        {"expert_counts", L.expert_counts,
         (uint32_t)(layers * NMC_NUM_EXPERTS)},
    };
    constexpr int kNumNamed = (int)(sizeof(named) / sizeof(named[0]));

    if (out_regions) {
        out_regions->clear();
        out_regions->reserve((size_t)kNumNamed);
    }
    if (out_layout) *out_layout = json::array();

    uint32_t cursor = 0;
    for (int i = 0; i < kNumNamed; ++i) {
        if (named[i].src == 0 || named[i].n_u32 == 0) continue;
        if (out_regions) {
            NmcWatchdogSnapRegion reg{};
            reg.src = named[i].src;
            reg.n_u32 = named[i].n_u32;
            reg.dst_offset = cursor;
            out_regions->push_back(reg);
        }
        if (out_layout) {
            json entry;
            entry["name"] = named[i].name;
            entry["n_u32"] = named[i].n_u32;
            entry["dst_offset"] = cursor;
            entry["src"] = named[i].src;
            out_layout->push_back(entry);
        }
        cursor += named[i].n_u32;
    }
    return cursor;
}

}  // namespace

// ── Per-step attention-drain queue builders ──────────────────────────────────
// Must remain bit-identical with schedule.py's split and queue policy; divergence
// can hang or corrupt output. mk::attn_drain_splits exposes the native policy
// for parity tests.

// Release tiling always uses QKV BN=64 (see default_nmc_tiling_for_bs). The
// wait target baked into each ATTN_DECODE queue word is
// (Hq/Hkv + 2) * (head_dim / BN). Recompute rather than plumb another ABI
// field; if the tiling ever changes BN this must change with it.
constexpr int kNmcQkvBn = 64;
constexpr int kAttnSplitCapShortContext = 16;
constexpr int kAttnSplitShortContextTokens = 65537;
constexpr int kAttnDrainMinSplits = 2;  // matches ATTN_DRAIN_MIN_SPLITS
constexpr int kAttnTileBatchShift = 16;

inline int nmc_div_ceil(int n, int d) {
    return (n + d - 1) / d;
}

// Fill splits_out[batch] from live cache_seqlens (write positions). Floors at
// kAttnDrainMinSplits so the unconditional combine wave always has a wait.
inline void nmc_attn_drain_splits(
    const int* cache_seqlens,
    int batch,
    int hkv,
    int num_sms,
    int page_block,
    int max_attn_splits,
    int min_attn_chunk,
    int oversub_k,
    int* splits_out) {
    if (batch <= 0 || hkv <= 0 || num_sms <= 0 || page_block <= 0 ||
        min_attn_chunk <= 0 || min_attn_chunk % page_block != 0) {
        throw std::runtime_error("nmc_attn_drain_splits: invalid geometry");
    }
    int max_splits = std::max(1, max_attn_splits);
    if (max_splits < kAttnDrainMinSplits) {
        throw std::runtime_error(
            "nmc_attn_drain_splits: max_attn_splits < ATTN_DRAIN_MIN_SPLITS");
    }
    const int base_tasks = std::max(1, batch * hkv);
    if (base_tasks >= num_sms) {
        for (int b = 0; b < batch; ++b) splits_out[b] = kAttnDrainMinSplits;
        return;
    }
    std::vector<int> eff(batch);
    int max_eff = 0;
    for (int b = 0; b < batch; ++b) {
        // Full attention only (window=0). attn_decode covers cache_seqlens+1.
        eff[b] = std::max(1, cache_seqlens[b] + 1);
        max_eff = std::max(max_eff, eff[b]);
    }
    if (max_eff <= kAttnSplitShortContextTokens) {
        max_splits = std::min(max_splits, kAttnSplitCapShortContext);
    }
    if (max_splits < kAttnDrainMinSplits) max_splits = kAttnDrainMinSplits;

    const int min_blocks = std::max(1, min_attn_chunk / page_block);
    std::vector<int> w(batch);
    int total_work = 0;
    for (int b = 0; b < batch; ++b) {
        w[b] = std::max(1, nmc_div_ceil(eff[b], page_block));
        total_work += w[b];
    }
    total_work = std::max(1, total_work);
    const int k = std::max(1, oversub_k);
    const int g_den = std::max(1, k * num_sms);
    const int g = std::max(min_blocks, nmc_div_ceil(hkv * total_work, g_den));
    for (int b = 0; b < batch; ++b) {
        int splits = (g > 0) ? (w[b] + g / 2) / g : 1;
        const int max_by_chunk = std::max(1, nmc_div_ceil(w[b], min_blocks));
        splits_out[b] = std::max(
            kAttnDrainMinSplits,
            std::min({splits, max_splits, max_by_chunk}));
    }
}

// Emit ATTN_DECODE instruction words into host_words (capacity words). Returns
// the live queue length. Layout matches Python _attn_decode_inst with LAYER=0
// placeholder (patched at claim time) and WINDOW_SIZE=0 (full only).
inline int nmc_build_attn_queue_words(
    const int* splits,
    int batch,
    int hkv,
    int hq,
    int head_dim,
    int* host_words,
    int capacity_words) {
    if (head_dim % kNmcQkvBn != 0) {
        throw std::runtime_error(
            "nmc_build_attn_queue_words: head_dim must be a multiple of QKV BN=64");
    }
    const int hr = hq / hkv;
    const int qkv_wait = (hr + 2) * (head_dim / kNmcQkvBn);
    int cursor = 0;
    for (int row = 0; row < batch; ++row) {
        const int nsplit = splits[row];
        for (int kvh = 0; kvh < hkv; ++kvh) {
            for (int split = 0; split < nsplit; ++split) {
                if (cursor >= capacity_words) {
                    throw std::runtime_error(
                        "nmc_build_attn_queue_words: queue exceeds capacity");
                }
                int* w = host_words + cursor * NMC_INSTRUCTION_WIDTH;
                for (int i = 0; i < NMC_INSTRUCTION_WIDTH; ++i) w[i] = 0;
                w[attn_field::OPCODE] = static_cast<int>(NmcOpcode::ATTN_DECODE);
                w[attn_field::LAYER] = 0;  // placeholder; patched at claim time
                w[attn_field::BATCH_IDX] = row;
                w[attn_field::KV_HEAD_IDX] = kvh;
                w[attn_field::SPLIT_IDX] = split;
                w[attn_field::NUM_SPLITS] = nsplit;
                w[attn_field::WINDOW_SIZE] = 0;
                w[attn_field::WAIT_TARGET] = qkv_wait;
                w[attn_field::NUM_TILES] = 1;
                w[attn_field::TILE_IDS] =
                    (row << kAttnTileBatchShift) | kvh;
                ++cursor;
            }
        }
    }
    return cursor;
}

// Refresh launch.attn_queue_words / attn_num_splits from live cache_seqlens.
// Re-uploads only when the split vector changed; always publishes queue_len.
// host_splits_cache is length batch (caller-owned sticky cache, init to -1s).
inline void nmc_refresh_attn_drain_queue(
    NmcLaunchDesc* ld,
    const int* cache_seqlens,
    int max_attn_splits,
    int min_attn_chunk,
    std::vector<int>& host_splits_cache,
    std::vector<int>& host_words_scratch,
    cudaStream_t stream) {
    const int B = ld->BS;
    const int hkv = ld->Hkv;
    const int capacity = B * hkv * std::max(1, max_attn_splits);
    if ((int)host_splits_cache.size() != B) {
        host_splits_cache.assign(B, -1);
    }
    std::vector<int> splits(B);
    nmc_attn_drain_splits(
        cache_seqlens, B, hkv, ld->num_sms, ld->page_block_size,
        max_attn_splits, min_attn_chunk, /*oversub_k=*/1, splits.data());
    const bool changed =
        !std::equal(splits.begin(), splits.end(), host_splits_cache.begin());
    host_words_scratch.assign(
        (size_t)capacity * NMC_INSTRUCTION_WIDTH, 0);
    const int queue_len = nmc_build_attn_queue_words(
        splits.data(), B, hkv, ld->Hq, ld->head_dim,
        host_words_scratch.data(), capacity);
    if (changed) {
        if (!ld->attn_queue_words || !ld->attn_num_splits) {
            throw std::runtime_error(
                "attn_drain requires launch.attn_queue_words and attn_num_splits");
        }
        MK_NMC_CUDA_CHECK(cudaMemcpyAsync(
            reinterpret_cast<void*>(ld->attn_queue_words),
            host_words_scratch.data(),
            (size_t)queue_len * NMC_INSTRUCTION_WIDTH * sizeof(int),
            cudaMemcpyHostToDevice, stream));
        MK_NMC_CUDA_CHECK(cudaMemcpyAsync(
            reinterpret_cast<void*>(ld->attn_num_splits),
            splits.data(),
            (size_t)B * sizeof(int),
            cudaMemcpyHostToDevice, stream));
        host_splits_cache = splits;
    }
    ld->attn_queue_len = queue_len;
}

}  // namespace mk

namespace mk {

std::vector<int32_t> attn_drain_splits(const std::vector<int32_t>& cache_seqlens,
                                       int32_t hkv, int32_t num_sms,
                                       int32_t page_block,
                                       int32_t max_attn_splits,
                                       int32_t min_attn_chunk,
                                       int32_t oversub_k) {
    const int32_t batch = static_cast<int32_t>(cache_seqlens.size());
    if (batch < 1) throw MkError("attn_drain_splits needs at least one row");
    std::vector<int32_t> splits(static_cast<size_t>(batch));
    nmc_attn_drain_splits(cache_seqlens.data(), batch, hkv, num_sms, page_block,
                          max_attn_splits, min_attn_chunk, oversub_k,
                          splits.data());
    return splits;
}

}  // namespace mk

namespace mk {

void runtime_generate(NmcRuntimeGenerateDesc& desc) {
    using mk::NmcLaunchDesc;
    float* sample_partial_vals = nullptr;
    int* sample_partial_idxs = nullptr;
    mk::NmcZeroRegion* zero_regions_dev = nullptr;
    const bool detailed_timing = mk::nmc_detailed_timing_enabled();
    try {
        const NmcLaunchDesc& L = desc.launch;
        const int B = L.BS;
        const int D = L.D;
        const int V = desc.vocab_size;
        const int max_new = desc.max_new;
        if (max_new < 1) throw std::runtime_error("max_new must be >= 1");
        if (!is_supported_nmc_batch_size(B)) {
            throw std::runtime_error("BS must be one of {1, 2, 4, 8}");
        }
        if (D < 1) throw std::runtime_error("hidden size must be >= 1");
        if (V < 1) throw std::runtime_error("vocab_size must be >= 1");
        if (!std::isfinite(desc.temperature) || desc.temperature < 0.0f) {
            throw std::runtime_error(
                "NMC runtime temperature must be finite and non-negative");
        }
        if (!std::isfinite(desc.rms_norm_eps) || desc.rms_norm_eps <= 0.0f) {
            throw std::runtime_error(
                "NMC runtime rms_norm_eps must be finite and positive");
        }
        if (desc.start_pos < 0) {
            throw std::runtime_error("NMC runtime start_pos must be non-negative");
        }
        if (!desc.generated_ids) {
            throw std::runtime_error("generated_ids (int64 [BS, max_new]) is required");
        }
        if (!L.W_lmhead || !L.x_resid || !L.x_raw || !L.lm_logits ||
            !desc.w_ln0) {
            throw std::runtime_error(
                "NMC runtime descriptor is missing required model buffers");
        }
        if (!desc.jit_handle) {
            throw std::runtime_error("NMC runtime requires a JIT handle");
        }
        if (!desc.kv_handle) {
            throw std::runtime_error("NMC runtime requires a KV handle");
        }
        const auto& schedule_variants = desc.schedule_variants;
        const int num_schedule_variants =
            static_cast<int>(schedule_variants.size());
        int previous_bucket_upper = 0;
        for (const auto& variant : schedule_variants) {
            if (variant.bucket_upper <= previous_bucket_upper) {
                throw std::runtime_error(
                    "NMC runtime schedule bucket bounds must be strictly increasing");
            }
            if (!variant.inst_buf || !variant.num_inst_per_sm ||
                !variant.jit_handle || variant.max_inst <= 0) {
                throw std::runtime_error(
                    "NMC runtime schedule variant has invalid pointers or dimensions");
            }
            previous_bucket_upper = variant.bucket_upper;
        }
        // Per-row prompt lengths for ragged batches. An empty
        // start_pos_per_row reproduces the uniform behaviour exactly.
        std::vector<int> start_pos_host(B, desc.start_pos);
        if (!desc.start_pos_per_row.empty()) {
            if (static_cast<int>(desc.start_pos_per_row.size()) != B) {
                throw MkError(
                    "start_pos_per_row must have exactly BS entries");
            }
            int max_start_pos = 0;
            for (int b = 0; b < B; ++b) {
                const int32_t row = desc.start_pos_per_row[b];
                if (row < 0) {
                    throw MkError("start_pos_per_row must be non-negative");
                }
                start_pos_host[b] = row;
                max_start_pos = std::max(max_start_pos, row);
            }
            // The horizon check below, the caller's KV arena sizing, and the
            // bucket ladder are all built from the longest row, so a `start_pos`
            // that disagrees with max(start_pos_per_row) would silently
            // under-reserve. Reject instead of quietly picking one.
            if (max_start_pos != desc.start_pos) {
                throw std::runtime_error(
                    "start_pos must equal max(start_pos_per_row) when the "
                    "per-row array is supplied");
            }
        }
        const int64_t required_context =
            static_cast<int64_t>(desc.start_pos) + std::max(0, max_new - 1);
        if (required_context > std::numeric_limits<int>::max()) {
            throw std::runtime_error(
                "NMC runtime decode horizon exceeds the int32 position ABI");
        }
        if (num_schedule_variants > 0 &&
            static_cast<int64_t>(
                schedule_variants[num_schedule_variants - 1].bucket_upper) <
                required_context) {
            throw std::runtime_error(
                "NMC runtime schedule variants do not cover the decode horizon");
        }
        uint64_t total_generated_tokens = 0;
        uint64_t new_tokens_since_last_report = 0;
        auto start_time = std::chrono::high_resolution_clock::now();
        auto last_report_time = start_time;
        auto report_progress = [&](uint64_t new_tokens) {
            new_tokens_since_last_report += new_tokens;
            total_generated_tokens += new_tokens;
            auto current_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_report_time);
            auto avg_throughput_last_interval = double(new_tokens_since_last_report) / double(duration.count()) * 1000.0;
            auto avg_throughput_total = double(total_generated_tokens) / double(std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time).count()) * 1000.0;
            if (duration.count() >= mk::kProgressReportIntervalMs) {
                printf("[mk-release] avg throughput: %.2f tokens/s in last (%.2f) seconds, total throughput: %.2f tokens/s\n", avg_throughput_last_interval, duration.count() / 1000.0, avg_throughput_total);
                fflush(stdout);
                last_report_time = current_time;
                new_tokens_since_last_report = 0;
            }
        };
        cudaStream_t stream = reinterpret_cast<cudaStream_t>(L.stream_u64);
        const int steps = max_new - 1;
        // Outputs live in the descriptor, so size and clear them here rather
        // than trusting whatever a reused descriptor was left holding.
        desc.executed_steps = 0;
        desc.timing_ms.assign(desc.timing ? static_cast<size_t>(std::max(0, steps)) : 0,
                              0.0f);
        long long* generated = reinterpret_cast<long long*>(desc.generated_ids);
        const __nv_bfloat16* embed =
            reinterpret_cast<const __nv_bfloat16*>(L.W_lmhead);
        const __nv_bfloat16* w_ln0 =
            reinterpret_cast<const __nv_bfloat16*>(desc.w_ln0);
        const int* row_active_dev = reinterpret_cast<const int*>(L.row_active);
        const float eps = desc.rms_norm_eps;
        const auto token_callback =
            reinterpret_cast<mk::NmcRuntimeTokenCallback>(
                desc.token_callback);
        void* token_callback_context =
            reinterpret_cast<void*>(desc.token_callback_context);

        // Uploaded straight from the descriptor: it already holds the packed
        // device-side layout, so there is nothing to repack here.
        const int nz = static_cast<int>(desc.zero_regions.size());
        for (const auto& region : desc.zero_regions) {
            if (region.bytes > 0 && !region.ptr) {
                throw MkError(
                    "NMC runtime zero region has bytes but no device pointer");
            }
        }
        if (nz > 0) {
            const size_t zero_bytes_total =
                desc.zero_regions.size() * sizeof(mk::NmcZeroRegion);
            MK_NMC_CUDA_CHECK(cudaMalloc(&zero_regions_dev, zero_bytes_total));
            MK_NMC_CUDA_CHECK(cudaMemcpyAsync(
                zero_regions_dev, desc.zero_regions.data(), zero_bytes_total,
                cudaMemcpyHostToDevice, stream));
        }

        const auto& eos = desc.eos_token_ids;
        auto is_eos = [&](long long t) {
            for (int64_t x : eos) if (t == static_cast<long long>(x)) return true;
            return false;
        };

        // Greedy partition count mirrors runtime_init_sampling in decode/runtime.cu.
        const int num_sms = std::max(1, L.num_sms);
        int partitions = std::max(
            mk::kNmcMinSamplePartitions, (num_sms + B - 1) / B);
        partitions = std::min(partitions, mk::kNmcMaxSamplePartitions);
        partitions = std::min(partitions, V);
        const size_t part_elems = (size_t)B * partitions;
        MK_NMC_CUDA_CHECK(cudaMalloc(&sample_partial_vals, part_elems * sizeof(float)));
        MK_NMC_CUDA_CHECK(cudaMalloc(&sample_partial_idxs, part_elems * sizeof(int)));
        const size_t sample_smem =
            (size_t)mk::kNmcSampleThreads * (sizeof(float) + sizeof(int));

        std::vector<int> row_active(B, 1);
        // Honor the launch descriptor's device row_active (synthetic fixtures
        // may pre-mask empty slots via --fake-prompt-len -1). Fall back to all-
        // active when the pointer is null.
        if (row_active_dev != nullptr) {
            MK_NMC_CUDA_CHECK(cudaMemcpy(
                row_active.data(), row_active_dev,
                (size_t)B * sizeof(int), cudaMemcpyDeviceToHost));
        }
        std::vector<int> gen_len(B, max_new > 0 ? 1 : 0);
        std::vector<int> finish(B, static_cast<int>(FinishReason::kNone));
        std::vector<long long> sampled(B, 0);
        std::vector<int> emitted_rows(B, 0);
        // Sticky per-row split cache for the attn-drain queue refresh. Empty
        // until the first step; nmc_refresh_attn_drain_queue fills it.
        std::vector<int> attn_splits_cache;
        std::vector<int> attn_words_scratch;

        // Seed-token EOS check (column 0 came from prefill).
        if (!eos.empty()) {
            for (int b = 0; b < B; ++b) {
                if (row_active[b] == 0) continue;
                long long tok = 0;
                MK_NMC_CUDA_CHECK(cudaMemcpy(
                    &tok, generated + (size_t)b * max_new,
                    sizeof(long long), cudaMemcpyDeviceToHost));
                if (is_eos(tok)) {
                    row_active[b] = 0;
                    finish[b] = static_cast<int>(FinishReason::kEos);
                }
            }
        }
        // max_new is the generated-id width including the prefill bootstrap in
        // column 0. Zero decode steps means that single token is the entire
        // completion: remaining active rows have hit the length stop.
        if (steps <= 0) {
            for (int b = 0; b < B; ++b) {
                if (row_active[b] == 0) continue;
                row_active[b] = 0;
                finish[b] = static_cast<int>(FinishReason::kLength);
            }
        }

        int executed = 0;
        int schedule_variant_idx = 0;
        for (int step = 0; step < steps; ++step) {
            bool any_active = false;
            for (int b = 0; b < B; ++b) any_active = any_active || (row_active[b] != 0);
            if (!any_active) break;
            ++executed;

            // Debug-only CUDA timeline for one complete decode step. All event
            // pairs measure stream work only; CPU-side KV-manager bookkeeping,
            // kernel launch overhead, and EOS host decisions are deliberately
            // excluded. `total` ends after sampling, before EOS D2H.
            cudaEvent_t step_start = nullptr;
            cudaEvent_t kv_end = nullptr;
            cudaEvent_t prep_end = nullptr;
            cudaEvent_t nmc_end = nullptr;
            cudaEvent_t sample_end = nullptr;
            if (detailed_timing) {
                MK_NMC_CUDA_CHECK(cudaEventCreate(&step_start));
                MK_NMC_CUDA_CHECK(cudaEventCreate(&kv_end));
                MK_NMC_CUDA_CHECK(cudaEventCreate(&prep_end));
                MK_NMC_CUDA_CHECK(cudaEventCreate(&nmc_end));
                MK_NMC_CUDA_CHECK(cudaEventCreate(&sample_end));
                MK_NMC_CUDA_CHECK(cudaEventRecord(step_start, stream));
            }

            // 1) KV page-table update for each row's absolute position. Mirrors
            //    the Python prefill.kv_cache.step_decode_positions(positions,...).
            //    Active rows advance in lockstep (column == step). Masked /
            //    finished rows keep start_pos so the attn-drain refresh does not
            //    invent growing seqlens for empty slots (matches NmcDecodeService).
            std::vector<int> positions(B);
            for (int b = 0; b < B; ++b) {
                positions[b] = start_pos_host[b] +
                    ((row_active[b] != 0) ? step : 0);
            }
            desc.kv_handle->step_decode_positions(
                positions.data(), row_active.data(),
                reinterpret_cast<uint64_t>(stream));
            if (detailed_timing) {
                MK_NMC_CUDA_CHECK(cudaEventRecord(kv_end, stream));
            }

            // 2) Reset scratch + barriers in one multi-region CUDA kernel.
            // The host-to-device region table copy above is ordered before this
            // launch on the same stream and is amortized over the whole decode.
            if (nz > 0) {
                const dim3 zero_grid(
                    static_cast<unsigned>(nz),
                    static_cast<unsigned>(mk::kNmcZeroMaxWorkersPerRegion));
                mk::nmc_zero_regions_kernel<<<
                    zero_grid, mk::kNmcZeroThreads, 0, stream>>>(
                    zero_regions_dev, nz);
                MK_NMC_CUDA_CHECK(cudaGetLastError());
            }

            // 3) Seed layer-0 input from the current token (col == step).
            mk::nmc_seed_input_kernel<<<B, mk::kNmcSeedThreads, 0, stream>>>(
                generated, row_active_dev, step, max_new, embed, w_ln0,
                reinterpret_cast<__nv_bfloat16*>(L.x_resid),
                reinterpret_cast<__nv_bfloat16*>(L.x_raw),
                B, D, eps);
            MK_NMC_CUDA_CHECK(cudaGetLastError());
            if (detailed_timing) {
                MK_NMC_CUDA_CHECK(cudaEventRecord(prep_end, stream));
            }

            // 4) Launch the persistent NMC kernel for one decode step.
            NmcLaunchDesc ld = L;
            mk::JitKernel* step_jit_handle = desc.jit_handle.get();
            if (num_schedule_variants > 0) {
                auto prev_schedule_variant_idx = schedule_variant_idx;
                // Bucket by the longest active row. Selection only advances;
                // after EOS, retaining a wider schedule safely over-splits
                // shorter rows and avoids rewinding queue/combine state.
                int context_len = 0;
                for (int b = 0; b < B; ++b) {
                    if (row_active[b] != 0) {
                        context_len =
                            std::max(context_len, start_pos_host[b] + step + 1);
                    }
                }
                while (schedule_variant_idx + 1 < num_schedule_variants &&
                       context_len > schedule_variants[schedule_variant_idx].bucket_upper) {
                    ++schedule_variant_idx;
                }
                const auto& variant = schedule_variants[schedule_variant_idx];
                if (context_len > variant.bucket_upper) {
                    throw std::runtime_error(
                        "NMC runtime has no schedule bucket for current context");
                }
                if (prev_schedule_variant_idx != schedule_variant_idx) {
                    printf("[mk] using new schedule with bucket_upper: %d\n", variant.bucket_upper);
                }
                ld.inst_buf = variant.inst_buf;
                ld.num_inst_per_sm = variant.num_inst_per_sm;
                ld.max_inst = variant.max_inst;
                // The queue is state-owned and refreshed below from live
                // lengths; variants carry no queue of their own.
                step_jit_handle = variant.jit_handle.get();
            }
            if (desc.attn_drain) {
                // positions[] already holds this step's write positions
                // (cache_seqlens after the KV step). Rebuild the full-attention
                // drain queue from them; skip the H2D when S is unchanged.
                mk::nmc_refresh_attn_drain_queue(
                    &ld, positions.data(),
                    desc.max_attn_splits, desc.min_attn_chunk,
                    attn_splits_cache, attn_words_scratch, stream);
            }
            ld.timing = (desc.timing && step >= desc.warmup) ? 1 : 0;
            const float ms = step_jit_handle->decode_launch(ld);
            if (desc.timing && step >= desc.warmup) {
                desc.timing_ms[step] = ms;
            }
            if (detailed_timing) {
                MK_NMC_CUDA_CHECK(cudaEventRecord(nmc_end, stream));
            }

            // 5) Sample into column (step + 1). The final reduction is shared:
            // partial scores are either logits (greedy) or logits/temperature
            // plus a stateless Philox-derived Gumbel variate.
            dim3 partial_grid((unsigned)B, (unsigned)partitions);
            if (desc.temperature == 0.0f) {
                mk::nmc_greedy_partial_kernel<<<
                    partial_grid, mk::kNmcSampleThreads, sample_smem, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(L.lm_logits),
                    row_active_dev, sample_partial_vals, sample_partial_idxs,
                    V, partitions);
            } else {
                mk::nmc_gumbel_partial_kernel<<<
                    partial_grid, mk::kNmcSampleThreads, sample_smem, stream>>>(
                    reinterpret_cast<const __nv_bfloat16*>(L.lm_logits),
                    row_active_dev, sample_partial_vals, sample_partial_idxs,
                    V, partitions, desc.temperature, desc.sampling_seed, step);
            }
            MK_NMC_CUDA_CHECK(cudaGetLastError());
            mk::nmc_greedy_final_kernel<<<B, mk::kNmcSampleThreads, sample_smem, stream>>>(
                sample_partial_vals, sample_partial_idxs, row_active_dev,
                generated, step + 1, max_new, partitions);
            MK_NMC_CUDA_CHECK(cudaGetLastError());

            if (detailed_timing) {
                MK_NMC_CUDA_CHECK(cudaEventRecord(sample_end, stream));
                MK_NMC_CUDA_CHECK(cudaEventSynchronize(sample_end));
                float kv_ms = 0.0f;
                float prep_ms = 0.0f;
                float nmc_ms = 0.0f;
                float sample_ms = 0.0f;
                float total_ms = 0.0f;
                MK_NMC_CUDA_CHECK(cudaEventElapsedTime(&kv_ms, step_start, kv_end));
                MK_NMC_CUDA_CHECK(cudaEventElapsedTime(&prep_ms, kv_end, prep_end));
                MK_NMC_CUDA_CHECK(cudaEventElapsedTime(&nmc_ms, prep_end, nmc_end));
                MK_NMC_CUDA_CHECK(cudaEventElapsedTime(&sample_ms, nmc_end, sample_end));
                MK_NMC_CUDA_CHECK(cudaEventElapsedTime(&total_ms, step_start, sample_end));
                std::printf(
                    "[mk-release] detailed CUDA step=%d kv=%.3f ms prep=%.3f ms "
                    "nmc=%.3f ms sample=%.3f ms total=%.3f ms\n",
                    step + 1, kv_ms, prep_ms, nmc_ms, sample_ms, total_ms);
                std::fflush(stdout);
                MK_NMC_CUDA_CHECK(cudaEventDestroy(step_start));
                MK_NMC_CUDA_CHECK(cudaEventDestroy(kv_end));
                MK_NMC_CUDA_CHECK(cudaEventDestroy(prep_end));
                MK_NMC_CUDA_CHECK(cudaEventDestroy(nmc_end));
                MK_NMC_CUDA_CHECK(cudaEventDestroy(sample_end));
            }

            // 6) Length / finish-reason bookkeeping + EOS stop.
            {
                auto new_tokens = 0;
                for (int b = 0; b < B; ++b) {
                    if (row_active[b] != 0) {
                        gen_len[b] = step + 2;
                        ++new_tokens;
                    }
                }
                report_progress(new_tokens);
            }

            // EOS handling already makes the sampled token host-visible each
            // step. Streaming reuses that same D2H/synchronization and invokes
            // a deliberately short host callback afterward. The callback must
            // only queue IDs; detokenization/parsing belongs to another thread.
            // When EOS is disabled, enabling the callback necessarily adds the
            // D2H/sync because Python cannot consume a device-only token.
            if (!eos.empty() || token_callback != nullptr) {
                emitted_rows = row_active;

                for (int b = 0; b < B; ++b) {
                    if (row_active[b] == 0) continue;
                    MK_NMC_CUDA_CHECK(cudaMemcpyAsync(
                        sampled.data() + b,
                        generated + (size_t)b * max_new + step + 1,
                        sizeof(long long), cudaMemcpyDeviceToHost, stream));
                }
                MK_NMC_CUDA_CHECK(cudaStreamSynchronize(stream));
                for (int b = 0; b < B; ++b) {
                    if (row_active[b] == 0) continue;
                    if (is_eos(sampled[b])) {
                        row_active[b] = 0;
                        finish[b] = static_cast<int>(FinishReason::kEos);
                    }
                }
                if (token_callback != nullptr) {
                    const int callback_result = token_callback(
                        token_callback_context,
                        sampled.data(),
                        emitted_rows.data(),
                        B);
                    if (callback_result != 0) {
                        break;
                    }
                }
            }

            // Length stop is independent of EOS / the token callback: after this
            // step the row holds `step + 2` tokens (column 0 plus columns
            // 1..step+1). Hitting max_new deactivates the row so later steps do
            // not keep launching work over a finished sequence, and records
            // FinishReason::kLength. EOS already cleared row_active, so those
            // rows keep kEos. This does not need the sampled ids.
            if (step + 2 >= max_new) {
                for (int b = 0; b < B; ++b) {
                    if (row_active[b] == 0) continue;
                    row_active[b] = 0;
                    finish[b] = static_cast<int>(FinishReason::kLength);
                }
            }

        }

        if (desc.generated_lengths) {
            MK_NMC_CUDA_CHECK(cudaMemcpyAsync(
                reinterpret_cast<void*>(desc.generated_lengths),
                gen_len.data(), B * sizeof(int), cudaMemcpyHostToDevice, stream));
        }
        if (desc.finish_reasons) {
            MK_NMC_CUDA_CHECK(cudaMemcpyAsync(
                reinterpret_cast<void*>(desc.finish_reasons),
                finish.data(), B * sizeof(int), cudaMemcpyHostToDevice, stream));
        }
        MK_NMC_CUDA_CHECK(cudaStreamSynchronize(stream));
        desc.executed_steps = executed;
        if (zero_regions_dev) {
            MK_NMC_CUDA_CHECK(cudaFree(zero_regions_dev));
            zero_regions_dev = nullptr;
        }
        MK_NMC_CUDA_CHECK(cudaFree(sample_partial_vals));
        sample_partial_vals = nullptr;
        MK_NMC_CUDA_CHECK(cudaFree(sample_partial_idxs));
        sample_partial_idxs = nullptr;
    } catch (...) {
        // Scratch is raw cudaMalloc, so unwinding has to release it here.
        // noexcept frees: a cleanup failure must not replace the original.
        cuda_free_checked_noexcept(zero_regions_dev, "runtime zero-region table");
        cuda_free_checked_noexcept(sample_partial_vals, "runtime sample values");
        cuda_free_checked_noexcept(sample_partial_idxs, "runtime sample indices");
        throw;
    }
}

}  // namespace mk

// ─────────────────────────────────────────────────────────────────────────────
// Long-lived decode service (single-GPU prefill/decode disaggregation).
//
// Atomic pause/stop/activity flags cover edges while decoding; a condition
// variable blocks while paused or idle. Python may mutate batch composition or
// switch geometry only while parked with the GPU idle. A geometry switch must
// rebind KV ownership, call set_geometry, and restore live slots before resume,
// so the megakernel never observes mixed-geometry pointers.
namespace mk {

class NmcDecodeService {
public:
    explicit NmcDecodeService(const NmcDecodeServiceDesc& d) {
        // Both streams must exist before apply_desc: apply_desc preallocates and
        // uploads the watchdog snapshot table, which needs a stream to copy on.
        // Non-blocking so diagnostic work never joins the hung decode stream.
        MK_NMC_CUDA_CHECK(cudaStreamCreateWithFlags(
            &watchdog_diag_stream_, cudaStreamNonBlocking));
        MK_NMC_CUDA_CHECK(cudaStreamCreateWithFlags(
            &watchdog_copy_stream_, cudaStreamNonBlocking));
        apply_desc(d);
        active_.assign(B_, 0);
        start_pos_.assign(B_, 0);
        gen_col_.assign(B_, 0);
        temperature_.assign(B_, 0.0f);
        seed_.assign(B_, 0ull);
        finish_.assign(B_, 0);
    }

    ~NmcDecodeService() {
        cuda_free_checked_noexcept(sample_partial_vals_, "service sample values");
        cuda_free_checked_noexcept(sample_partial_idxs_, "service sample indices");
        cuda_free_checked_noexcept(zero_regions_dev_, "service zero-region table");
        cuda_free_checked_noexcept(watchdog_staging_dev_, "watchdog staging");
        cuda_free_checked_noexcept(watchdog_regions_dev_, "watchdog snap regions");
        cuda_free_host_checked_noexcept(sampled_host_, "service sampled tokens");
        cuda_free_host_checked_noexcept(watchdog_staging_host_, "watchdog staging host");
        for (cudaStream_t* s : {&watchdog_diag_stream_, &watchdog_copy_stream_}) {
            if (*s == nullptr) continue;
            try {
                MK_NMC_CUDA_CHECK(cudaStreamDestroy(*s));
            } catch (const std::exception& error) {
                std::fprintf(
                    stderr,
                    "[mk-release] failed to destroy watchdog diag stream: %s\n",
                    error.what());
            }
            *s = nullptr;
        }
    }

    // Adopt a new descriptor (launch geometry + schedule variants + zero
    // regions). Deep-copies the host arrays Python may free. NOT locked; the
    // constructor calls it single-threaded and set_geometry holds m_.
    void apply_desc(const NmcDecodeServiceDesc& d) {
        if (!is_supported_nmc_batch_size(d.launch.BS) ||
            d.max_new < 1 || d.vocab_size < 1 ||
            d.max_seq_len < 1) {
            throw std::runtime_error(
                "decode service BS must be one of {1, 2, 4, 8} and dimensions must be positive");
        }
        if (!d.generated_ids || !d.kv_handle || !d.w_ln0 ||
            !d.d_gen_col || !d.d_temperature || !d.d_seed) {
            throw std::runtime_error(
                "decode service descriptor is missing required buffers");
        }
        if (d.schedule_variants.empty() && !d.jit_handle) {
            throw std::runtime_error(
                "decode service requires a JIT handle or schedule variants");
        }
        if (!std::isfinite(d.rms_norm_eps) || d.rms_norm_eps <= 0.0f) {
            throw std::runtime_error(
                "decode service rms_norm_eps must be finite and positive");
        }
        int previous_bucket = 0;
        for (const auto& variant : d.schedule_variants) {
            if (!variant.inst_buf || !variant.num_inst_per_sm ||
                !variant.jit_handle || variant.max_inst < 1 ||
                variant.bucket_upper <= previous_bucket) {
                throw std::runtime_error(
                    "decode service schedule variants are invalid or unsorted");
            }
            previous_bucket = variant.bucket_upper;
        }
        for (const auto& region : d.zero_regions) {
            if (region.bytes > 0 && !region.ptr) {
                throw std::runtime_error(
                    "decode service zero region has bytes but no device pointer");
            }
        }

        // Commit only after every validating allocation succeeds. A failed
        // geometry switch must leave the running service on its old descriptor.
        // The vectors are copied with the descriptor, so the service owns its
        // own storage; nothing here borrows the caller's arrays.
        desc_ = d;
        B_ = d.launch.BS;
        // Force the device zero-region mirror to refresh (pointers changed).
        zero_regions_dev_count_ = -1;
        // Geometry change => force a full queue rebuild on the next step.
        attn_splits_cache_.clear();
        attn_words_scratch_.clear();
        // Debug diagnostics are preallocated while the GPU is healthy; the
        // watchdog path cannot allocate behind a wedge. set_debug must run
        // before service creation or snapshots degrade to host-only.
        if (nmc_debug_mode()) preallocate_watchdog_diag_buffers();
    }

    // Switch the decode geometry to a different session BS mid-flight. The
    // caller (Python scheduler) MUST have the service paused (GPU idle) and
    // MUST re-populate per-row slot state via set_slot for the new geometry
    // afterwards (all rows start inactive here). Always logs the adoption:
    // a same-BS descriptor can still change its schedule/configuration.
    void set_geometry(const NmcDecodeServiceDesc& d) {
        std::lock_guard<std::mutex> lk(m_);
        if (!is_supported_nmc_batch_size(d.launch.BS)) {
            throw std::runtime_error("decode service BS must be one of {1, 2, 4, 8}");
        }
        const int old_B = B_;
        std::vector<int> active(d.launch.BS, 0);
        std::vector<int> start_pos(d.launch.BS, 0);
        std::vector<int> gen_col(d.launch.BS, 0);
        std::vector<float> temperature(d.launch.BS, 0.0f);
        std::vector<unsigned long long> seed(d.launch.BS, 0ull);
        std::vector<int> finish(d.launch.BS, 0);
        apply_desc(d);
        active_ = std::move(active);
        start_pos_ = std::move(start_pos);
        gen_col_ = std::move(gen_col);
        temperature_ = std::move(temperature);
        seed_ = std::move(seed);
        finish_ = std::move(finish);
        recompute_active_rows_locked();
        fprintf(stderr,
                "[nmc-service] geometry switched: BS=%d -> %d, "
                "schedule_buckets=%zu\n",
                old_B, B_, desc_.schedule_variants.size());
        fflush(stderr);
        cv_.notify_all();
    }

    void signal_pause(int v) {
        std::lock_guard<std::mutex> lk(m_);
        pause_req_.store(v ? 1 : 0, std::memory_order_release);
        cv_.notify_all();
    }
    // Non-zero while the run loop has self-parked because the shared KV pool
    // cannot satisfy the next decode step's block allocation. The Python
    // scheduler polls this, frees blocks (reap finished + preempt newest), then
    // calls resume_from_pressure() to release the loop. While parked the loop
    // also acks paused_ack_ (GPU idle), so the normal pause handshake works too.
    int kv_pressure() { return kv_pressure_.load(std::memory_order_acquire); }
    void resume_from_pressure() {
        std::lock_guard<std::mutex> lk(m_);
        kv_pressure_.store(0, std::memory_order_release);
        cv_.notify_all();
    }
    void signal_stop() {
        std::lock_guard<std::mutex> lk(m_);
        stop_.store(1, std::memory_order_release);
        cv_.notify_all();
    }
    void notify() {
        std::lock_guard<std::mutex> lk(m_);
        cv_.notify_all();
    }
    // Blocks until the loop is parked (GPU idle) or timeout. Returns 1 if
    // parked, 0 on timeout. timeout_ms < 0 waits forever.
    int wait_paused(int timeout_ms) {
        std::unique_lock<std::mutex> lk(m_);
        auto ready = [&] {
            return paused_ack_.load(std::memory_order_acquire) != 0 ||
                   stop_.load(std::memory_order_acquire) != 0;
        };
        if (timeout_ms < 0) {
            cv_.wait(lk, ready);
            return ready() ? 1 : 0;
        }
        const bool ok = cv_.wait_for(
            lk, std::chrono::milliseconds(timeout_ms), ready);
        return ok ? 1 : 0;
    }

    // Mutate a slot's control state. Must be called only while the loop is
    // parked (paused or idle); the caller guarantees this via the pause
    // handshake or by admitting into an idle service. Locks m_ for ordering.
    void set_slot(int row, int active, int start_pos, int gen_col,
                  float temperature, unsigned long long seed) {
        std::lock_guard<std::mutex> lk(m_);
        if (row < 0 || row >= B_) {
            throw std::runtime_error("decode service slot row is out of range");
        }
        if (start_pos < 0 || gen_col < 0 ||
            (active && (
                gen_col >= desc_.max_new ||
                static_cast<int64_t>(start_pos) + gen_col >=
                    desc_.max_seq_len))) {
            throw std::runtime_error(
                "decode service slot position is outside the configured context");
        }
        if (!std::isfinite(temperature) || temperature < 0.0f) {
            throw std::runtime_error(
                "decode service slot temperature must be finite and non-negative");
        }
        active_[row] = active ? 1 : 0;
        start_pos_[row] = start_pos;
        gen_col_[row] = gen_col;
        temperature_[row] = temperature;
        seed_[row] = seed;
        finish_[row] = 0;
        recompute_active_rows_locked();
        cv_.notify_all();
    }

    void get_state(int32_t* out_active, int32_t* out_gen_col,
                   int32_t* out_finish) {
        std::lock_guard<std::mutex> lk(m_);
        for (int b = 0; b < B_; ++b) {
            if (out_active) out_active[b] = active_[b];
            if (out_gen_col) out_gen_col[b] = gen_col_[b];
            if (out_finish) out_finish[b] = finish_[b];
        }
    }

    // Under the same lock as get_state, because set_geometry mutates B_ and a
    // caller that read a stale batch would size its get_state buffers wrong --
    // which is the overrun this accessor exists to prevent.
    int batch_size() {
        std::lock_guard<std::mutex> lk(m_);
        return B_;
    }

    int run();

private:
    void recompute_active_rows_locked() {
        int n = 0;
        for (int b = 0; b < B_; ++b) n += (active_[b] != 0);
        active_rows_.store(n, std::memory_order_release);
    }

    bool is_eos(long long t) const {
        // EOS is geometry-independent (the same stop tokens for every BS), but
        // it rides along in the descriptor, so a set_geometry that changed it
        // would take effect -- which is the honest behaviour.
        for (int64_t x : desc_.eos_token_ids) {
            if (t == static_cast<long long>(x)) return true;
        }
        return false;
    }

    // Copied, so the service owns its schedule variants, zero regions and EOS
    // ids outright and depends on no caller storage.
    NmcDecodeServiceDesc desc_;
    int B_ = 0;

    // Host-authoritative per-row state (guarded by m_ for cross-thread writes;
    // read by the loop only while running, i.e. after Python released it).
    std::vector<int> active_;
    std::vector<int> start_pos_;
    std::vector<int> gen_col_;
    std::vector<float> temperature_;
    std::vector<unsigned long long> seed_;
    std::vector<int> finish_;  // 0 none, 1 eos, 2 length.

    std::mutex m_;
    std::condition_variable cv_;
    std::atomic<int> pause_req_{0};
    std::atomic<int> paused_ack_{0};
    std::atomic<int> stop_{0};
    std::atomic<int> active_rows_{0};
    std::atomic<int> kv_pressure_{0};

    // Device scratch owned by the run loop; survives geometry switches so a
    // mid-flight BS change does not require tearing down / recreating the
    // service. (Re)allocated lazily as the active geometry demands.
    float* sample_partial_vals_ = nullptr;
    int* sample_partial_idxs_ = nullptr;
    size_t sample_capacity_ = 0;
    // D2H destinations must be page-locked. A cudaMemcpyAsync into an ordinary
    // std::vector may block the decode thread behind a wedged kernel before it
    // can enter synchronize_decode_step_with_watchdog().
    long long* sampled_host_ = nullptr;
    size_t sampled_host_capacity_ = 0;
    NmcZeroRegion* zero_regions_dev_ = nullptr;
    int zero_regions_dev_count_ = -1;

    // Side-stream snapshot state, allocated eagerly while CUDA is healthy.
    // The watchdog path must never allocate behind a wedged context.
    cudaStream_t watchdog_diag_stream_ = nullptr;
    // Separate stream for the copy-engine fallback: the volatile-load kernel is
    // already queued (and possibly never schedulable) on watchdog_diag_stream_,
    // so a fallback memcpy on that same stream would sit behind it forever.
    cudaStream_t watchdog_copy_stream_ = nullptr;
    uint32_t* watchdog_staging_dev_ = nullptr;
    uint32_t* watchdog_staging_host_ = nullptr;
    size_t watchdog_staging_u32_ = 0;
    NmcWatchdogSnapRegion* watchdog_regions_dev_ = nullptr;
    int watchdog_regions_cap_ = 0;
    // Host mirror of what was uploaded to watchdog_regions_dev_, plus its total
    // size. The snapshot path reads these, so the JSON layout can never
    // disagree with the table the kernel reads.
    std::vector<NmcWatchdogSnapRegion> watchdog_regions_host_;
    json watchdog_layout_;
    uint32_t watchdog_need_u32_ = 0;
    // True once both the volatile kernel and the copy-engine path have each run
    // successfully at least once on a healthy GPU. The watchdog refuses to be
    // the first user of the kernel: a first-ever launch triggers lazy module
    // loading, which blocks behind a wedged megakernel.
    bool watchdog_warmup_ok_ = false;
    // Copy-engine values from the current watchdog pass, kept so the volatile
    // pass can report where the two mechanisms disagree.
    std::vector<uint32_t> watchdog_copy_engine_values_;
    // Why the eager allocation failed/was skipped, surfaced in the dump JSON.
    std::string watchdog_prealloc_error_;

    // Sticky per-row split cache for the attn-drain queue refresh. Cleared on
    // geometry change so the next step always re-uploads.
    std::vector<int> attn_splits_cache_;
    std::vector<int> attn_words_scratch_;

    // After watchdog: side-stream L2-visible snapshot (best effort). May touch
    // CUDA; must not throw. Caller then parks on the decode stream for gdb.
    void dump_watchdog_diagnostics(
        const char* where,
        int attn_queue_len,
        const std::vector<int>* active_snapshot,
        const std::vector<int>* gen_col_snapshot);

    // Allocation-free snapshot via copy engine or volatile-load kernel.
    // The copy-engine pass must run first because it needs no free SM.
    json watchdog_snapshot_stage(bool use_volatile);

    // Preallocate for the current parked geometry; diagnostic failure is nonfatal.
    void preallocate_watchdog_diag_buffers() noexcept;

    // Grow the per-row sample partials to hold at least `need` elements.
    void ensure_sample_capacity(size_t need) {
        if (need <= sample_capacity_) return;
        sample_capacity_ = 0;
        if (sample_partial_vals_) {
            MK_NMC_CUDA_CHECK(cudaFree(sample_partial_vals_));
            sample_partial_vals_ = nullptr;
        }
        if (sample_partial_idxs_) {
            MK_NMC_CUDA_CHECK(cudaFree(sample_partial_idxs_));
            sample_partial_idxs_ = nullptr;
        }
        MK_NMC_CUDA_CHECK(cudaMalloc(&sample_partial_vals_, need * sizeof(float)));
        MK_NMC_CUDA_CHECK(cudaMalloc(&sample_partial_idxs_, need * sizeof(int)));
        sample_capacity_ = need;
    }

    // Grow the page-locked D2H token buffer to the largest geometry seen.
    // Geometry changes happen while the service is parked, so replacing the
    // old allocation cannot race an in-flight copy.
    void ensure_sampled_host_capacity(size_t need) {
        if (need <= sampled_host_capacity_) return;
        sampled_host_capacity_ = 0;
        if (sampled_host_) {
            MK_NMC_CUDA_CHECK(cudaFreeHost(sampled_host_));
            sampled_host_ = nullptr;
        }
        MK_NMC_CUDA_CHECK(cudaMallocHost(
            reinterpret_cast<void**>(&sampled_host_),
            need * sizeof(*sampled_host_)));
        sampled_host_capacity_ = need;
    }

    // Refresh the device mirror of the current geometry's zero regions.
    void ensure_zero_regions_dev(cudaStream_t stream) {
        const int nz = (int)desc_.zero_regions.size();
        if (zero_regions_dev_count_ == nz) return;  // -1 after apply_desc forces refresh
        if (zero_regions_dev_) {
            MK_NMC_CUDA_CHECK(cudaFree(zero_regions_dev_));
            zero_regions_dev_ = nullptr;
        }
        if (nz > 0) {
            MK_NMC_CUDA_CHECK(cudaMalloc(
                &zero_regions_dev_, (size_t)nz * sizeof(NmcZeroRegion)));
            MK_NMC_CUDA_CHECK(cudaMemcpyAsync(
                zero_regions_dev_, desc_.zero_regions.data(),
                (size_t)nz * sizeof(NmcZeroRegion), cudaMemcpyHostToDevice, stream));
            MK_NMC_CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        zero_regions_dev_count_ = nz;
    }
};

void NmcDecodeService::preallocate_watchdog_diag_buffers() noexcept {
    watchdog_prealloc_error_.clear();
    watchdog_regions_host_.clear();
    watchdog_layout_ = json::array();
    watchdog_need_u32_ = 0;
    watchdog_warmup_ok_ = false;
    try {
        if (watchdog_diag_stream_ == nullptr) {
            watchdog_prealloc_error_ = "watchdog_diag_stream_ is null";
            return;
        }
        std::vector<NmcWatchdogSnapRegion> regions;
        json layout;
        const uint32_t need_u32 =
            build_watchdog_snap_regions(desc_.launch, &regions, &layout);
        if (regions.empty() || need_u32 == 0) {
            watchdog_prealloc_error_ = "no device regions to snapshot";
            return;
        }

        // Grow-only: geometry switches reuse a big enough existing allocation so
        // a BS shrink does not churn device memory.
        if ((size_t)need_u32 > watchdog_staging_u32_) {
            cuda_free_checked_noexcept(watchdog_staging_dev_, "watchdog staging");
            watchdog_staging_dev_ = nullptr;
            cuda_free_host_checked_noexcept(
                watchdog_staging_host_, "watchdog staging host");
            watchdog_staging_host_ = nullptr;
            watchdog_staging_u32_ = 0;
            MK_NMC_CUDA_CHECK(cudaMalloc(
                &watchdog_staging_dev_, (size_t)need_u32 * sizeof(uint32_t)));
            MK_NMC_CUDA_CHECK(cudaMallocHost(
                reinterpret_cast<void**>(&watchdog_staging_host_),
                (size_t)need_u32 * sizeof(uint32_t)));
            watchdog_staging_u32_ = (size_t)need_u32;
        }
        const int nreg = (int)regions.size();
        if (nreg > watchdog_regions_cap_) {
            cuda_free_checked_noexcept(
                watchdog_regions_dev_, "watchdog snap regions");
            watchdog_regions_dev_ = nullptr;
            watchdog_regions_cap_ = 0;
            MK_NMC_CUDA_CHECK(cudaMalloc(
                &watchdog_regions_dev_,
                (size_t)nreg * sizeof(NmcWatchdogSnapRegion)));
            watchdog_regions_cap_ = nreg;
        }

        // Upload the table now: the watchdog path must not enqueue an H2D that
        // could fail or stall. GPU is idle here (service paused), so the
        // blocking sync is safe.
        MK_NMC_CUDA_CHECK(cudaMemcpyAsync(
            watchdog_regions_dev_, regions.data(),
            (size_t)nreg * sizeof(NmcWatchdogSnapRegion),
            cudaMemcpyHostToDevice, watchdog_diag_stream_));
        MK_NMC_CUDA_CHECK(
            cudaStreamSynchronize(watchdog_diag_stream_));

        // Warm up before a wedge: lazy module loading can otherwise block behind
        // the resident kernel. A real transfer validates the path.
        nmc_watchdog_volatile_snapshot_kernel<<<
            nreg, 256, 0, watchdog_diag_stream_>>>(
            watchdog_regions_dev_, nreg, watchdog_staging_dev_);
        MK_NMC_CUDA_CHECK(cudaGetLastError());
        MK_NMC_CUDA_CHECK(cudaMemcpyAsync(
            watchdog_staging_host_, watchdog_staging_dev_,
            (size_t)need_u32 * sizeof(uint32_t),
            cudaMemcpyDeviceToHost, watchdog_diag_stream_));
        synchronize_stream_with_timeout(
            watchdog_diag_stream_, kWatchdogWarmupTimeout);
        // Same for the copy-engine path: exercise it once so the watchdog is
        // never the first user of either mechanism.
        for (const auto& reg : regions) {
            MK_NMC_CUDA_CHECK(cudaMemcpyAsync(
                watchdog_staging_host_ + reg.dst_offset,
                reinterpret_cast<const void*>(reg.src),
                (size_t)reg.n_u32 * sizeof(uint32_t),
                cudaMemcpyDeviceToHost, watchdog_copy_stream_));
        }
        synchronize_stream_with_timeout(
            watchdog_copy_stream_, kWatchdogWarmupTimeout);
        watchdog_warmup_ok_ = true;

        watchdog_regions_host_ = std::move(regions);
        watchdog_layout_ = std::move(layout);
        watchdog_need_u32_ = need_u32;
    } catch (const std::exception& e) {
        watchdog_prealloc_error_ = e.what();
        watchdog_regions_host_.clear();
        watchdog_need_u32_ = 0;
    } catch (...) {
        watchdog_prealloc_error_ = "unknown";
        watchdog_regions_host_.clear();
        watchdog_need_u32_ = 0;
    }
}

json NmcDecodeService::watchdog_snapshot_stage(bool use_volatile) {
    json out;
    out["attempted"] = true;
    out["method"] = use_volatile ? "volatile_kernel" : "copy_engine";
    try {
        const NmcLaunchDesc& L = desc_.launch;
        out["moe_row_blocks_assumed"] = NMC_NUM_EXPERTS;
        out["num_layers"] = std::max(0, L.num_layers);
        out["BS"] = L.BS;
        out["layout"] = watchdog_layout_;

        // Report missing preallocation instead of allocating behind the wedge.
        if (watchdog_diag_stream_ == nullptr ||
            watchdog_copy_stream_ == nullptr) {
            out["ok"] = false;
            out["error"] = "watchdog diag streams are null";
            return out;
        }
        if (watchdog_need_u32_ == 0 || watchdog_regions_host_.empty() ||
            watchdog_staging_dev_ == nullptr ||
            watchdog_staging_host_ == nullptr ||
            watchdog_regions_dev_ == nullptr) {
            out["ok"] = false;
            out["error"] =
                watchdog_prealloc_error_.empty()
                    ? std::string("watchdog buffers were never preallocated")
                    : ("watchdog prealloc failed: " + watchdog_prealloc_error_);
            return out;
        }

        const std::vector<NmcWatchdogSnapRegion>& regions_host =
            watchdog_regions_host_;
        const json& layout = watchdog_layout_;
        const size_t need_u32 = (size_t)watchdog_need_u32_;
        const int nreg = (int)regions_host.size();

        if (use_volatile) {
            if (!watchdog_warmup_ok_) {
                out["ok"] = false;
                out["error"] =
                    "refusing to launch the volatile kernel: warmup never "
                    "succeeded, and a first-ever launch triggers lazy CUDA "
                    "module loading which blocks behind the wedged kernel";
                return out;
            }
            // One block per region minimizes footprint, but still needs one free SM.
            nmc_watchdog_volatile_snapshot_kernel<<<
                nreg, 256, 0, watchdog_diag_stream_>>>(
                watchdog_regions_dev_, nreg, watchdog_staging_dev_);
            MK_NMC_CUDA_CHECK(cudaGetLastError());
            MK_NMC_CUDA_CHECK(cudaMemcpyAsync(
                watchdog_staging_host_, watchdog_staging_dev_,
                need_u32 * sizeof(uint32_t),
                cudaMemcpyDeviceToHost, watchdog_diag_stream_));
            synchronize_stream_with_timeout(
                watchdog_diag_stream_, kWatchdogDiagTimeout);
            out["note"] =
                "volatile device loads into staging, then side-stream D2H; "
                "matches what the waiting CTAs see. does not sync the hung "
                "decode stream";
        } else {
            // Copy-engine path needs no free SM or kernel launch and runs first.
            for (const auto& reg : regions_host) {
                MK_NMC_CUDA_CHECK(cudaMemcpyAsync(
                    watchdog_staging_host_ + reg.dst_offset,
                    reinterpret_cast<const void*>(reg.src),
                    (size_t)reg.n_u32 * sizeof(uint32_t),
                    cudaMemcpyDeviceToHost, watchdog_copy_stream_));
            }
            synchronize_stream_with_timeout(
                watchdog_copy_stream_, kWatchdogDiagTimeout);
            out["note"] =
                "copy-engine D2H on a side stream; no kernel launch and no free "
                "SM required. does not sync the hung decode stream";
        }

        const char* const kBinPath =
            use_volatile ? "nmc_watchdog_device_bars_volatile.bin"
                         : "nmc_watchdog_device_bars.bin";
        {
            std::ofstream bin(
                kBinPath, std::ios::out | std::ios::binary | std::ios::trunc);
            if (!bin) {
                out["ok"] = false;
                out["error"] = "failed to open nmc_watchdog_device_bars.bin";
                return out;
            }
            bin.write(
                reinterpret_cast<const char*>(watchdog_staging_host_),
                static_cast<std::streamsize>(need_u32 * sizeof(uint32_t)));
            if (!bin.good()) {
                out["ok"] = false;
                out["error"] = "failed writing nmc_watchdog_device_bars.bin";
                return out;
            }
        }
        out["device_bars_bin"] = kBinPath;
        out["device_bars_bin_dtype"] = "uint32_le";
        out["device_bars_bin_nbytes"] =
            static_cast<long long>(need_u32 * sizeof(uint32_t));

        auto slice_u32 = [&](uint32_t off, uint32_t n) {
            json arr = json::array();
            for (uint32_t i = 0; i < n; ++i) {
                arr.push_back(watchdog_staging_host_[off + i]);
            }
            return arr;
        };
        for (const auto& reg : regions_host) {
            // Embed only the small per-layer vectors in JSON; large MoE bars
            // stay in the bin (indexed via layout).
            if (reg.n_u32 <= 256) {
                for (const auto& entry : layout) {
                    if (entry["dst_offset"].get<uint32_t>() == reg.dst_offset) {
                        out[entry["name"].get<std::string>()] =
                            slice_u32(reg.dst_offset, reg.n_u32);
                        break;
                    }
                }
            }
        }

        // Quick tip scan for MoE down: first layer with a cell in (0, 16).
        // Must match build_watchdog_snap_regions' row-block assumption.
        const int layers = std::max(0, L.num_layers);
        const int moe_row_blocks = NMC_NUM_EXPERTS;
        if (L.bar_moe_down != 0 && layers * moe_row_blocks > 0) {
            uint32_t moe_off = 0;
            bool found_off = false;
            for (const auto& entry : layout) {
                if (entry["name"] == "bar_moe_down") {
                    moe_off = entry["dst_offset"].get<uint32_t>();
                    found_off = true;
                    break;
                }
            }
            if (found_off) {
                json tips = json::array();
                constexpr uint32_t kExpect = 16;  // TinyM down: SPLIT_K*n_tiles
                for (int Lidx = 0; Lidx < layers && (int)tips.size() < 8; ++Lidx) {
                    for (int local = 0; local < moe_row_blocks; ++local) {
                        const uint32_t v = watchdog_staging_host_[
                            moe_off + (uint32_t)(Lidx * moe_row_blocks + local)];
                        if (v > 0 && v < kExpect) {
                            json tip;
                            tip["layer"] = Lidx;
                            tip["local"] = local;
                            tip["flat"] = Lidx * moe_row_blocks + local;
                            tip["value"] = v;
                            tip["expected"] = kExpect;
                            tips.push_back(tip);
                            if ((int)tips.size() >= 8) break;
                        }
                    }
                }
                out["bar_moe_down_partial_tips"] = tips;
            }
        }

        // MoE queue drain check. Workers claim by atomically bumping *_task_head
        // until the claimed index exceeds *_task_count, so a fully drained layer
        // ends at head == count + num_sms (every CTA over-claims exactly once).
        // A layer short of that has CTAs which never reached the claim loop —
        // that is the fingerprint of the hang, so compute it here rather than
        // leaving it to be rediscovered by hand each time.
        {
            auto region_off = [&](const char* name, uint32_t* off) -> bool {
                for (const auto& entry : layout) {
                    if (entry["name"] == name) {
                        *off = entry["dst_offset"].get<uint32_t>();
                        return true;
                    }
                }
                return false;
            };
            json drain = json::array();
            for (const char* phase : {"up", "down"}) {
                const std::string cnt_name =
                    std::string("moe_") + phase + "_task_count";
                const std::string head_name =
                    std::string("moe_") + phase + "_task_head";
                uint32_t cnt_off = 0;
                uint32_t head_off = 0;
                if (!region_off(cnt_name.c_str(), &cnt_off) ||
                    !region_off(head_name.c_str(), &head_off)) {
                    continue;
                }
                for (int Lidx = 0; Lidx < layers; ++Lidx) {
                    const uint32_t cnt =
                        watchdog_staging_host_[cnt_off + (uint32_t)Lidx];
                    const uint32_t head =
                        watchdog_staging_host_[head_off + (uint32_t)Lidx];
                    if (cnt == 0 && head == 0) continue;  // layer not reached
                    const long long over = (long long)head - (long long)cnt;
                    if (over == (long long)L.num_sms) continue;  // fully drained
                    json e;
                    e["phase"] = phase;
                    e["layer"] = Lidx;
                    e["count"] = cnt;
                    e["head"] = head;
                    e["over_claims"] = over;
                    e["expected_over_claims"] = L.num_sms;
                    e["ctas_not_at_claim_loop"] = (long long)L.num_sms - over;
                    drain.push_back(e);
                }
            }
            out["moe_queue_not_drained"] = drain;
        }

        // Record / compare the two mechanisms. The copy-engine pass runs first,
        // so by the time the volatile pass lands we can point at any address
        // where the L2-visible value differs from what the copy engine saw.
        if (!use_volatile) {
            watchdog_copy_engine_values_.assign(
                watchdog_staging_host_, watchdog_staging_host_ + need_u32);
        } else if (watchdog_copy_engine_values_.size() == need_u32) {
            json diffs = json::array();
            size_t n_diff = 0;
            for (size_t i = 0; i < need_u32; ++i) {
                if (watchdog_copy_engine_values_[i] ==
                    watchdog_staging_host_[i]) {
                    continue;
                }
                ++n_diff;
                if (diffs.size() >= 32) continue;
                json d;
                d["index"] = i;
                d["copy_engine"] = watchdog_copy_engine_values_[i];
                d["volatile"] = watchdog_staging_host_[i];
                for (const auto& entry : layout) {
                    const uint32_t off = entry["dst_offset"].get<uint32_t>();
                    const uint32_t n = entry["n_u32"].get<uint32_t>();
                    if (i >= off && i < off + n) {
                        d["name"] = entry["name"];
                        d["elem"] = i - off;
                        break;
                    }
                }
                diffs.push_back(d);
            }
            out["diff_vs_copy_engine_count"] = n_diff;
            out["diff_vs_copy_engine"] = diffs;
            // A nonzero count means the copy engine and the SMs disagree, i.e.
            // some writes are not L2-visible to the copy engine. Both passes are
            // separated in time, so a busy (non-hung) region can also differ.
        }

        out["ok"] = true;
        return out;
    } catch (const std::exception& e) {
        out["ok"] = false;
        out["error"] = e.what();
        return out;
    } catch (...) {
        out["ok"] = false;
        out["error"] = "unknown";
        return out;
    }
}

void NmcDecodeService::dump_watchdog_diagnostics(
    const char* where,
    int attn_queue_len,
    const std::vector<int>* active_snapshot,
    const std::vector<int>* gen_col_snapshot) {
    try {
        const NmcLaunchDesc& L = desc_.launch;
        const int B = B_;
        const int layers = std::max(0, L.num_layers);
        const int hkv = std::max(0, L.Hkv);

        std::vector<int> active_host = active_;
        std::vector<int> gen_col_host = gen_col_;
        std::vector<int> start_pos_host = start_pos_;
        std::vector<int> finish_host = finish_;
        std::vector<int> splits_cache_host = attn_splits_cache_;
        if (active_snapshot != nullptr) active_host = *active_snapshot;
        if (gen_col_snapshot != nullptr) gen_col_host = *gen_col_snapshot;

        std::vector<int> seqlens(std::max(B, 0), 0);
        int context_len = 0;
        for (int b = 0; b < B; ++b) {
            const int sp = (b < (int)start_pos_host.size()) ? start_pos_host[b] : 0;
            const int gc = (b < (int)gen_col_host.size()) ? gen_col_host[b] : 0;
            seqlens[b] = sp + gc;
            const int is_active =
                (b < (int)active_host.size()) ? active_host[b] : 0;
            if (is_active) {
                context_len = std::max(context_len, seqlens[b] + 1);
            }
        }

        std::vector<int> splits_recomputed;
        std::string splits_error;
        if (B > 0 && desc_.attn_drain && L.num_sms > 0 &&
            L.page_block_size > 0 && hkv > 0) {
            try {
                splits_recomputed.assign(B, 0);
                nmc_attn_drain_splits(
                    seqlens.data(), B, hkv, L.num_sms, L.page_block_size,
                    desc_.max_attn_splits, desc_.min_attn_chunk,
                    /*oversub_k=*/1, splits_recomputed.data());
            } catch (const std::exception& e) {
                splits_recomputed.clear();
                splits_error = e.what();
            }
        }

        const auto now = std::chrono::system_clock::now();
        const auto now_s = std::chrono::duration_cast<std::chrono::seconds>(
            now.time_since_epoch()).count();

        int expected_queue_len = 0;
        if (!splits_recomputed.empty() && hkv > 0) {
            for (int s : splits_recomputed) expected_queue_len += s * hkv;
        }

        json dump;
        dump["error"] =
            "NMC decode-step watchdog expired; GPU kernel did not complete";
        dump["where"] = where ? where : "";
        dump["unix_time_s"] = now_s;
        dump["watchdog_timeout_s"] =
            static_cast<long long>(kDecodeStepWatchdogTimeout.count());
        dump["note"] =
            "written twice: host-only first (durable before any CUDA call), then "
            "again with device_snapshot. device_snapshot=={pending:true} means "
            "the side-stream snapshot itself hung/crashed. kv/page_table are "
            "host mirrors. decode stream is NOT synced here — caller parks for gdb";
        dump["BS"] = B;
        dump["num_layers"] = layers;
        dump["Hkv"] = hkv;
        dump["Hq"] = L.Hq;
        dump["num_sms"] = L.num_sms;
        dump["page_block_size"] = L.page_block_size;
        dump["attn_drain"] = desc_.attn_drain;
        dump["max_attn_splits"] = desc_.max_attn_splits;
        dump["min_attn_chunk"] = desc_.min_attn_chunk;
        dump["attn_queue_len"] = attn_queue_len;
        dump["context_len"] = context_len;
        dump["active"] = json_int_array(active_host);
        dump["start_pos"] = json_int_array(start_pos_host);
        dump["gen_col"] = json_int_array(gen_col_host);
        dump["finish"] = json_int_array(finish_host);
        dump["seqlens_host"] = json_int_array(seqlens);
        dump["attn_splits_cache_host"] = json_int_array(splits_cache_host);
        dump["attn_splits_recomputed_host"] = json_int_array(splits_recomputed);
        if (!splits_error.empty()) {
            dump["attn_splits_recompute_error"] = splits_error;
        }
        dump["expected_queue_len_from_recomputed"] = expected_queue_len;
        dump["queue_len_matches_recomputed"] =
            (!splits_recomputed.empty() && attn_queue_len == expected_queue_len);
        dump["splits_cache_matches_recomputed"] =
            (splits_cache_host == splits_recomputed);

        // Host-only KV mirrors.
        {
            kv_pool::KvHostSnapshot snap{};
            snap.page_table_out = nullptr;
            snap.page_table_cap = 0;
            snap.cache_seqlens_out = nullptr;
            snap.row_active_out = nullptr;
            bool have_geometry = true;
            try {
                desc_.kv_handle->host_snapshot(snap);
            } catch (const std::exception& e) {
                dump["kv_host_snapshot_error"] = e.what();
                have_geometry = false;
            }
            if (have_geometry) {
                json pool;
                pool["total_blocks"] = snap.stats.total;
                pool["allocated_blocks"] = snap.stats.allocated;
                pool["peak_allocated_blocks"] = snap.stats.peak_allocated;
                pool["free_blocks"] = snap.stats.free;
                pool["num_phys_pages"] = snap.num_phys_pages;
                pool["page_block_size"] = snap.page_block_size;
                pool["max_pages_per_seq"] = snap.max_pages_per_seq;
                pool["num_layers"] = snap.num_layers;
                pool["BS"] = snap.BS;
                pool["sw_size"] = snap.sw_size;
                pool["sw_pattern"] = snap.sw_pattern;
                pool["page_table_elems"] = snap.page_table_elems;
                dump["kv_pool"] = pool;

                std::vector<int> pt;
                std::vector<int> kv_seqlens(std::max(snap.BS, 0), 0);
                std::vector<int> kv_active(std::max(snap.BS, 0), 0);
                if (snap.page_table_elems > 0) {
                    pt.assign(static_cast<size_t>(snap.page_table_elems), -1);
                }
                snap.page_table_out = pt.empty() ? nullptr : pt.data();
                snap.page_table_cap = snap.page_table_elems;
                snap.cache_seqlens_out =
                    kv_seqlens.empty() ? nullptr : kv_seqlens.data();
                snap.row_active_out =
                    kv_active.empty() ? nullptr : kv_active.data();
                bool have_mirrors = true;
                try {
                    desc_.kv_handle->host_snapshot(snap);
                } catch (const std::exception& e) {
                    dump["kv_host_snapshot_error"] = e.what();
                    have_mirrors = false;
                }
                if (have_mirrors) {
                    dump["cache_seqlens_kv_host"] = json_int_array(kv_seqlens);
                    dump["row_active_kv_host"] = json_int_array(kv_active);
                    dump["page_table_host_layout"] =
                        "[layer][row][page] flat: "
                        "((layer*BS)+row)*max_pages_per_seq + page";
                    constexpr const char* kPtBinPath =
                        "nmc_watchdog_page_table_host.bin";
                    constexpr int kEmbedPtElemsMax = 262144;
                    {
                        std::ofstream bin(
                            kPtBinPath,
                            std::ios::out | std::ios::binary | std::ios::trunc);
                        if (!bin) {
                            dump["page_table_host_bin_error"] =
                                "failed to open nmc_watchdog_page_table_host.bin";
                        } else {
                            bin.write(
                                reinterpret_cast<const char*>(pt.data()),
                                static_cast<std::streamsize>(
                                    pt.size() * sizeof(int32_t)));
                            if (!bin.good()) {
                                dump["page_table_host_bin_error"] =
                                    "failed writing nmc_watchdog_page_table_host.bin";
                            } else {
                                dump["page_table_host_bin"] = kPtBinPath;
                                dump["page_table_host_bin_dtype"] = "int32_le";
                                dump["page_table_host_bin_nbytes"] =
                                    static_cast<long long>(
                                        pt.size() * sizeof(int32_t));
                            }
                        }
                    }
                    if ((int)pt.size() <= kEmbedPtElemsMax) {
                        dump["page_table_host"] = json_int_array(pt);
                    } else {
                        dump["page_table_host_omitted_from_json"] = true;
                        dump["page_table_host_omitted_reason"] =
                            "page_table_elems > 262144; see page_table_host_bin";
                    }

                    // Live-window page_table sanity (host mirror).
                    const int phys = snap.num_phys_pages;
                    const int max_pages = snap.max_pages_per_seq;
                    const int snap_layers = snap.num_layers;
                    const int snap_bs = snap.BS;
                    int missing = 0, invalid_id = 0, oob = 0;
                    json issues = json::array();
                    constexpr int kMaxIssues = 64;
                    for (int layer = 0; layer < snap_layers; ++layer) {
                        for (int row = 0; row < snap_bs && row < B; ++row) {
                            const int seqlen =
                                (row < (int)kv_seqlens.size()) ? kv_seqlens[row] : 0;
                            const int n_pages =
                                max_pages > 0
                                    ? std::min(
                                          max_pages,
                                          (seqlen + snap.page_block_size - 1) /
                                              std::max(snap.page_block_size, 1))
                                    : 0;
                            for (int page = 0; page < n_pages; ++page) {
                                const size_t idx =
                                    ((size_t)layer * (size_t)snap_bs +
                                     (size_t)row) *
                                        (size_t)max_pages +
                                    (size_t)page;
                                if (idx >= pt.size()) {
                                    ++oob;
                                    continue;
                                }
                                const int id = pt[idx];
                                if (id < 0) {
                                    ++missing;
                                    if ((int)issues.size() < kMaxIssues) {
                                        issues.push_back({
                                            {"kind", "unmapped"},
                                            {"layer", layer},
                                            {"row", row},
                                            {"page", page},
                                            {"id", id},
                                        });
                                    }
                                } else if (id >= phys) {
                                    ++invalid_id;
                                    if ((int)issues.size() < kMaxIssues) {
                                        issues.push_back({
                                            {"kind", "invalid_id"},
                                            {"layer", layer},
                                            {"row", row},
                                            {"page", page},
                                            {"id", id},
                                        });
                                    }
                                }
                            }
                        }
                    }
                    int global_invalid = 0;
                    for (int id : pt) {
                        if (id < -1 || id >= phys) ++global_invalid;
                    }
                    json pt_check;
                    pt_check["live_window_unmapped"] = missing;
                    pt_check["live_window_invalid_block_id"] = invalid_id;
                    pt_check["live_window_write_blk_oob"] = oob;
                    pt_check["any_entry_invalid_block_id"] = global_invalid;
                    pt_check["issues_truncated"] =
                        (missing + invalid_id + oob) > (int)issues.size();
                    pt_check["issues"] = issues;
                    dump["page_table_host_check"] = pt_check;
                }
            }
        }

        constexpr const char* kDumpPath = "nmc_watchdog_dump.json";
        auto write_dump = [&]() -> bool {
            std::ofstream out(kDumpPath, std::ios::out | std::ios::trunc);
            if (!out) {
                std::fprintf(
                    stderr,
                    "[mk-release] watchdog dump: failed to open %s\n",
                    kDumpPath);
                return false;
            }
            out << dump.dump(2) << '\n';
            if (!out.good()) {
                std::fprintf(
                    stderr,
                    "[mk-release] watchdog dump: failed writing %s\n",
                    kDumpPath);
                return false;
            }
            return true;
        };

        // Production writes only host state and makes no CUDA call, remaining
        // dependable when the context is wedged.
        if (!nmc_debug_mode()) {
            dump["device_snapshot"] = json{{"skipped", "not in debug mode"}};
            if (!write_dump()) return;
            std::fprintf(
                stderr,
                "[mk-release] watchdog host dump written to ./%s "
                "(where=%s attn_queue_len=%d); run with --debug for the device "
                "snapshot and a cuda-gdb park\n",
                kDumpPath, where ? where : "?", attn_queue_len);
            return;
        }

        // Persist host state before any CUDA operation can block.
        dump["device_snapshot"] = json{{"pending", true}};
        if (!write_dump()) return;
        std::fprintf(
            stderr,
            "[mk-release] watchdog host dump written to ./%s; attempting "
            "side-stream device snapshot\n",
            kDumpPath);

        // Persist the no-SM copy-engine pass before attempting volatile loads.
        const json copy_snap = watchdog_snapshot_stage(/*use_volatile=*/false);
        dump["device_snapshot"] = copy_snap;
        if (!write_dump()) return;
        std::fprintf(
            stderr,
            "[mk-release] watchdog copy-engine snapshot ok=%s%s%s\n",
            copy_snap.value("ok", false) ? "true" : "false",
            copy_snap.contains("error") ? " error=" : "",
            copy_snap.value("error", std::string("")).c_str());

        // Volatile loads match CTA visibility but require a free SM.
        if (copy_snap.value("ok", false)) {
            const json vol_snap = watchdog_snapshot_stage(/*use_volatile=*/true);
            dump["device_snapshot_volatile"] = vol_snap;
            if (!write_dump()) return;
            std::fprintf(
                stderr,
                "[mk-release] watchdog volatile snapshot ok=%s "
                "diff_vs_copy_engine=%lld%s%s\n",
                vol_snap.value("ok", false) ? "true" : "false",
                (long long)vol_snap.value("diff_vs_copy_engine_count", 0),
                vol_snap.contains("error") ? " error=" : "",
                vol_snap.value("error", std::string("")).c_str());
        }
        std::fprintf(
            stderr,
            "[mk-release] watchdog dump written to ./%s "
            "(where=%s attn_queue_len=%d)\n",
            kDumpPath,
            where ? where : "?",
            attn_queue_len);
        if (dump.contains("device_snapshot")) {
            const auto& ds = dump["device_snapshot"];
            std::fprintf(
                stderr,
                "[mk-release] watchdog device_snapshot ok=%s method=%s\n",
                ds.value("ok", false) ? "true" : "false",
                ds.value("method", std::string("?")).c_str());
        }
        if (dump.contains("page_table_host_check")) {
            const auto& chk = dump["page_table_host_check"];
            std::fprintf(
                stderr,
                "[mk-release] watchdog kv_pool allocated=%d/%d free=%d; "
                "page_table live_window unmapped=%d invalid_id=%d "
                "(see page_table_host_bin / page_table_host_check)\n",
                dump.contains("kv_pool")
                    ? dump["kv_pool"].value("allocated_blocks", -1) : -1,
                dump.contains("kv_pool")
                    ? dump["kv_pool"].value("total_blocks", -1) : -1,
                dump.contains("kv_pool")
                    ? dump["kv_pool"].value("free_blocks", -1) : -1,
                chk.value("live_window_unmapped", -1),
                chk.value("live_window_invalid_block_id", -1));
        }
        std::fflush(stderr);
    } catch (const std::exception& e) {
        std::fprintf(
            stderr,
            "[mk-release] watchdog dump failed: %s\n",
            e.what());
        std::fflush(stderr);
    } catch (...) {
        std::fprintf(stderr, "[mk-release] watchdog dump failed: unknown\n");
        std::fflush(stderr);
    }
}

int NmcDecodeService::run() {
    // Geometry may change while parked, so snapshot desc_/B_ each step. Device
    // scratch persists across switches.
    const size_t sample_smem =
        (size_t)kNmcSampleThreads * (sizeof(float) + sizeof(int));

    // Reusable host buffers (resized per step when the geometry BS changes).
    std::vector<int> positions;
    std::vector<int> host_active_snapshot;
    std::vector<int> host_gen_col_snapshot;
    std::vector<float> host_temp_snapshot;
    std::vector<unsigned long long> host_seed_snapshot;
    std::vector<int> emitted_rows;
    // The selected bucket is normally stable for many decode steps. Emit a
    // diagnostic only when the active geometry or schedule configuration changes.
    int logged_bucket_bs = -1;
    int logged_bucket_upper = -1;
    uint64_t logged_bucket_inst_buf = 0;
    mk::JitKernel* logged_bucket_jit_handle = nullptr;

    try {
        for (;;) {
            if (stop_.load(std::memory_order_acquire)) break;

            // Pause only after a watchdog-bounded stream sync confirms GPU idle.
            // Read the geometry-independent stream fresh each iteration.
            cudaStream_t stream =
                reinterpret_cast<cudaStream_t>(desc_.launch.stream_u64);
            if (pause_req_.load(std::memory_order_acquire)) {
                try {
                    synchronize_decode_step_with_watchdog(stream);
                } catch (const DecodeStepWatchdogExpired&) {
                    // Pause sync fires when a prior step's kernel is already
                    // wedged; dump whatever host/device mirrors we still have.
                    dump_watchdog_diagnostics(
                        "pause_sync",
                        /*attn_queue_len=*/desc_.launch.attn_queue_len,
                        /*active_snapshot=*/nullptr,
                        /*gen_col_snapshot=*/nullptr);
                    finish_watchdog_wedge(stream);
                    throw;
                }
                std::unique_lock<std::mutex> lk(m_);
                paused_ack_.store(1, std::memory_order_release);
                cv_.notify_all();
                cv_.wait(lk, [&] {
                    return pause_req_.load(std::memory_order_acquire) == 0 ||
                           stop_.load(std::memory_order_acquire) != 0;
                });
                paused_ack_.store(0, std::memory_order_release);
                continue;
            }

            // Park on idle: no active rows -> sleep (don't launch a full-BS
            // kernel over an all-masked batch) until admitted / paused / stopped.
            if (active_rows_.load(std::memory_order_acquire) <= 0) {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [&] {
                    return active_rows_.load(std::memory_order_acquire) > 0 ||
                           pause_req_.load(std::memory_order_acquire) != 0 ||
                           stop_.load(std::memory_order_acquire) != 0;
                });
                continue;
            }

            // ── Per-step geometry snapshot (stable until the next pause). ──
            const NmcDecodeServiceDesc d = desc_;
            const int B = B_;
            const int D = d.launch.D;
            const int V = d.vocab_size;
            const int max_new = d.max_new;
            const int num_sms = std::max(1, d.launch.num_sms);
            const float eps = d.rms_norm_eps;
            const __nv_bfloat16* embed =
                reinterpret_cast<const __nv_bfloat16*>(d.launch.W_lmhead);
            const __nv_bfloat16* w_ln0 =
                reinterpret_cast<const __nv_bfloat16*>(d.w_ln0);
            long long* generated = reinterpret_cast<long long*>(d.generated_ids);
            const int* row_active_dev =
                reinterpret_cast<const int*>(d.launch.row_active);
            int* d_gen_col = reinterpret_cast<int*>(d.d_gen_col);
            float* d_temperature = reinterpret_cast<float*>(d.d_temperature);
            unsigned long long* d_seed =
                reinterpret_cast<unsigned long long*>(d.d_seed);
            const auto token_callback =
                reinterpret_cast<NmcRuntimeTokenCallback>(d.token_callback);
            void* token_callback_context =
                reinterpret_cast<void*>(d.token_callback_context);

            int partitions = std::max(
                kNmcMinSamplePartitions, (num_sms + B - 1) / B);
            partitions = std::min(partitions, kNmcMaxSamplePartitions);
            partitions = std::min(partitions, V);
            ensure_sample_capacity((size_t)B * partitions);
            ensure_sampled_host_capacity((size_t)B);
            ensure_zero_regions_dev(stream);
            const int nz = (int)desc_.zero_regions.size();

            positions.assign(B, 0);
            host_temp_snapshot.assign(B, 0.0f);
            host_seed_snapshot.assign(B, 0ull);

            // Snapshot per-row state under the lock, then run the step without
            // holding it (kernels + sync take a while; Python only mutates while
            // parked, so this snapshot is stable for the step).
            {
                std::lock_guard<std::mutex> lk(m_);
                host_active_snapshot = active_;
                host_gen_col_snapshot = gen_col_;
                host_temp_snapshot = temperature_;
                host_seed_snapshot = seed_;
                // Length guard: rows that cannot write another column finish now.
                for (int b = 0; b < B; ++b) {
                    if (host_active_snapshot[b] && gen_col_[b] + 1 >= max_new) {
                        active_[b] = 0;
                        finish_[b] = static_cast<int>(FinishReason::kLength);
                        host_active_snapshot[b] = 0;
                    }
                }
                recompute_active_rows_locked();
            }
            bool any = false;
            for (int b = 0; b < B; ++b) any = any || host_active_snapshot[b];
            if (!any) continue;  // re-evaluate idle park

            // 1) KV page-table update: per-row absolute positions.
            for (int b = 0; b < B; ++b) {
                positions[b] = start_pos_[b] + host_gen_col_snapshot[b];
            }

            // 1a) Conservatively reserve one block per layer for each row
            // crossing a page boundary. Park for Python preemption before OOM;
            // ignoring SWA eviction and prefix reuse can only park early.
            {
                const int page_block = d.launch.page_block_size;
                const int num_layers = std::max(1, d.launch.num_layers);
                int crossing = 0;
                for (int b = 0; b < B; ++b) {
                    if (!host_active_snapshot[b]) continue;
                    if (page_block > 0 && (positions[b] % page_block) == 0) {
                        ++crossing;
                    }
                }
                if (crossing > 0) {
                    const int need = crossing * num_layers;
                    const int free_blocks = d.kv_handle->free_blocks();
                    if (free_blocks < need) {
                        std::unique_lock<std::mutex> lk(m_);
                        kv_pressure_.store(1, std::memory_order_release);
                        paused_ack_.store(1, std::memory_order_release);
                        cv_.notify_all();
                        cv_.wait(lk, [&] {
                            return kv_pressure_.load(std::memory_order_acquire) == 0 ||
                                   stop_.load(std::memory_order_acquire) != 0;
                        });
                        paused_ack_.store(0, std::memory_order_release);
                        continue;  // re-snapshot geometry/state and retry the step
                    }
                }
            }

            d.kv_handle->step_decode_positions(
                positions.data(), host_active_snapshot.data(),
                reinterpret_cast<uint64_t>(stream));

            // 2) Reset scratch + barriers.
            if (nz > 0) {
                const dim3 zero_grid(
                    static_cast<unsigned>(nz),
                    static_cast<unsigned>(kNmcZeroMaxWorkersPerRegion));
                nmc_zero_regions_kernel<<<zero_grid, kNmcZeroThreads, 0, stream>>>(
                    zero_regions_dev_, nz);
                MK_NMC_CUDA_CHECK(cudaGetLastError());
            }

            // 3) Upload per-row control to device (tiny; ordered on the stream).
            MK_NMC_CUDA_CHECK(cudaMemcpyAsync(
                d_gen_col, host_gen_col_snapshot.data(), B * sizeof(int),
                cudaMemcpyHostToDevice, stream));
            MK_NMC_CUDA_CHECK(cudaMemcpyAsync(
                d_temperature, host_temp_snapshot.data(), B * sizeof(float),
                cudaMemcpyHostToDevice, stream));
            MK_NMC_CUDA_CHECK(cudaMemcpyAsync(
                d_seed, host_seed_snapshot.data(), B * sizeof(unsigned long long),
                cudaMemcpyHostToDevice, stream));

            // 4) Seed layer-0 input from each row's current column.
            nmc_seed_input_ragged_kernel<<<B, kNmcSeedThreads, 0, stream>>>(
                generated, row_active_dev, d_gen_col, max_new, embed, w_ln0,
                reinterpret_cast<__nv_bfloat16*>(d.launch.x_resid),
                reinterpret_cast<__nv_bfloat16*>(d.launch.x_raw),
                B, D, eps);
            MK_NMC_CUDA_CHECK(cudaGetLastError());

            // 5) Select schedule variant by the MAX active context length, then
            //    launch the persistent kernel for one decode step.
            NmcLaunchDesc ld = d.launch;
            mk::JitKernel* step_jit_handle = d.jit_handle.get();
            if (!desc_.schedule_variants.empty()) {
                int context_len = 0;
                for (int b = 0; b < B; ++b) {
                    if (host_active_snapshot[b]) {
                        context_len = std::max(
                            context_len, start_pos_[b] + host_gen_col_snapshot[b] + 1);
                    }
                }
                int vi = 0;
                while (vi + 1 < (int)desc_.schedule_variants.size() &&
                       context_len > desc_.schedule_variants[vi].bucket_upper) {
                    ++vi;
                }
                if (context_len > desc_.schedule_variants[vi].bucket_upper) {
                    throw std::runtime_error(
                        "decode service has no schedule bucket for current context");
                }
                const auto& variant = desc_.schedule_variants[vi];
                if (logged_bucket_bs != B ||
                    logged_bucket_upper != variant.bucket_upper ||
                    logged_bucket_inst_buf != variant.inst_buf ||
                    logged_bucket_jit_handle != variant.jit_handle.get()) {
                    fprintf(stderr,
                            "[nmc-service] using bucket (bs=%d, seqlen=%d) "
                            "context=%d variant=%d/%zu\n",
                            B, variant.bucket_upper, context_len, vi + 1,
                            desc_.schedule_variants.size());
                    fflush(stderr);
                    logged_bucket_bs = B;
                    logged_bucket_upper = variant.bucket_upper;
                    logged_bucket_inst_buf = variant.inst_buf;
                    logged_bucket_jit_handle = variant.jit_handle.get();
                }
                ld.inst_buf = variant.inst_buf;
                ld.num_inst_per_sm = variant.num_inst_per_sm;
                ld.max_inst = variant.max_inst;
                // Queue is state-owned and refreshed below from live lengths.
                step_jit_handle = variant.jit_handle.get();
            }
            if (d.attn_drain) {
                // Live write positions = start_pos + gen_col for active rows.
                // Finished / empty rows keep start_pos (harmless: their tasks
                // still claim but row_is_active skips the real work).
                std::vector<int> seqlens(B);
                for (int b = 0; b < B; ++b) {
                    seqlens[b] = start_pos_[b] + host_gen_col_snapshot[b];
                }
                mk::nmc_refresh_attn_drain_queue(
                    &ld, seqlens.data(),
                    d.max_attn_splits, d.min_attn_chunk,
                    attn_splits_cache_, attn_words_scratch_, stream);
            }
            ld.timing = 0;
            const float ms = step_jit_handle->decode_launch(ld);
            
            // 6) Sample into each active row's next column (per-row greedy/Gumbel).
            dim3 partial_grid((unsigned)B, (unsigned)partitions);
            nmc_sample_partial_ragged_kernel<<<
                partial_grid, kNmcSampleThreads, sample_smem, stream>>>(
                reinterpret_cast<const __nv_bfloat16*>(d.launch.lm_logits),
                row_active_dev, d_gen_col, d_temperature, d_seed,
                sample_partial_vals_, sample_partial_idxs_, V, partitions);
            MK_NMC_CUDA_CHECK(cudaGetLastError());
            nmc_sample_final_ragged_kernel<<<B, kNmcSampleThreads, sample_smem, stream>>>(
                sample_partial_vals_, sample_partial_idxs_, row_active_dev,
                d_gen_col, generated, max_new, partitions);
            MK_NMC_CUDA_CHECK(cudaGetLastError());

            // 7) D2H each active row's sampled token, then EOS + advance + stream.
            emitted_rows = host_active_snapshot;
            for (int b = 0; b < B; ++b) {
                if (!host_active_snapshot[b]) continue;
                MK_NMC_CUDA_CHECK(cudaMemcpyAsync(
                    sampled_host_ + b,
                    generated + (size_t)b * max_new + host_gen_col_snapshot[b] + 1,
                    sizeof(long long), cudaMemcpyDeviceToHost, stream));
            }
            try {
                synchronize_decode_step_with_watchdog(stream);
            } catch (const DecodeStepWatchdogExpired&) {
                dump_watchdog_diagnostics(
                    "decode_step_sync",
                    /*attn_queue_len=*/ld.attn_queue_len,
                    &host_active_snapshot,
                    &host_gen_col_snapshot);
                finish_watchdog_wedge(stream);
                throw;
            }

            {
                std::lock_guard<std::mutex> lk(m_);
                for (int b = 0; b < B; ++b) {
                    if (!host_active_snapshot[b]) continue;
                    // Advance the row's column (it now holds one more token).
                    gen_col_[b] = host_gen_col_snapshot[b] + 1;
                    if (is_eos(sampled_host_[b])) {
                        active_[b] = 0;
                        finish_[b] = static_cast<int>(FinishReason::kEos);
                    }
                }
                recompute_active_rows_locked();
            }

            // Stream tokens to Python (per-slot demux happens Python-side). The
            // callback must be short (queue IDs only). A nonzero result is fatal
            // because that is how the binding reports a Python exception, which
            // has no other way across this C ABI.
            if (token_callback != nullptr) {
                const int callback_result = token_callback(
                    token_callback_context, sampled_host_,
                    emitted_rows.data(), B);
                if (callback_result != 0) {
                    throw std::runtime_error("NMC token callback failed");
                }
            }
        }
        return 0;
    } catch (const DecodeStepWatchdogExpired& e) {
        // Preserve the watchdog-specific return code through the C ABI. Generic
        // service failures remain recoverable at the session layer; this one is
        // process-fatal because the kernel is still resident on the GPU.
        g_nmc_last_error = e.what();
        return kDecodeServiceWatchdogExpired;
    } catch (const std::exception& e) {
        // Device scratch is freed by the destructor (members outlive run()).
        g_nmc_last_error = e.what();
        return -1;
    }
}

}  // namespace mk

// ─────────────────────────────────────────────────────────────────────────────
// Public API (declared in decode/abi.h).
//
// Nanobind objects and descriptors share ownership of services and JIT kernels.

namespace mk {

// ── Profiler ─────────────────────────────────────────────────────────────────

Profiler::Profiler(int32_t num_sms, int32_t max_events) {
    if (num_sms <= 0) throw MkError("profiler num_sms must be >= 1");
    if (max_events <= 0) throw MkError("profiler max_events must be >= 1");
    buffer_ = sm_profiler_create_buffer(
        static_cast<uint32_t>(num_sms),
        /*num_groups=*/1,
        static_cast<uint32_t>(max_events),
        /*enabled=*/1);
    if (!buffer_) throw MkError("sm_profiler_create_buffer failed");

    using O = NmcOpcode;
    sm_profiler_register_event(buffer_, (uint32_t)O::NOP, "nop");
    sm_profiler_register_event(buffer_, (uint32_t)O::FFN_DOWN, "ffn_down");
    sm_profiler_register_event(buffer_, (uint32_t)O::QKV_PROJ, "qkv");
    sm_profiler_register_event(buffer_, (uint32_t)O::ATTN_DECODE, "attn_decode");
    sm_profiler_register_event(buffer_, (uint32_t)O::ATTN_COMBINE, "attn_combine");
    sm_profiler_register_event(buffer_, (uint32_t)O::O_PROJ, "oproj");
    sm_profiler_register_event(buffer_, (uint32_t)O::LM_HEAD, "lm_head");
    sm_profiler_register_event(buffer_, (uint32_t)O::FFN_UPGATE_ACT, "ffn_upgate_act");
    sm_profiler_register_event(buffer_, (uint32_t)O::ATTN_DRAIN, "attn_drain");
    sm_profiler_register_event(buffer_, (uint32_t)O::ROUTER_GEMM, "router");
    sm_profiler_register_event(buffer_, (uint32_t)O::ROUTER_TOPK, "router_topk");
    sm_profiler_register_event(buffer_, (uint32_t)O::ROUTE_FINALIZE, "route_finalize");
    sm_profiler_register_event(buffer_, (uint32_t)O::MOE_GATHER, "moe_gather");
    sm_profiler_register_event(
        buffer_, (uint32_t)O::MOE_UPGATE_ACT_DRAIN, "moe_upgate_act");
    sm_profiler_register_event(buffer_, (uint32_t)O::MOE_DOWN_DRAIN, "moe_down");
    sm_profiler_register_event(buffer_, (uint32_t)O::MOE_COMBINE, "moe_combine");
    sm_profiler_register_event(buffer_, (uint32_t)O::ADD_RMSNORM, "rmsnorm");
    sm_profiler_register_event(buffer_, (uint32_t)O::GRID_SYNC, "GRID_SYNC");
    sm_profiler_init_buffer(buffer_);
}

Profiler::~Profiler() { sm_profiler_destroy_buffer(buffer_); }

uint64_t Profiler::device_ptr() const {
    return reinterpret_cast<uint64_t>(sm_profiler_get_device_ptr(buffer_));
}

void Profiler::init() { sm_profiler_init_buffer(buffer_); }

void Profiler::export_to(const std::string& filename) {
    if (sm_profiler_export_to_file_compact(buffer_, filename.c_str()) != 0) {
        throw MkError("profiler export failed: " + filename);
    }
}

// ── DecodeService ────────────────────────────────────────────────────────────

// NmcDecodeService is defined above in this TU and holds CUDA-typed members, so
// it cannot appear in decode/abi.h. `Impl` is just that class under the name the
// header promised.
class DecodeService::Impl : public NmcDecodeService {
 public:
    using NmcDecodeService::NmcDecodeService;
};

DecodeService::DecodeService(const NmcDecodeServiceDesc& desc) {
    if (!is_supported_nmc_batch_size(desc.launch.BS)) {
        throw MkError("service BS must be one of {1, 2, 4, 8}");
    }
    if (desc.max_new < 1) throw MkError("service max_new must be >= 1");
    if (!desc.generated_ids || !desc.kv_handle) {
        throw MkError("service requires generated_ids and kv_handle");
    }
    p_ = std::make_shared<Impl>(desc);
}

DecodeService::~DecodeService() = default;

// Methods copy the shared_ptr so concurrent close cannot free an in-flight call;
// calls made after close throw.
#define MK_SERVICE_IMPL()                                                     \
    std::shared_ptr<Impl> impl = p_;                                          \
    if (!impl) throw MkError("decode service is closed")

int32_t DecodeService::run() {
    // Blocking: called from the decode-driver thread with the GIL released.
    std::shared_ptr<Impl> impl = p_;
    if (!impl) { g_nmc_last_error = "decode service is closed"; return -1; }
    return impl->run();
}

void DecodeService::signal_pause(bool paused) {
    MK_SERVICE_IMPL();
    impl->signal_pause(paused ? 1 : 0);
}

void DecodeService::signal_stop() {
    // Tolerates a closed service: close() already stopped the loop, and
    // shutdown paths call these in whatever order they unwind.
    std::shared_ptr<Impl> impl = p_;
    if (impl) impl->signal_stop();
}

bool DecodeService::wait_paused(int32_t timeout_ms) {
    std::shared_ptr<Impl> impl = p_;
    return impl ? impl->wait_paused(timeout_ms) != 0 : false;
}

void DecodeService::notify() {
    MK_SERVICE_IMPL();
    impl->notify();
}

void DecodeService::set_slot(int32_t row, bool active, int32_t start_pos,
                             int32_t gen_col, float temperature, uint64_t seed) {
    MK_SERVICE_IMPL();
    impl->set_slot(row, active ? 1 : 0, start_pos, gen_col, temperature, seed);
}

void DecodeService::get_state(int32_t* out_active, int32_t* out_gen_col,
                              int32_t* out_finish) {
    MK_SERVICE_IMPL();
    impl->get_state(out_active, out_gen_col, out_finish);
}

int32_t DecodeService::batch_size() {
    MK_SERVICE_IMPL();
    return impl->batch_size();
}

void DecodeService::set_geometry(const NmcDecodeServiceDesc& desc) {
    MK_SERVICE_IMPL();
    impl->set_geometry(desc);
}

int32_t DecodeService::kv_pressure() {
    std::shared_ptr<Impl> impl = p_;
    return impl ? impl->kv_pressure() : 0;
}

void DecodeService::resume_from_pressure() {
    MK_SERVICE_IMPL();
    impl->resume_from_pressure();
}

void DecodeService::close() {
    std::shared_ptr<Impl> impl = std::move(p_);
    p_.reset();
    if (impl) impl->signal_stop();
    // impl's own reference dies here, but any in-flight call holds one of its
    // own, so the service outlives this call if it has to.
}

#undef MK_SERVICE_IMPL

// ── Free functions ───────────────────────────────────────────────────────────

const char* last_error() { return g_nmc_last_error.c_str(); }

void set_debug(bool enabled) {
    g_nmc_debug_mode.store(enabled, std::memory_order_relaxed);
}

}  // namespace mk
