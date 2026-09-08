# Build and Run

Detailed build, run, benchmark, and profiling instructions for the megakernel
serving engine. For a project overview and a minimal quick start, see
[README.md](README.md). For the design story, see
the [blog post](https://cohere.com/blog/megakernels).

## Repository layout

Source is grouped by inference phase. The megakernel covers decode only, so
it lives entirely under `src/decode/`. Prefill is a PyTorch path of its own,
and the KV cache is shared by both phases.

**`src/decode/` — the megakernel**

| Path | Role |
| --- | --- |
| `megakernel.cuh` | Persistent workers, GEMMs, paged attention, MoE |
| `launch.cuh` | Kernel parameters and launch configuration |
| `gemm-n8-wgmma.cuh` | Extra WGMMA `m64n8k16` path for decode GEMMs (not in stock ThunderKittens) |
| `runtime.cu` | Host runtime, sampling kernels, decode service, JIT integration |
| `abi.h` | Descriptor structs and runtime classes shared by `runtime.cu` and the bindings |
| `schedule.py` | Megakernel task schedule construction, JIT launch, decode-descriptor construction |

**`src/prefill/` — PyTorch prefill**

| Path | Role |
| --- | --- |
| `torch_prefill.py` | Checkpoint loading and chunked PyTorch/FlashAttention prefill |

**`src/kv/` — shared by both phases**

| Path | Role |
| --- | --- |
| `cache.{h,cpp}` | Paged KV pool, sliding-window eviction, prefix cache |
| `pool.py` | Python-side paged-KV state and the shared physical K/V arena |

**`src/serving/` — OpenAI-compatible server**

| Path | Role |
| --- | --- |
| `server.py` | OpenAI-compatible HTTP server |
| `session.py` | Continuous batching, request admission, prefix-aware prefill |
| `parse.py` | Cohere Melody cmd4 output parsing |

**Everything else**

| Path | Role |
| --- | --- |
| `src/native.py` | Binds the process to one build (`--lib`) and hands out the extension module |
| `src/runner.py` | Standalone generation, benchmarks, profiling |
| `src/bindings/` | nanobind bindings, built into `mk_ext.abi3.so` |
| `src/jit.{cpp,hpp}` | `little_jit`, the content-addressed NVRTC/NVCC cache |
| `src/sm-profiler/` | Per-SM profiling, adapted from [leepoly/sm-profiler](https://github.com/leepoly/sm-profiler) |
| `src/tests/` | Binding, token-callback, and per-layer decode suites |
| `ext/ThunderKittens/` | CUDA tile primitives (submodule) |
| `ext/nanobind/` | Python binding library (submodule) |

## Requirements

The build targets a narrow, tested configuration:

- Linux
- NVIDIA H100 / Hopper with compute capability `sm_90a`
- CUDA 13 or newer (CUDA 12 is not recommended)
- A C++20-compatible host compiler
- CMake 3.24+ and Ninja
- OpenSSL development headers and libraries
- CPython 3.12+, with the packages in `requirements.txt` plus FlashAttention-3
- A local North Mini Code checkpoint

Other NVIDIA architectures are not currently built by `CMakeLists.txt`.

## Install and build

Clone with submodules ([ThunderKittens](https://github.com/HazyResearch/ThunderKittens)
and [nanobind](https://github.com/wjakob/nanobind) are required):

```bash
git clone --recurse-submodules https://github.com/cohere-ai/megakernel.git
cd megakernel
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

Create an environment (we use [`uv`](https://docs.astral.sh/uv/)) and install
the Python dependencies plus the OpenSSL headers CMake needs:

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

sudo apt update
sudo apt install -y libssl-dev ninja-build
```

FlashAttention-3 is not on PyPI for CUDA 13; install it from the PyTorch
wheel index:

```bash
uv pip install flash-attn-3 --index-url https://download.pytorch.org/whl/cu130
```

Build from the repository root. **Pass the interpreter you intend to run
with** — the build produces a Python extension module, and CMake otherwise
picks whichever Python it finds first, which may not be your virtualenv:

```bash
cmake -S . -B build -G Ninja \
  -DPython_EXECUTABLE="$(which python)" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

This produces two files, `build/libmk_release.so` and `build/mk_ext.abi3.so`.
They must stay side by side: the extension links the library with
`RUNPATH=$ORIGIN`, so copying only one elsewhere fails at *import* time
rather than at build time. The extension is built against the CPython stable
ABI, so it works with any CPython ≥ 3.12 without rebuilding.

## Tests

The suites live under `src/tests/` and run as plain scripts. They look for
`build/libmk_release.so` next to the repo root (`--lib-path` overrides that).
`test_kv_bindings.py` and `test_decode_layers.py` need a GPU;
`test_token_callback.py` does not. `test_decode_layers.py` also needs a
local checkpoint.

```bash
python src/tests/test_kv_bindings.py        # paged-KV / prefix-cache bindings
python src/tests/test_token_callback.py     # token-callback trampoline (no CUDA needed)
python src/tests/test_decode_layers.py --device cuda:0 --checkpoint <checkpoint-path>
```

`src/tests/test_binding_invariants.py` is a helper module, not a suite: it
holds the shared runner plus the source-level audits (every binding that
calls into native code must release the GIL) that the two binding suites
invoke.

## Server

The server is OpenAI-compatible and supports continuous batching, prefix
caching, streaming, and tool calling. Maximum batch size is 8.

Run from the repository root:

```bash
python src/serving/server.py \
  --host 127.0.0.1 --port 8000 \
  --lib "$PWD/build/libmk_release.so" \
  --ckpt <checkpoint-path> \
  --device cuda:0 \
  --bs 8 \
  --frac-vram-utilization 0.7 \
  --max-context 262144 \
  --prefill-chunk-size 4096
```

### Server flags

Commonly used flags (defaults from `src/serving/server.py`):

| Flag | Default | Notes |
| --- | --- | --- |
| `--host` / `--port` | `0.0.0.0` / `8000` | Bind address. |
| `--lib` | — | Path to `build/libmk_release.so`. Required. |
| `--ckpt` | — | North Mini Code checkpoint directory. Required. |
| `--device` | `cuda` | e.g. `cuda:0`. |
| `--bs` | `4` | Max batch size; one of 1, 2, 4, 8. |
| `--frac-vram-utilization` | `0.90` | Fraction of VRAM for weights + KV pool. |
| `--max-context` | `262144` | Per-request context cap. |
| `--prefill-chunk-size` | `1024` | Tokens per prefill chunk. Larger chunks (e.g. `4096`) speed up long prompts. |
| `--num-sms` | `132` | SMs the megakernel occupies. |
| `--temperature` | `0.0` | Server-side default sampling temperature. (Can be overridden by the client.) |
| `--batched-prefill` / `--no-batched-prefill` | on | Coalesce multiple pending prefills. |

### Verify with curl

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "North-Mini-Code-1.0",
    "messages": [{"role": "user", "content": "Write a Python Fibonacci function."}],
    "max_tokens": 128,
    "temperature": 0
  }'
```

The server also exposes `POST /v1/completions` and `GET /v1/models`.

### OpenCode

Add an OpenAI-compatible provider to `opencode.json`:

```json
{
  "provider": {
    "megakernel": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Megakernel",
      "options": {
        "baseURL": "http://127.0.0.1:8000/v1",
        "apiKey": "EMPTY"
      },
      "models": {
        "North-Mini-Code-1.0": {
          "name": "North-Mini-Code-1.0 (MK)"
        }
      }
    }
  }
}
```

## Standalone runner

`src/runner.py` supports real prompts, isolated decode benchmarks, ragged
batches, and Perfetto traces. Use it when you want to run without starting
the HTTP server.

`--checkpoint` is required for real prompts and `--real-weight`.
Synthetic `--fast` (no `--real-weight`) uses the baked-in North Mini
Code geometry and does not need a checkpoint.

### Real prompt

```bash
python src/runner.py \
  --mk \
  --checkpoint <checkpoint-path> \
  --lib-path "$PWD/build/libmk_release.so" \
  --msgs "Introduce yourself." \
  --max-new-tokens 256 \
  --cpp-decode-runtime
```

Multiple `--msgs` values form a ragged real-prompt batch:

```bash
python src/runner.py \
  --mk \
  --checkpoint <checkpoint-path> \
  --lib-path "$PWD/build/libmk_release.so" \
  --msgs "Introduce yourself." "What is 1+1?" "Write a factorial function." "Tell me something about LLMs." \
  --max-new-tokens 256 \
  --cpp-decode-runtime
```

### Benchmark

The decode-only benchmark measures throughput and TPOT for a given
input length and batch size. Two setups, matching the two regimes reported
in the README:

**Uniform routing** — `--fast` skips checkpoint loading, uses synthetic
weights, and fills the KV cache with `N(0, 0.01)` noise. Random weights give
approximately uniform expert routing. Compare against vLLM with simulated
uniform routing; both engines skip some real work, but the comparison is
apples-to-apples in workload shape. Output tokens are suppressed because
they are not meaningful model output.

```bash
run_benchmark() {
  local INLEN=$1 BS=$2 OUTLEN=$3
  python src/runner.py --mk --fast \
    --lib-path "$PWD/build/libmk_release.so" \
    --batch-size "$BS" --max-new-tokens "$OUTLEN" \
    --fake-prompt-len "$INLEN" \
    --num-sms 132 --frac-vram-utilization 0.9 \
    --cpp-decode-runtime | grep metrics
}

run_benchmark 4096 1 512
```
See also `tools/print_perf_table.py` to print the results into a markdown table.

vLLM baseline:

```bash
VLLM_MOE_ROUTING_SIMULATION_STRATEGY=uniform_random vllm serve North-Mini-Code-1.0 \
  --port 8000 \
  --max-model-len 320000 \
  --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector": "DecodeBenchConnector", "kv_role": "kv_both", "kv_connector_extra_config": {"fill_mean": 0.0, "fill_std": 0.01}}'

# then run vllm bench serve against it
```

**Real expert traffic** — add `--real-weight --checkpoint <checkpoint-path>` to
keep the synthetic KV cache but use real weights and the checkpoint's
naturally correlated expert routing. 

```bash
python src/runner.py --mk --fast --real-weight \
  --checkpoint <checkpoint-path> \
  --lib-path "$PWD/build/libmk_release.so" \
  --batch-size 1 --max-new-tokens 512 \
  --fake-prompt-len 8192 \
  --num-sms 132 --frac-vram-utilization 0.9 \
  --cpp-decode-runtime | grep metrics
```

vLLM baseline (same synthetic KV cache, no routing simulation):

```bash
vllm serve North-Mini-Code-1.0 \
  --port 8000 \
  --max-model-len 320000 \
  --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector": "DecodeBenchConnector", "kv_role": "kv_both", "kv_connector_extra_config": {"fill_mean": 0.0, "fill_std": 0.01}}'
```

Expected ballpark on a single H100 (real checkpoint, 8K context): BS=1
≈292 tok/s, BS=8 ≈1,000 tok/s. See the full benchmark tables in
[README.md — Performance](README.md#performance) and the raw 1K–256K
context-length sweep in the [blog](https://cohere.com/blog/megakernels).

Synthetic ragged sequence lengths:

```bash
python src/runner.py \
  --mk --fast \
  --lib-path "$PWD/build/libmk_release.so" \
  --fake-prompt-len 8192 \
  --batch-size 8 \
  --ragged-batch --seqlen-variance 4096 \
  --max-new-tokens 512 \
  --cpp-decode-runtime
```

### Profiling

The Python decode loop can emit a [Perfetto](https://ui.perfetto.dev) trace
through `sm-profiler`. Profiling is incompatible with `--cpp-decode-runtime`; 
use Python decode loop when profiling.

```bash
mkdir -p dump
python src/runner.py \
  --mk --fast \
  --lib-path "$PWD/build/libmk_release.so" \
  --fake-prompt-len 4096 \
  --batch-size 1 \
  --max-new-tokens 512 \
  --profile dump/profile-seq4k-bs1.json
```

Open the resulting JSON at <https://ui.perfetto.dev>.

## Known limitations

- The megakernel covers decode only; prefill uses regular PyTorch kernels.
- Prefill pauses active decode requests; mixed prefill/decode is not
  supported yet. For workloads of many very short requests this hurts
  throughput.
- The build targets H100/SM90a and BF16 only.
- The server accepts batch sizes 1-8.
- Sampling: greedy or temperature. `top_p` must be `1.0`, `top_k` must be
  `1` or unset, and `n` must be `1`.
- The implementation is model-specific and requires a local
  North Mini Code checkpoint.
