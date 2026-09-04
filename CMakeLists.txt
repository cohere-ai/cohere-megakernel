cmake_minimum_required(VERSION 3.24)
project(megakernel-nmc LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_ARCHITECTURES 90a)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)

find_package(CUDAToolkit REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(Threads REQUIRED)
if(NOT TARGET OpenSSL::Crypto AND OPENSSL_CRYPTO_LIBRARY)
    add_library(OpenSSL::Crypto UNKNOWN IMPORTED)
    set_target_properties(OpenSSL::Crypto PROPERTIES
        IMPORTED_LOCATION "${OPENSSL_CRYPTO_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${OPENSSL_INCLUDE_DIR}")
endif()

# ── Python + nanobind (vendored) ────────────────────────────────────────────
# Development.Module is headers-only and never links libpython on Linux, which
# is what keeps this build from being tied to whichever venv happens to be
# active. Development.SABIModule is what enables the STABLE_ABI build below.
#
# Select the interpreter explicitly when configuring, e.g.
#   cmake -S . -B build -G Ninja -DPython_EXECUTABLE=<path to python>
# Without it CMake picks up whatever `python3` is first on PATH, and the
# resulting extension may not be importable from the venv you actually run.
find_package(Python 3.12 REQUIRED
    COMPONENTS Interpreter Development.Module
    OPTIONAL_COMPONENTS Development.SABIModule)

# nanobind comes from ext/nanobind, NOT from the venv's site-packages, so no
# `python -m nanobind --cmake_dir` / CMAKE_PREFIX_PATH dance is needed. It has
# its own submodule (robin_map): `git submodule update --init --recursive`.
add_subdirectory(ext/nanobind)



# ── Self-contained release bundle ───────────────────────────────────────
add_library(little_jit_release STATIC src/jit.cpp)
target_include_directories(little_jit_release PUBLIC src)
target_link_libraries(little_jit_release PUBLIC OpenSSL::Crypto ${CMAKE_DL_LIBS})
target_compile_features(little_jit_release PUBLIC cxx_std_20)
set_target_properties(little_jit_release PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(mk_release SHARED
    src/decode/runtime.cu
    src/kv/cache.cpp
    src/sm-profiler/sm_profiler.cu
)
target_include_directories(mk_release PRIVATE
    ext/ThunderKittens/include
    ext/ThunderKittens/prototype
    src
    src/sm-profiler
)
target_compile_definitions(mk_release PRIVATE KITTENS_HOPPER)
target_link_libraries(mk_release PRIVATE
    CUDA::cuda_driver
    CUDA::nvtx3
    little_jit_release
)
target_compile_options(mk_release PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --expt-extended-lambda -O3 -lineinfo --use_fast_math --ptxas-options=-v,-warn-spills>
)
set_target_properties(mk_release PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION OFF
)
# ─────────────────────────────────────────────────────────────────────────────

# ── nanobind extension module ───────────────────────────────────────────────
# Bindings live in their own translation unit and only include headers, so
# editing them never triggers a recompile of the (very slow) src/decode/runtime.cu.
#
# Nanobind is the only front-end, so a static link would be safe on those
# grounds alone. 
#
# STABLE_ABI produces `mk_ext.abi3.so`, usable from any CPython >= 3.12 rather
# than only the exact minor version it was compiled against. It costs a little
# per-call performance (nanobind loses access to CPython internals); revisit if
# the decode hot path ever shows it.
nanobind_add_module(mk_ext
    STABLE_ABI
    NB_STATIC
    src/bindings/mk_ext.cpp
    src/bindings/kv.cpp
    src/bindings/launch.cpp
    src/bindings/service.cpp
)
target_include_directories(mk_ext PRIVATE src)
target_link_libraries(mk_ext PRIVATE mk_release)
# $ORIGIN, not an absolute path: mk_ext.abi3.so and libmk_release.so are both
# emitted into the build directory and must stay side by side. Copying only one
# of them elsewhere will fail at import time, not at build time.
set_target_properties(mk_ext PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
)
# ─────────────────────────────────────────────────────────────────────────────
