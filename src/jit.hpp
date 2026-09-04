#pragma once
#include <string>
#include <vector>
namespace little_jit {
    // Compiles the code string and source files into a shared library using NVCC.
    // hash_files are included in the cache key only; they are not compiled.
    // A SHA-256 hash is computed over all inputs plus the selected NVCC identity;
    // the resulting .so is named after that hash and cached in
    // LITTLE_JIT_CACHE_DIR (or /tmp/little_jit).
    // Returns the absolute path to the compiled shared library.
    // Loaded modules use RTLD_GLOBAL; generated exported symbol names should be
    // unique when multiple kernels can coexist in one process.
    std::string jit_compile(const std::string& code,
        const std::vector<std::string>& source_files,
        const std::vector<std::string>& include_paths,
        const std::vector<std::string>& defines,
        const std::vector<std::string>& options,
        const std::vector<std::string>& flags,
        const std::vector<std::string>& hash_files = {});

    struct jit_function {
        void* function = nullptr;  // resolved symbol pointer
        void* module   = nullptr;  // dlopen handle
        std::string hash;

        jit_function() = default;
        jit_function(const jit_function&) = delete;
        jit_function& operator=(const jit_function&) = delete;
        jit_function(jit_function&& o) noexcept;
        jit_function& operator=(jit_function&& o) noexcept;
        ~jit_function();
    };

    // Opens the shared library at path and resolves symbol, returning a jit_function.
    // Throws std::runtime_error on dlopen or dlsym failure.
    jit_function load_function(const std::string& path, const std::string& symbol);
}
