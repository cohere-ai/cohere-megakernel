#include "jit.hpp"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <regex>
#include <fstream>
#include <stdexcept>
#include <filesystem>
#include <vector>
#include <string>
#include <sstream>
#include <thread>

#include <dlfcn.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <openssl/evp.h>

namespace fs = std::filesystem;

namespace little_jit {

static bool is_valid_elf_shared_object(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    unsigned char magic[4] = {0, 0, 0, 0};
    in.read(reinterpret_cast<char*>(magic), sizeof(magic));
    if (in.gcount() != static_cast<std::streamsize>(sizeof(magic))) return false;
    return magic[0] == 0x7f && magic[1] == 'E' && magic[2] == 'L' && magic[3] == 'F';
}

static fs::path make_temp_so_path(const fs::path& cache_dir, const std::string& hash) {
    std::ostringstream tid;
    tid << std::this_thread::get_id();
    return cache_dir /
        ("lib" + hash + ".tmp." + std::to_string(getpid()) + "." + tid.str() + ".so");
}

static fs::path make_ptxas_log_path(const fs::path& cache_dir, const std::string& hash) {
    return cache_dir / (hash + ".ptxas.txt");
}

// ─── SHA-256 via OpenSSL EVP ─────────────────────────────────────────────────

static std::string sha256_hex(const std::string& data) {
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (!ctx) throw std::runtime_error("EVP_MD_CTX_new failed");
    if (EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) != 1 ||
        EVP_DigestUpdate(ctx, data.data(), data.size()) != 1) {
        EVP_MD_CTX_free(ctx);
        throw std::runtime_error("EVP_DigestUpdate failed");
    }
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int  digest_len = 0;
    if (EVP_DigestFinal_ex(ctx, digest, &digest_len) != 1) {
        EVP_MD_CTX_free(ctx);
        throw std::runtime_error("EVP_DigestFinal_ex failed");
    }
    EVP_MD_CTX_free(ctx);
    char hex[65];
    for (unsigned int i = 0; i < digest_len; ++i)
        snprintf(hex + 2 * i, 3, "%02x", digest[i]);
    return std::string(hex, 64);
}

// ─── Locate NVCC ─────────────────────────────────────────────────────────────

static std::string find_nvcc() {
    for (const char* var : {"NVCC", "CUDACXX"}) {
        const char* v = std::getenv(var);
        if (v && v[0] != '\0') return v;
    }
    return "nvcc";
}

// ─── Run a subprocess, throw on nonzero exit ─────────────────────────────────

static std::string join_argv(const std::vector<std::string>& argv_str) {
    std::string cmd;
    for (const auto& s : argv_str) {
        cmd += s;
        cmd += ' ';
    }
    return cmd;
}

static void emit_ptxas_diagnostics(const std::string& output,
                                   const std::string& tag) {
    static const std::regex ptxas_diagnostic_re(R"((^|\n)(ptxas (warning|info|error)[^\n]*)(?=\n|$))");

    bool emitted = false;

    for (std::sregex_iterator it(output.begin(), output.end(), ptxas_diagnostic_re), end;
         it != end; ++it) {
        if (!emitted) {
            std::fprintf(stderr, "[little_jit][ptxas] %s\n", tag.c_str());
            emitted = true;
        }
        std::fprintf(stderr, "  %s\n", (*it)[2].str().c_str());
    }
}

static std::string run_argv(const std::vector<std::string>& argv_str,
                            const std::string& tag) {
    std::vector<const char*> argv;
    argv.reserve(argv_str.size() + 1);
    for (const auto& s : argv_str) argv.push_back(s.c_str());
    argv.push_back(nullptr);

    int pipefd[2];
    if (pipe(pipefd) != 0)
        throw std::runtime_error("pipe() failed");

    pid_t pid = fork();
    if (pid < 0) {
        close(pipefd[0]);
        close(pipefd[1]);
        throw std::runtime_error("fork() failed");
    }
    if (pid == 0) {
        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);
        dup2(pipefd[1], STDERR_FILENO);
        close(pipefd[1]);
        execvp(argv[0], const_cast<char* const*>(argv.data()));
        // exec failed
        std::fprintf(stderr, "[little_jit] execvp(%s) failed: %s\n",
                     argv[0], std::strerror(errno));
        _exit(127);
    }

    close(pipefd[1]);
    std::string output;
    char buffer[4096];
    while (true) {
        ssize_t n = read(pipefd[0], buffer, sizeof(buffer));
        if (n < 0) {
            if (errno == EINTR) continue;
            const int read_error = errno;
            close(pipefd[0]);
            kill(pid, SIGKILL);
            while (waitpid(pid, nullptr, 0) < 0 && errno == EINTR) {
            }
            throw std::runtime_error(
                std::string("read() failed: ") + std::strerror(read_error));
        }
        if (n == 0) break;
        output.append(buffer, static_cast<size_t>(n));
    }
    close(pipefd[0]);

    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno == EINTR) continue;
        throw std::runtime_error(
            std::string("waitpid() failed: ") + std::strerror(errno));
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        std::string err =
            "[little_jit] NVCC failed (exit " +
            std::to_string(WIFEXITED(status) ? WEXITSTATUS(status) : -1) +
            "): " + join_argv(argv_str);
        if (!output.empty()) err += "\n" + output;
        throw std::runtime_error(err);
    }
    emit_ptxas_diagnostics(output, tag);
    return output;
}

// ─── jit_compile ─────────────────────────────────────────────────────────────

std::string jit_compile(const std::string& code,
    const std::vector<std::string>& source_files,
    const std::vector<std::string>& include_paths,
    const std::vector<std::string>& defines,
    const std::vector<std::string>& options,
    const std::vector<std::string>& flags,
    const std::vector<std::string>& hash_files)
{
    std::vector<std::string> effective_options = options;
    bool has_ptxas_verbose = false;
    for (const auto& o : effective_options) {
        if (o.find("-Xptxas") != std::string::npos &&
            o.find("-v") != std::string::npos) {
            has_ptxas_verbose = true;
            break;
        }
    }
    if (!has_ptxas_verbose) effective_options.push_back("-Xptxas=-v");

    const std::string nvcc = find_nvcc();
    const std::string nvcc_identity = run_argv(
        {nvcc, "--version"},
        "nvcc-version");

    // Build the hash input: accumulate all inputs into one string.
    std::string hash_input;
    hash_input += nvcc;
    hash_input += '\0';
    hash_input += nvcc_identity;
    hash_input += '\0';
    hash_input += code;
    hash_input += '\0';
    for (const auto& sf : source_files) {
        hash_input += sf;
        hash_input += '\0';
        // include file contents so a content change triggers recompile
        std::ifstream f(sf, std::ios::binary);
        if (f) {
            hash_input += std::string(std::istreambuf_iterator<char>(f),
                                      std::istreambuf_iterator<char>());
        }
        hash_input += '\0';
    }
    for (const auto& hf : hash_files) {
        hash_input += hf;
        hash_input += '\0';
        std::ifstream f(hf, std::ios::binary);
        if (f) {
            hash_input += std::string(std::istreambuf_iterator<char>(f),
                                      std::istreambuf_iterator<char>());
        }
        hash_input += '\0';
    }
    for (const auto& p : include_paths) { hash_input += p; hash_input += '\0'; }
    for (const auto& d : defines)       { hash_input += d; hash_input += '\0'; }
    for (const auto& o : effective_options) { hash_input += o; hash_input += '\0'; }
    for (const auto& f : flags)         { hash_input += f; hash_input += '\0'; }

    const std::string hash = sha256_hex(hash_input);

    // Resolve cache directory.
    fs::path cache_dir;
    if (const char* env = std::getenv("LITTLE_JIT_CACHE_DIR")) {
        cache_dir = env;
    } else {
        cache_dir = fs::temp_directory_path() / "little_jit";
    }
    fs::create_directories(cache_dir);

    const fs::path so_path  = cache_dir / ("lib" + hash + ".so");
    const fs::path cu_path  = cache_dir / (hash + ".cu");
    const fs::path tmp_so_path = make_temp_so_path(cache_dir, hash);

    // Cache hit — return immediately if the cached artifact is a plausible ELF.
    // Older interrupted runs could leave behind a zero-byte or otherwise corrupt
    // .so; rebuild those instead of surfacing dlopen() failures later.
    if (fs::exists(so_path)) {
        if (is_valid_elf_shared_object(so_path)) return so_path.string();
        std::error_code ec;
        fs::remove(so_path, ec);
    }

    // Write generated source.
    {
        std::ofstream out(cu_path, std::ios::binary);
        if (!out) throw std::runtime_error("[little_jit] cannot write " + cu_path.string());
        out << code;
    }

    // Build argv for NVCC.
    std::vector<std::string> argv;
    argv.push_back(nvcc);
    argv.push_back("-shared");
    argv.push_back("-Xcompiler");
    argv.push_back("-fPIC");
    argv.push_back("-std=c++20");
    argv.push_back("-o");
    argv.push_back(tmp_so_path.string());
    argv.push_back(cu_path.string());
    for (const auto& sf : source_files) argv.push_back(sf);
    for (const auto& ip : include_paths) argv.push_back("-I" + ip);
    for (const auto& d  : defines)       argv.push_back("-D" + d);
    for (const auto& o  : effective_options) argv.push_back(o);
    for (const auto& f  : flags)         argv.push_back(f);

    std::string nvcc_output;
    try {
        nvcc_output = run_argv(argv, cu_path.filename().string());
    } catch (...) {
        std::error_code ec;
        fs::remove(tmp_so_path, ec);
        throw;
    }
    {
        std::ofstream out(make_ptxas_log_path(cache_dir, hash), std::ios::binary);
        if (out) out << nvcc_output;
    }
    if (!is_valid_elf_shared_object(tmp_so_path)) {
        std::error_code ec;
        fs::remove(tmp_so_path, ec);
        throw std::runtime_error("[little_jit] NVCC produced an invalid shared object: " +
                                 tmp_so_path.string());
    }
    {
        std::error_code ec;
        fs::rename(tmp_so_path, so_path, ec);
        if (ec) {
            // Another worker may have won the race to publish the same hash.
            if (fs::exists(so_path) && is_valid_elf_shared_object(so_path)) {
                fs::remove(tmp_so_path, ec);
            } else {
                fs::remove(tmp_so_path, ec);
                throw std::runtime_error("[little_jit] failed to publish cache artifact " +
                                         so_path.string() + ": " + ec.message());
            }
        }
    }
    return so_path.string();
}

// ─── jit_function move / destructor ──────────────────────────────────────────

jit_function::jit_function(jit_function&& o) noexcept
    : function(o.function), module(o.module), hash(std::move(o.hash))
{
    o.function = nullptr;
    o.module   = nullptr;
}

jit_function& jit_function::operator=(jit_function&& o) noexcept {
    if (this != &o) {
        if (module) dlclose(module);
        function = o.function; o.function = nullptr;
        module   = o.module;   o.module   = nullptr;
        hash     = std::move(o.hash);
    }
    return *this;
}

jit_function::~jit_function() {
    if (module) { dlclose(module); module = nullptr; }
}

// ─── load_function ───────────────────────────────────────────────────────────

jit_function load_function(const std::string& path, const std::string& symbol) {
    dlerror(); // clear any previous error
    // RTLD_GLOBAL exposes this module's symbols to subsequently loaded JIT
    // modules. Generated exported symbol names must therefore be unique.
    void* mod = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!mod) {
        throw std::runtime_error("[little_jit] dlopen(" + path + ") failed: " +
                                 std::string(dlerror()));
    }
    dlerror();
    void* sym = dlsym(mod, symbol.c_str());
    const char* err = dlerror();
    if (err) {
        dlclose(mod);
        throw std::runtime_error("[little_jit] dlsym(" + symbol + ") failed: " +
                                 std::string(err));
    }
    jit_function jf;
    jf.function = sym;
    jf.module   = mod;
    // Extract hash from filename: lib<hash>.so
    fs::path p(path);
    std::string stem = p.stem().string(); // "lib<hash>"
    if (stem.size() > 3 && stem.substr(0, 3) == "lib")
        jf.hash = stem.substr(3);
    else
        jf.hash = stem;
    return jf;
}

} // namespace little_jit
