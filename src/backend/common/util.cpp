/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

/// This file contains platform independent utility functions
#if defined(OS_WIN)
#include <Windows.h>
#else
#include <pwd.h>
#include <unistd.h>
#endif

#include <common/Logger.hpp>
#include <common/TemplateArg.hpp>
#include <common/defines.hpp>
#include <common/util.hpp>
#include <optypes.hpp>
#include <af/defines.h>

#include <nonstd/span.hpp>
#include <sys/stat.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef __has_include
#if __has_include(<charconv>)
#include <charconv>
#endif
#if __has_include(<version>)
#include <version>
#endif
#endif

using nonstd::span;
using std::accumulate;
using std::array;
using std::hash;
using std::ofstream;
using std::once_flag;
using std::rename;
using std::size_t;
using std::string;
using std::stringstream;
using std::thread;
using std::to_string;
using std::uint8_t;
using std::vector;

namespace arrayfire {
namespace common {
// http://stackoverflow.com/questions/216823/whats-the-best-way-to-trim-stdstring/217605#217605
// trim from start
string& ltrim(string& s) {
    s.erase(s.begin(),
            find_if(s.begin(), s.end(), [](char c) { return !isspace(c); }));
    return s;
}

string getEnvVar(const string& key) {
#if defined(OS_WIN)
    DWORD bufSize =
        32767;  // limit according to GetEnvironment Variable documentation
    string retVal;
    retVal.resize(bufSize);
    bufSize = GetEnvironmentVariable(key.c_str(), &retVal[0], bufSize);
    if (!bufSize) {
        return string("");
    } else {
        retVal.resize(bufSize);
        return retVal;
    }
#else
    char* str = getenv(key.c_str());
    return str == NULL ? string("") : string(str);
#endif
}

const char* getName(af_dtype type) {
    switch (type) {
        case f32: return "float";
        case f64: return "double";
        case c32: return "complex float";
        case c64: return "complex double";
        case u32: return "unsigned int";
        case s32: return "int";
        case u16: return "unsigned short";
        case s16: return "short";
        case u64: return "unsigned long long";
        case s64: return "long long";
        case u8: return "unsigned char";
        case b8: return "bool";
        default: return "unknown type";
    }
}

void saveKernel(const string& funcName, const string& jit_ker,
                const string& ext) {
    static constexpr const char* saveJitKernelsEnvVarName =
        "AF_JIT_KERNEL_TRACE";
    static const char* jitKernelsOutput = getenv(saveJitKernelsEnvVarName);
    if (!jitKernelsOutput) { return; }
    if (strcmp(jitKernelsOutput, "stdout") == 0) {
        fputs(jit_ker.c_str(), stdout);
        return;
    }
    if (strcmp(jitKernelsOutput, "stderr") == 0) {
        fputs(jit_ker.c_str(), stderr);
        return;
    }
    // Path to a folder
    const string ffp =
        string(jitKernelsOutput) + AF_PATH_SEPARATOR + funcName + ext;

#if defined(OS_WIN)
    FILE* f = fopen(ffp.c_str(), "w");
#else
    FILE* f = fopen(ffp.c_str(), "we");
#endif

    if (!f) {
        fprintf(stderr, "Cannot open file %s\n", ffp.c_str());
        return;
    }
    if (fputs(jit_ker.c_str(), f) == EOF) {
        fprintf(stderr, "Failed to write kernel to file %s\n", ffp.c_str());
    }
    fclose(f);
}

#if defined(OS_WIN)
string getTemporaryDirectory() {
    DWORD bufSize = 261;  // limit according to GetTempPath documentation
    string retVal;
    retVal.resize(bufSize);
    bufSize = GetTempPathA(bufSize, &retVal[0]);
    retVal.resize(bufSize);
    return retVal;
}
#else
string getHomeDirectory() {
    string home = getEnvVar("XDG_CACHE_HOME");
    if (!home.empty()) { return home; }

    home = getEnvVar("HOME");
    if (!home.empty()) { return home; }

    return getpwuid(getuid())->pw_dir;
}
#endif

bool directoryExists(const string& path) {
#if defined(OS_WIN)
    struct _stat status;
    return _stat(path.c_str(), &status) == 0 && (status.st_mode & S_IFDIR) != 0;
#else
    struct stat status {};
    // NOLINTNEXTLINE(hicpp-signed-bitwise)
    return stat(path.c_str(), &status) == 0 && (status.st_mode & S_IFDIR) != 0;
#endif
}

bool createDirectory(const string& path) {
#if defined(OS_WIN)
    return CreateDirectoryA(path.c_str(), NULL) != 0;
#else
    return mkdir(path.c_str(), 0777) == 0;
#endif
}

bool removeFile(const string& path) {
#if defined(OS_WIN)
    return DeleteFileA(path.c_str()) != 0;
#else
    return unlink(path.c_str()) == 0;
#endif
}

bool renameFile(const string& sourcePath, const string& destPath) {
    return rename(sourcePath.c_str(), destPath.c_str()) == 0;
}

bool isDirectoryWritable(const string& path) {
    if (!directoryExists(path) && !createDirectory(path)) { return false; }

    const string testPath = path + AF_PATH_SEPARATOR + "test";
    if (!ofstream(testPath).is_open()) { return false; }
    removeFile(testPath);

    return true;
}

#ifndef NOSPDLOG
string& getCacheDirectory() {
    static once_flag flag;
    static string cacheDirectory;

    call_once(flag, []() {
        string pathList[] = {
#if defined(OS_WIN)
            getTemporaryDirectory() + "\\ArrayFire"
#else
            getHomeDirectory() + "/.arrayfire",
            "/tmp/arrayfire"
#endif
        };

        auto env_path = getEnvVar(JIT_KERNEL_CACHE_DIRECTORY_ENV_NAME);
        if (!env_path.empty() && !isDirectoryWritable(env_path)) {
            spdlog::get("platform")
                ->warn(
                    "The environment variable {}({}) is "
                    "not writeable. Falling back to default.",
                    JIT_KERNEL_CACHE_DIRECTORY_ENV_NAME, env_path);
            env_path.clear();
        }

        if (env_path.empty()) {
            auto iterDir =
                find_if(begin(pathList), end(pathList), isDirectoryWritable);

            cacheDirectory = iterDir != end(pathList) ? *iterDir : "";
        } else {
            cacheDirectory = env_path;
        }
    });

    return cacheDirectory;
}
#endif

string makeTempFilename() {
    thread_local size_t fileCount = 0u;

    ++fileCount;
    const size_t threadID = hash<thread::id>{}(std::this_thread::get_id());

    return to_string(
        hash<string>{}(to_string(threadID) + "_" + to_string(fileCount)));
}

template<typename T>
string toString(T value) {
#ifdef __cpp_lib_to_chars
    array<char, 128> out;
    if (auto [ptr, ec] = std::to_chars(out.data(), out.data() + 128, value);
        ec == std::errc()) {
        return string(out.data(), ptr);
    } else {
        return string("#error invalid conversion");
    }
#else
    stringstream ss;
    ss.imbue(std::locale::classic());
    ss << value;
    return ss.str();
#endif
}

template string toString<int>(int);
template string toString<unsigned short>(unsigned short);
template string toString<short>(short);
template string toString<unsigned char>(unsigned char);
template string toString<char>(char);
template string toString<long>(long);
template string toString<long long>(long long);
template string toString<unsigned>(unsigned);
template string toString<unsigned long>(unsigned long);
template string toString<unsigned long long>(unsigned long long);
template string toString<float>(float);
template string toString<double>(double);
template string toString<long double>(long double);

template<>
string toString(TemplateArg arg) {
    return arg._tparam;
}

template<>
string toString(bool val) {
    return string(val ? "true" : "false");
}

template<>
string toString(const char* str) {
    return string(str);
}

template<>
string toString(const string str) {
    return str;
}

template<>
string toString(af_op_t val) {
    const char* retVal = NULL;
#define CASE_STMT(v) \
    case v: retVal = #v; break
    switch (val) {
        CASE_STMT(af_add_t);
        CASE_STMT(af_sub_t);
        CASE_STMT(af_mul_t);
        CASE_STMT(af_div_t);

        CASE_STMT(af_and_t);
        CASE_STMT(af_or_t);
        CASE_STMT(af_eq_t);
        CASE_STMT(af_neq_t);
        CASE_STMT(af_lt_t);
        CASE_STMT(af_le_t);
        CASE_STMT(af_gt_t);
        CASE_STMT(af_ge_t);

        CASE_STMT(af_bitnot_t);
        CASE_STMT(af_bitor_t);
        CASE_STMT(af_bitand_t);
        CASE_STMT(af_bitxor_t);
        CASE_STMT(af_bitshiftl_t);
        CASE_STMT(af_bitshiftr_t);

        CASE_STMT(af_min_t);
        CASE_STMT(af_max_t);
        CASE_STMT(af_cplx2_t);
        CASE_STMT(af_atan2_t);
        CASE_STMT(af_pow_t);
        CASE_STMT(af_hypot_t);

        CASE_STMT(af_sin_t);
        CASE_STMT(af_cos_t);
        CASE_STMT(af_tan_t);
        CASE_STMT(af_asin_t);
        CASE_STMT(af_acos_t);
        CASE_STMT(af_atan_t);

        CASE_STMT(af_sinh_t);
        CASE_STMT(af_cosh_t);
        CASE_STMT(af_tanh_t);
        CASE_STMT(af_asinh_t);
        CASE_STMT(af_acosh_t);
        CASE_STMT(af_atanh_t);

        CASE_STMT(af_exp_t);
        CASE_STMT(af_expm1_t);
        CASE_STMT(af_erf_t);
        CASE_STMT(af_erfc_t);

        CASE_STMT(af_log_t);
        CASE_STMT(af_log10_t);
        CASE_STMT(af_log1p_t);
        CASE_STMT(af_log2_t);

        CASE_STMT(af_sqrt_t);
        CASE_STMT(af_cbrt_t);

        CASE_STMT(af_abs_t);
        CASE_STMT(af_cast_t);
        CASE_STMT(af_cplx_t);
        CASE_STMT(af_real_t);
        CASE_STMT(af_imag_t);
        CASE_STMT(af_conj_t);

        CASE_STMT(af_floor_t);
        CASE_STMT(af_ceil_t);
        CASE_STMT(af_round_t);
        CASE_STMT(af_trunc_t);
        CASE_STMT(af_signbit_t);

        CASE_STMT(af_rem_t);
        CASE_STMT(af_mod_t);

        CASE_STMT(af_tgamma_t);
        CASE_STMT(af_lgamma_t);

        CASE_STMT(af_notzero_t);

        CASE_STMT(af_iszero_t);
        CASE_STMT(af_isinf_t);
        CASE_STMT(af_isnan_t);

        CASE_STMT(af_sigmoid_t);

        CASE_STMT(af_noop_t);

        CASE_STMT(af_select_t);
        CASE_STMT(af_not_select_t);
        CASE_STMT(af_rsqrt_t);
        CASE_STMT(af_moddims_t);

        CASE_STMT(af_none_t);
    }
#undef CASE_STMT
    return retVal;
}

template<>
string toString(af_interp_type p) {
    const char* retVal = NULL;
#define CASE_STMT(v) \
    case v: retVal = #v; break
    switch (p) {
        CASE_STMT(AF_INTERP_NEAREST);
        CASE_STMT(AF_INTERP_LINEAR);
        CASE_STMT(AF_INTERP_BILINEAR);
        CASE_STMT(AF_INTERP_CUBIC);
        CASE_STMT(AF_INTERP_LOWER);
        CASE_STMT(AF_INTERP_LINEAR_COSINE);
        CASE_STMT(AF_INTERP_BILINEAR_COSINE);
        CASE_STMT(AF_INTERP_BICUBIC);
        CASE_STMT(AF_INTERP_CUBIC_SPLINE);
        CASE_STMT(AF_INTERP_BICUBIC_SPLINE);
    }
#undef CASE_STMT
    return retVal;
}

template<>
string toString(af_border_type p) {
    const char* retVal = NULL;
#define CASE_STMT(v) \
    case v: retVal = #v; break
    switch (p) {
        CASE_STMT(AF_PAD_ZERO);
        CASE_STMT(AF_PAD_SYM);
        CASE_STMT(AF_PAD_CLAMP_TO_EDGE);
        CASE_STMT(AF_PAD_PERIODIC);
    }
#undef CASE_STMT
    return retVal;
}

template<>
string toString(af_moment_type p) {
    const char* retVal = NULL;
#define CASE_STMT(v) \
    case v: retVal = #v; break
    switch (p) {
        CASE_STMT(AF_MOMENT_M00);
        CASE_STMT(AF_MOMENT_M01);
        CASE_STMT(AF_MOMENT_M10);
        CASE_STMT(AF_MOMENT_M11);
        CASE_STMT(AF_MOMENT_FIRST_ORDER);
    }
#undef CASE_STMT
    return retVal;
}

template<>
string toString(af_match_type p) {
    const char* retVal = NULL;
#define CASE_STMT(v) \
    case v: retVal = #v; break
    switch (p) {
        CASE_STMT(AF_SAD);
        CASE_STMT(AF_ZSAD);
        CASE_STMT(AF_LSAD);
        CASE_STMT(AF_SSD);
        CASE_STMT(AF_ZSSD);
        CASE_STMT(AF_LSSD);
        CASE_STMT(AF_NCC);
        CASE_STMT(AF_ZNCC);
        CASE_STMT(AF_SHD);
    }
#undef CASE_STMT
    return retVal;
}

template<>
string toString(af_flux_function p) {
    const char* retVal = NULL;
#define CASE_STMT(v) \
    case v: retVal = #v; break
    switch (p) {
        CASE_STMT(AF_FLUX_QUADRATIC);
        CASE_STMT(AF_FLUX_EXPONENTIAL);
        CASE_STMT(AF_FLUX_DEFAULT);
    }
#undef CASE_STMT
    return retVal;
}

template<>
string toString(AF_BATCH_KIND val) {
    const char* retVal = NULL;
#define CASE_STMT(v) \
    case v: retVal = #v; break
    switch (val) {
        CASE_STMT(AF_BATCH_NONE);
        CASE_STMT(AF_BATCH_LHS);
        CASE_STMT(AF_BATCH_RHS);
        CASE_STMT(AF_BATCH_SAME);
        CASE_STMT(AF_BATCH_DIFF);
        CASE_STMT(AF_BATCH_UNSUPPORTED);
    }
#undef CASE_STMT
    return retVal;
}

template<>
string toString(af_homography_type val) {
    const char* retVal = NULL;
#define CASE_STMT(v) \
    case v: retVal = #v; break
    switch (val) {
        CASE_STMT(AF_HOMOGRAPHY_RANSAC);
        CASE_STMT(AF_HOMOGRAPHY_LMEDS);
    }
#undef CASE_STMT
    return retVal;
}

}  // namespace common
}  // namespace arrayfire
