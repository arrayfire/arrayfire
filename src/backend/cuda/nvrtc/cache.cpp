/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <nvrtc/cache.hpp>

#include <common/Logger.hpp>
#include <common/internal_enums.hpp>
#include <common/util.hpp>
#include <device_manager.hpp>
#include <kernel_headers/jit_cuh.hpp>
#include <nvrtc_kernel_headers/Param_hpp.hpp>
#include <nvrtc_kernel_headers/assign_kernel_param_hpp.hpp>
#include <nvrtc_kernel_headers/backend_hpp.hpp>
#include <nvrtc_kernel_headers/cuComplex_h.hpp>
#include <nvrtc_kernel_headers/cuda_fp16_h.hpp>
#include <nvrtc_kernel_headers/cuda_fp16_hpp.hpp>
#include <nvrtc_kernel_headers/defines_h.hpp>
#include <nvrtc_kernel_headers/dims_param_hpp.hpp>
#include <nvrtc_kernel_headers/half_hpp.hpp>
#include <nvrtc_kernel_headers/internal_enums_hpp.hpp>
#include <nvrtc_kernel_headers/interp_hpp.hpp>
#include <nvrtc_kernel_headers/kernel_type_hpp.hpp>
#include <nvrtc_kernel_headers/math_constants_h.hpp>
#include <nvrtc_kernel_headers/math_hpp.hpp>
#include <nvrtc_kernel_headers/minmax_op_hpp.hpp>
#include <nvrtc_kernel_headers/ops_hpp.hpp>
#include <nvrtc_kernel_headers/optypes_hpp.hpp>
#include <nvrtc_kernel_headers/shared_hpp.hpp>
#include <nvrtc_kernel_headers/traits_hpp.hpp>
#include <nvrtc_kernel_headers/types_hpp.hpp>
#include <nvrtc_kernel_headers/utility_hpp.hpp>
#include <nvrtc_kernel_headers/version_h.hpp>
#include <optypes.hpp>
#include <platform.hpp>
#include <af/defines.h>
#include <af/version.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>

using std::accumulate;
using std::array;
using std::back_insert_iterator;
using std::begin;
using std::end;
using std::extent;
using std::find_if;
using std::make_pair;
using std::map;
using std::pair;
using std::string;
using std::to_string;
using std::transform;
using std::unique_ptr;
using std::vector;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

spdlog::logger *getLogger() {
    static std::shared_ptr<spdlog::logger> logger(common::loggerFactory("jit"));
    return logger.get();
}

namespace cuda {

using kc_t = map<string, Kernel>;

#ifdef NDEBUG
#define CU_LINK_CHECK(fn)                                                    \
    do {                                                                     \
        CUresult res = fn;                                                   \
        if (res == CUDA_SUCCESS) break;                                      \
        char cu_err_msg[1024 + 48];                                          \
        const char *cu_err_name;                                             \
        cuGetErrorName(res, &cu_err_name);                                   \
        snprintf(cu_err_msg, sizeof(cu_err_msg), "CU Error %s(%d): %s\n",    \
                 cu_err_name, (int)(res), linkError);                        \
        AF_TRACE("Driver API Call: {}\nError Message: {}", #fn, cu_err_msg); \
        AF_ERROR(cu_err_msg, AF_ERR_INTERNAL);                               \
    } while (0)
#else
#define CU_LINK_CHECK(fn) CU_CHECK(fn)
#endif

#ifndef NDEBUG
#define NVRTC_CHECK(fn)                                                 \
    do {                                                                \
        nvrtcResult res = fn;                                           \
        if (res == NVRTC_SUCCESS) break;                                \
        size_t logSize;                                                 \
        nvrtcGetProgramLogSize(prog, &logSize);                         \
        unique_ptr<char[]> log(new char[logSize + 1]);                  \
        char *logptr = log.get();                                       \
        nvrtcGetProgramLog(prog, logptr);                               \
        logptr[logSize] = '\0';                                         \
        AF_TRACE("NVRTC API Call: {}\nError Message: {}", #fn, logptr); \
        AF_ERROR("NVRTC ERROR", AF_ERR_INTERNAL);                       \
    } while (0)
#else
#define NVRTC_CHECK(fn)                                                   \
    do {                                                                  \
        nvrtcResult res = (fn);                                           \
        if (res == NVRTC_SUCCESS) break;                                  \
        char nvrtc_err_msg[1024];                                         \
        snprintf(nvrtc_err_msg, sizeof(nvrtc_err_msg),                    \
                 "NVRTC Error(%d): %s\n", res, nvrtcGetErrorString(res)); \
        AF_TRACE("NVRTC Error Message: {}", nvrtc_err_msg);               \
        AF_ERROR(nvrtc_err_msg, AF_ERR_INTERNAL);                         \
    } while (0)
#endif

void Kernel::setConstant(const char *name, CUdeviceptr src, size_t bytes) {
    CUdeviceptr dst = 0;
    size_t size     = 0;
    CU_CHECK(cuModuleGetGlobal(&dst, &size, prog, name));
    CU_CHECK(cuMemcpyDtoDAsync(dst, src, bytes, getActiveStream()));
}

template<typename T>
void Kernel::setScalar(const char *name, T value) {
    CUdeviceptr dst = 0;
    CU_CHECK(cuModuleGetGlobal(&dst, NULL, prog, name));
    CU_CHECK(cuMemcpyHtoDAsync(dst, &value, sizeof(T), getActiveStream()));
    CU_CHECK(cuStreamSynchronize(getActiveStream()));
}

template<typename T>
void Kernel::getScalar(T &out, const char *name) {
    CUdeviceptr src = 0;
    CU_CHECK(cuModuleGetGlobal(&src, NULL, prog, name));
    CU_CHECK(cuMemcpyDtoHAsync(&out, src, sizeof(T), getActiveStream()));
    CU_CHECK(cuStreamSynchronize(getActiveStream()));
}

template void Kernel::setScalar<int>(const char *, int);
template void Kernel::getScalar<int>(int &, const char *);

string getKernelCacheFilename(const int device, const string &nameExpr) {
    const string mangledName = "KER" + to_string(deterministicHash(nameExpr));

    const auto computeFlag = getComputeCapability(device);
    const string computeVersion =
        to_string(computeFlag.first) + to_string(computeFlag.second);

    return mangledName + "_CU_" + computeVersion + "_AF_" +
           to_string(AF_API_VERSION_CURRENT) + ".cubin";
}

Kernel buildKernel(const int device, const string &nameExpr,
                   const string &jit_ker, const vector<string> &opts,
                   const bool isJIT) {
    const char *ker_name = nameExpr.c_str();

    nvrtcProgram prog;
    if (isJIT) {
        array<const char *, 2> headers = {
            cuda_fp16_hpp,
            cuda_fp16_h,
        };
        array<const char *, 2> header_names = {"cuda_fp16.hpp", "cuda_fp16.h"};
        NVRTC_CHECK(nvrtcCreateProgram(&prog, jit_ker.c_str(), ker_name, 2,
                                       headers.data(), header_names.data()));
    } else {
        constexpr static const char *includeNames[] = {
            "math.h",          // DUMMY ENTRY TO SATISFY cuComplex_h inclusion
            "stdbool.h",       // DUMMY ENTRY TO SATISFY af/defines.h inclusion
            "stdlib.h",        // DUMMY ENTRY TO SATISFY af/defines.h inclusion
            "vector_types.h",  // DUMMY ENTRY TO SATISFY cuComplex_h inclusion
            "backend.hpp",
            "cuComplex.h",
            "jit.cuh",
            "math.hpp",
            "ops.hpp",
            "optypes.hpp",
            "Param.hpp",
            "shared.hpp",
            "types.hpp",
            "cuda_fp16.hpp",
            "cuda_fp16.h",
            "common/half.hpp",
            "common/kernel_type.hpp",
            "af/traits.hpp",
            "interp.hpp",
            "math_constants.h",
            "af/defines.h",
            "af/version.h",
            "utility.hpp",
            "assign_kernel_param.hpp",
            "dims_param.hpp",
            "common/internal_enums.hpp",
            "minmax_op.hpp",
        };

        constexpr size_t NumHeaders = extent<decltype(includeNames)>::value;
        static const std::array<string, NumHeaders> sourceStrings = {{
            string(""),  // DUMMY ENTRY TO SATISFY cuComplex_h inclusion
            string(""),  // DUMMY ENTRY TO SATISFY af/defines.h inclusion
            string(""),  // DUMMY ENTRY TO SATISFY af/defines.h inclusion
            string(""),  // DUMMY ENTRY TO SATISFY cuComplex_h inclusion
            string(backend_hpp, backend_hpp_len),
            string(cuComplex_h, cuComplex_h_len),
            string(jit_cuh, jit_cuh_len),
            string(math_hpp, math_hpp_len),
            string(ops_hpp, ops_hpp_len),
            string(optypes_hpp, optypes_hpp_len),
            string(Param_hpp, Param_hpp_len),
            string(shared_hpp, shared_hpp_len),
            string(types_hpp, types_hpp_len),
            string(cuda_fp16_hpp, cuda_fp16_hpp_len),
            string(cuda_fp16_h, cuda_fp16_h_len),
            string(half_hpp, half_hpp_len),
            string(kernel_type_hpp, kernel_type_hpp_len),
            string(traits_hpp, traits_hpp_len),
            string(interp_hpp, interp_hpp_len),
            string(math_constants_h, math_constants_h_len),
            string(defines_h, defines_h_len),
            string(version_h, version_h_len),
            string(utility_hpp, utility_hpp_len),
            string(assign_kernel_param_hpp, assign_kernel_param_hpp_len),
            string(dims_param_hpp, dims_param_hpp_len),
            string(internal_enums_hpp, internal_enums_hpp_len),
            string(minmax_op_hpp, minmax_op_hpp_len),
        }};

        static const char *headers[] = {
            sourceStrings[0].c_str(),  sourceStrings[1].c_str(),
            sourceStrings[2].c_str(),  sourceStrings[3].c_str(),
            sourceStrings[4].c_str(),  sourceStrings[5].c_str(),
            sourceStrings[6].c_str(),  sourceStrings[7].c_str(),
            sourceStrings[8].c_str(),  sourceStrings[9].c_str(),
            sourceStrings[10].c_str(), sourceStrings[11].c_str(),
            sourceStrings[12].c_str(), sourceStrings[13].c_str(),
            sourceStrings[14].c_str(), sourceStrings[15].c_str(),
            sourceStrings[16].c_str(), sourceStrings[17].c_str(),
            sourceStrings[18].c_str(), sourceStrings[19].c_str(),
            sourceStrings[20].c_str(), sourceStrings[21].c_str(),
            sourceStrings[22].c_str(), sourceStrings[23].c_str(),
            sourceStrings[24].c_str(), sourceStrings[25].c_str(),
            sourceStrings[26].c_str(),
        };
        NVRTC_CHECK(nvrtcCreateProgram(&prog, jit_ker.c_str(), ker_name,
                                       NumHeaders, headers, includeNames));
    }

    auto computeFlag = getComputeCapability(device);
    array<char, 32> arch{};
    snprintf(arch.data(), arch.size(), "--gpu-architecture=compute_%d%d",
             computeFlag.first, computeFlag.second);
    vector<const char *> compiler_options = {
        arch.data(),
        "--std=c++14",
#if !(defined(NDEBUG) || defined(__aarch64__) || defined(__LP64__))
        "--device-debug",
        "--generate-line-info"
#endif
    };
    if (!isJIT) {
        transform(begin(opts), end(opts),
                  back_insert_iterator<vector<const char *>>(compiler_options),
                  [](const std::string &s) { return s.data(); });

        compiler_options.push_back("--device-as-default-execution-space");
        NVRTC_CHECK(nvrtcAddNameExpression(prog, ker_name));
    }

    auto compile = high_resolution_clock::now();
    NVRTC_CHECK(nvrtcCompileProgram(prog, compiler_options.size(),
                                    compiler_options.data()));

    auto compile_end = high_resolution_clock::now();
    size_t ptx_size;
    vector<char> ptx;
    NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptx_size));
    ptx.resize(ptx_size);
    NVRTC_CHECK(nvrtcGetPTX(prog, ptx.data()));

    const size_t linkLogSize    = 1024;
    char linkInfo[linkLogSize]  = {0};
    char linkError[linkLogSize] = {0};

    CUlinkState linkState;
    CUjit_option linkOptions[] = {
        CU_JIT_INFO_LOG_BUFFER, CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        CU_JIT_LOG_VERBOSE};

    void *linkOptionValues[] = {
        linkInfo, reinterpret_cast<void *>(linkLogSize), linkError,
        reinterpret_cast<void *>(linkLogSize), reinterpret_cast<void *>(1)};

    auto link = high_resolution_clock::now();
    CU_LINK_CHECK(cuLinkCreate(5, linkOptions, linkOptionValues, &linkState));

    // cuLinkAddData accounts for most of the time spent linking
    CU_LINK_CHECK(cuLinkAddData(linkState, CU_JIT_INPUT_PTX, (void *)ptx.data(),
                                ptx.size(), ker_name, 0, NULL, NULL));

    void *cubin = nullptr;
    size_t cubinSize;

    CUmodule module;
    CUfunction kernel;
    CU_LINK_CHECK(cuLinkComplete(linkState, &cubin, &cubinSize));
    CU_CHECK(cuModuleLoadDataEx(&module, cubin, 0, 0, 0));
    auto link_end = high_resolution_clock::now();

    const char *name = ker_name;
    if (!isJIT) { NVRTC_CHECK(nvrtcGetLoweredName(prog, ker_name, &name)); }

    CU_CHECK(cuModuleGetFunction(&kernel, module, name));
    Kernel entry = {module, kernel};

#ifdef AF_CACHE_KERNELS
    // save kernel in cache
    const string &cacheDirectory = getCacheDirectory();
    if (!cacheDirectory.empty()) {
        const string cacheFile = cacheDirectory + AF_PATH_SEPARATOR +
                                 getKernelCacheFilename(device, nameExpr);
        const string tempFile =
            cacheDirectory + AF_PATH_SEPARATOR + makeTempFilename();

        // compute CUBIN hash
        const size_t cubinHash = deterministicHash(cubin, cubinSize);

        // write kernel function name and CUBIN binary data
        std::ofstream out(tempFile, std::ios::binary);
        const size_t nameSize = strlen(name);
        out.write(reinterpret_cast<const char *>(&nameSize), sizeof(nameSize));
        out.write(name, nameSize);
        out.write(reinterpret_cast<const char *>(&cubinHash),
                  sizeof(cubinHash));
        out.write(reinterpret_cast<const char *>(&cubinSize),
                  sizeof(cubinSize));
        out.write(static_cast<const char *>(cubin), cubinSize);
        out.close();

        // try to rename temporary file into final cache file, if this fails
        // this means another thread has finished compiling this kernel before
        // the current thread.
        if (!renameFile(tempFile, cacheFile)) { removeFile(tempFile); }
    }
#endif

    CU_LINK_CHECK(cuLinkDestroy(linkState));
    NVRTC_CHECK(nvrtcDestroyProgram(&prog));

    // skip --std=c++14 because it will stay the same. It doesn't
    // provide useful information
    auto listOpts = [](vector<const char *> &in) {
        return accumulate(begin(in) + 2, end(in), string(in[0]),
                          [](const string &lhs, const string &rhs) {
                              return lhs + ", " + rhs;
                          });
    };

    AF_TRACE("{{{:<30} : {{ compile:{:>5} ms, link:{:>4} ms, {{ {} }}, {} }}}}",
             nameExpr,
             duration_cast<milliseconds>(compile_end - compile).count(),
             duration_cast<milliseconds>(link_end - link).count(),
             listOpts(compiler_options), getDeviceProp(device).name);

    return entry;
}

Kernel loadKernel(const int device, const string &nameExpr) {
    const string &cacheDirectory = getCacheDirectory();
    if (cacheDirectory.empty()) return Kernel{nullptr, nullptr};

    const string cacheFile = cacheDirectory + AF_PATH_SEPARATOR +
                             getKernelCacheFilename(device, nameExpr);

    CUmodule module   = nullptr;
    CUfunction kernel = nullptr;

    try {
        std::ifstream in(cacheFile, std::ios::binary);
        if (!in.is_open()) return Kernel{nullptr, nullptr};

        in.exceptions(std::ios::failbit | std::ios::badbit);

        size_t nameSize = 0;
        in.read(reinterpret_cast<char *>(&nameSize), sizeof(nameSize));
        string name;
        name.resize(nameSize);
        in.read(&name[0], nameSize);

        size_t cubinHash = 0;
        in.read(reinterpret_cast<char *>(&cubinHash), sizeof(cubinHash));
        size_t cubinSize = 0;
        in.read(reinterpret_cast<char *>(&cubinSize), sizeof(cubinSize));
        vector<char> cubin(cubinSize);
        in.read(cubin.data(), cubinSize);
        in.close();

        // check CUBIN binary data has not been corrupted
        const size_t recomputedHash =
            deterministicHash(cubin.data(), cubinSize);
        if (recomputedHash != cubinHash) {
            AF_ERROR("cached kernel data is corrupted", AF_ERR_LOAD_SYM);
        }

        CU_CHECK(cuModuleLoadDataEx(&module, cubin.data(), 0, 0, 0));
        CU_CHECK(cuModuleGetFunction(&kernel, module, name.c_str()));

        AF_TRACE("{{{:<30} : loaded from {} for {} }}", nameExpr, cacheFile,
                 getDeviceProp(device).name);

        return Kernel{module, kernel};
    } catch (...) {
        if (module != nullptr) { CU_CHECK(cuModuleUnload(module)); }
        removeFile(cacheFile);
        return Kernel{nullptr, nullptr};
    }
}

kc_t &getCache(int device) {
    thread_local kc_t caches[DeviceManager::MAX_DEVICES];
    return caches[device];
}

void addKernelToCache(int device, const string &nameExpr, Kernel entry) {
    getCache(device).emplace(nameExpr, entry);
}

Kernel findKernel(int device, const string &nameExpr) {
    kc_t &cache = getCache(device);

    auto iter = cache.find(nameExpr);
    if (iter != cache.end()) return iter->second;

#ifdef AF_CACHE_KERNELS
    Kernel kernel = loadKernel(device, nameExpr);
    if (kernel.prog != nullptr && kernel.ker != nullptr) {
        addKernelToCache(device, nameExpr, kernel);
        return kernel;
    }
#endif

    return Kernel{nullptr, nullptr};
}

string getOpEnumStr(af_op_t val) {
    const char *retVal = NULL;
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
    }
#undef CASE_STMT
    return retVal;
}

template<typename T>
string toString(T value) {
    return to_string(value);
}

template string toString<int>(int);
template string toString<long>(long);
template string toString<long long>(long long);
template string toString<unsigned>(unsigned);
template string toString<unsigned long>(unsigned long);
template string toString<unsigned long long>(unsigned long long);
template string toString<float>(float);
template string toString<double>(double);
template string toString<long double>(long double);

template<>
string toString(bool val) {
    return string(val ? "true" : "false");
}

template<>
string toString(af_op_t val) {
    return getOpEnumStr(val);
}

template<>
string toString(const char *val) {
    return string(val);
}

template<>
string toString(af_interp_type val) {
    const char *retVal = NULL;
#define CASE_STMT(v) \
    case v: retVal = #v; break
    switch (val) {
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
string toString(af_border_type val) {
    const char *retVal = NULL;
#define CASE_STMT(v) \
    case v: retVal = #v; break
    switch (val) {
        CASE_STMT(AF_PAD_ZERO);
        CASE_STMT(AF_PAD_SYM);
        CASE_STMT(AF_PAD_CLAMP_TO_EDGE);
        CASE_STMT(AF_PAD_PERIODIC);
    }
#undef CASE_STMT
    return retVal;
}

template<>
string toString(af_moment_type val) {
    const char *retVal = NULL;
#define CASE_STMT(v) \
    case v: retVal = #v; break
    switch (val) {
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
string toString(af_match_type val) {
    const char *retVal = NULL;
#define CASE_STMT(v) \
    case v: retVal = #v; break
    switch (val) {
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
string toString(af_flux_function val) {
    const char *retVal = NULL;
#define CASE_STMT(v) \
    case v: retVal = #v; break
    switch (val) {
        CASE_STMT(AF_FLUX_QUADRATIC);
        CASE_STMT(AF_FLUX_EXPONENTIAL);
        CASE_STMT(AF_FLUX_DEFAULT);
    }
#undef CASE_STMT
    return retVal;
}

template<>
string toString(AF_BATCH_KIND val) {
    const char *retVal = NULL;
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

Kernel getKernel(const string &nameExpr, const string &source,
                 const vector<TemplateArg> &templateArgs,
                 const vector<string> &compileOpts) {
    vector<string> args;
    args.reserve(templateArgs.size());

    transform(templateArgs.begin(), templateArgs.end(),
              std::back_inserter(args),
              [](const TemplateArg &arg) -> string { return arg._tparam; });

    string tInstance = nameExpr + "<" + args[0];
    for (size_t i = 1; i < args.size(); ++i) { tInstance += ("," + args[i]); }
    tInstance += ">";

    int device    = getActiveDeviceId();
    Kernel kernel = findKernel(device, tInstance);

    if (kernel.prog == nullptr || kernel.ker == nullptr) {
        kernel = buildKernel(device, tInstance, source, compileOpts);
        addKernelToCache(device, tInstance, kernel);
    }

    return kernel;
}

}  // namespace cuda
