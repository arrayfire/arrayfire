/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/compile_kernel.hpp>

#include <Kernel.hpp>
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

#include <nvrtc.h>

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

using namespace cuda;

using detail::Kernel;
using std::accumulate;
using std::array;
using std::back_insert_iterator;
using std::begin;
using std::end;
using std::extent;
using std::find_if;
using std::make_pair;
using std::map;
using std::ofstream;
using std::pair;
using std::string;
using std::to_string;
using std::transform;
using std::unique_ptr;
using std::vector;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

#ifdef NDEBUG
#define CU_LINK_CHECK(fn)                                                 \
    do {                                                                  \
        CUresult res = fn;                                                \
        if (res == CUDA_SUCCESS) break;                                   \
        char cu_err_msg[2048];                                            \
        const char *cu_err_name;                                          \
        cuGetErrorName(res, &cu_err_name);                                \
        snprintf(cu_err_msg, sizeof(cu_err_msg), "CU Error %s(%d): %s\n", \
                 cu_err_name, (int)(res), linkError);                     \
        AF_ERROR(cu_err_msg, AF_ERR_INTERNAL);                            \
    } while (0)
#else
#define CU_LINK_CHECK(fn) CU_CHECK(fn)
#endif

#ifndef NDEBUG
#define NVRTC_CHECK(fn)                                \
    do {                                               \
        nvrtcResult res = fn;                          \
        if (res == NVRTC_SUCCESS) break;               \
        size_t logSize;                                \
        nvrtcGetProgramLogSize(prog, &logSize);        \
        unique_ptr<char[]> log(new char[logSize + 1]); \
        char *logptr = log.get();                      \
        nvrtcGetProgramLog(prog, logptr);              \
        logptr[logSize] = '\x0';                       \
        puts(logptr);                                  \
        AF_ERROR("NVRTC ERROR", AF_ERR_INTERNAL);      \
    } while (0)
#else
#define NVRTC_CHECK(fn)                                                   \
    do {                                                                  \
        nvrtcResult res = (fn);                                           \
        if (res == NVRTC_SUCCESS) break;                                  \
        char nvrtc_err_msg[2048];                                         \
        snprintf(nvrtc_err_msg, sizeof(nvrtc_err_msg),                    \
                 "NVRTC Error(%d): %s\n", res, nvrtcGetErrorString(res)); \
        AF_ERROR(nvrtc_err_msg, AF_ERR_INTERNAL);                         \
    } while (0)
#endif

spdlog::logger *getLogger() {
    static std::shared_ptr<spdlog::logger> logger(common::loggerFactory("jit"));
    return logger.get();
}

string getKernelCacheFilename(const int device, const string &nameExpr,
                              const vector<string> &sources) {
    const string srcs =
        accumulate(sources.begin(), sources.end(), std::string(""));
    const string mangledName =
        "KER" + to_string(deterministicHash(nameExpr + srcs));

    const auto computeFlag = getComputeCapability(device);
    const string computeVersion =
        to_string(computeFlag.first) + to_string(computeFlag.second);

    return mangledName + "_CU_" + computeVersion + "_AF_" +
           to_string(AF_API_VERSION_CURRENT) + ".cubin";
}

namespace common {

Kernel compileKernel(const string &kernelName, const string &nameExpr,
                     const vector<string> &sources, const vector<string> &opts,
                     const bool isJIT) {
    auto &jit_ker        = sources[0];
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
        static const array<string, NumHeaders> sourceStrings = {{
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

    int device       = cuda::getActiveDeviceId();
    auto computeFlag = cuda::getComputeCapability(device);
    array<char, 32> arch;
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
                  [](const string &s) { return s.data(); });

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

#ifdef AF_CACHE_KERNELS_TO_DISK
    // save kernel in cache
    const string &cacheDirectory = getCacheDirectory();
    if (!cacheDirectory.empty()) {
        const string cacheFile =
            cacheDirectory + AF_PATH_SEPARATOR +
            getKernelCacheFilename(device, nameExpr, sources);
        const string tempFile =
            cacheDirectory + AF_PATH_SEPARATOR + makeTempFilename();

        // compute CUBIN hash
        const size_t cubinHash = deterministicHash(cubin, cubinSize);

        // write kernel function name and CUBIN binary data
        ofstream out(tempFile, std::ios::binary);
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

Kernel loadKernel(const int device, const string &nameExpr,
                  const vector<string> &sources) {
    const string &cacheDirectory = getCacheDirectory();
    if (cacheDirectory.empty()) return Kernel{nullptr, nullptr};

    const string cacheFile = cacheDirectory + AF_PATH_SEPARATOR +
                             getKernelCacheFilename(device, nameExpr, sources);

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

}  // namespace common
