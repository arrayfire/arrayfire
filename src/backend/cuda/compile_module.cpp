/*******************************************************
 * Copyright (c) 2020, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <common/compile_module.hpp>  //compileModule & loadModuleFromDisk
#include <common/kernel_cache.hpp>    //getKernel(Module&, ...)

#include <Module.hpp>
#include <common/Logger.hpp>
#include <common/deterministicHash.hpp>
#include <common/internal_enums.hpp>
#include <common/util.hpp>
#include <device_manager.hpp>
#include <kernel_headers/jit_cuh.hpp>
#include <nvrtc_kernel_headers/Binary_hpp.hpp>
#include <nvrtc_kernel_headers/Param_hpp.hpp>
#include <nvrtc_kernel_headers/Transform_hpp.hpp>
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

#include <nonstd/span.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using arrayfire::common::getCacheDirectory;
using arrayfire::common::makeTempFilename;
using arrayfire::common::removeFile;
using arrayfire::common::renameFile;
using arrayfire::cuda::getComputeCapability;
using arrayfire::cuda::getDeviceProp;
using detail::Module;
using nonstd::span;
using std::accumulate;
using std::array;
using std::back_insert_iterator;
using std::begin;
using std::end;
using std::extent;
using std::find_if;
using std::make_pair;
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

constexpr size_t linkLogSize = 2048;

#define CU_LINK_CHECK(fn)                                               \
    do {                                                                \
        CUresult res = (fn);                                            \
        if (res == CUDA_SUCCESS) break;                                 \
        array<char, linkLogSize + 512> cu_err_msg;                      \
        const char *cu_err_name;                                        \
        cuGetErrorName(res, &cu_err_name);                              \
        snprintf(cu_err_msg.data(), cu_err_msg.size(),                  \
                 "CU Link Error %s(%d): %s\n", cu_err_name, (int)(res), \
                 linkError);                                            \
        AF_ERROR(cu_err_msg.data(), AF_ERR_INTERNAL);                   \
    } while (0)

#define NVRTC_CHECK(fn)                                                   \
    do {                                                                  \
        nvrtcResult res = (fn);                                           \
        if (res == NVRTC_SUCCESS) break;                                  \
        array<char, 4096> nvrtc_err_msg;                                  \
        snprintf(nvrtc_err_msg.data(), nvrtc_err_msg.size(),              \
                 "NVRTC Error(%d): %s\n", res, nvrtcGetErrorString(res)); \
        AF_ERROR(nvrtc_err_msg.data(), AF_ERR_INTERNAL);                  \
    } while (0)

#define NVRTC_COMPILE_CHECK(fn)                              \
    do {                                                     \
        nvrtcResult res = (fn);                              \
        if (res == NVRTC_SUCCESS) break;                     \
        size_t logSize;                                      \
        nvrtcGetProgramLogSize(prog, &logSize);              \
        vector<char> log(logSize + 1);                       \
        nvrtcGetProgramLog(prog, log.data());                \
        log[logSize] = '\0';                                 \
        array<char, 4096> nvrtc_err_msg;                     \
        snprintf(nvrtc_err_msg.data(), nvrtc_err_msg.size(), \
                 "NVRTC Error(%d): %s\nLog: \n%s\n", res,    \
                 nvrtcGetErrorString(res), log.data());      \
        AF_ERROR(nvrtc_err_msg.data(), AF_ERR_INTERNAL);     \
    } while (0)

spdlog::logger *getLogger() {
    static std::shared_ptr<spdlog::logger> logger(
        arrayfire::common::loggerFactory("jit"));
    return logger.get();
}

string getKernelCacheFilename(const int device, const string &key) {
    const auto computeFlag = getComputeCapability(device);
    const string computeVersion =
        to_string(computeFlag.first) + to_string(computeFlag.second);

    return "KER" + key + "_CU_" + computeVersion + "_AF_" +
           to_string(AF_API_VERSION_CURRENT) + ".bin";
}

namespace arrayfire {
namespace common {

Module compileModule(const string &moduleKey, const span<const string> sources,
                     const span<const string> opts,
                     const span<const string> kInstances,
                     const bool sourceIsJIT) {
    nvrtcProgram prog;
    using namespace arrayfire::cuda;
    if (sourceIsJIT) {
        constexpr const char *header_names[] = {
            "utility",
            "cuda_fp16.hpp",
            "cuda_fp16.h",
        };
        constexpr size_t numHeaders = extent<decltype(header_names)>::value;
        array<const char *, numHeaders> headers = {
            "",
            cuda_fp16_hpp,
            cuda_fp16_h,
        };
        static_assert(headers.size() == numHeaders,
                      "headers array contains fewer sources than header_names");
        NVRTC_CHECK(nvrtcCreateProgram(&prog, sources[0].c_str(),
                                       moduleKey.c_str(), numHeaders,
                                       headers.data(), header_names));
    } else {
        constexpr static const char *includeNames[] = {
            "math.h",          // DUMMY ENTRY TO SATISFY cuComplex_h inclusion
            "stdbool.h",       // DUMMY ENTRY TO SATISFY af/defines.h inclusion
            "stdlib.h",        // DUMMY ENTRY TO SATISFY af/defines.h inclusion
            "vector_types.h",  // DUMMY ENTRY TO SATISFY cuComplex_h inclusion
            "utility",         // DUMMY ENTRY TO SATISFY cuda_fp16.hpp inclusion
            "backend.hpp",
            "cuComplex.h",
            "jit.cuh",
            "math.hpp",
            "optypes.hpp",
            "Param.hpp",
            "shared.hpp",
            "types.hpp",
            "cuda_fp16.hpp",
            "cuda_fp16.h",
            "common/Binary.hpp",
            "common/Transform.hpp",
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

        constexpr size_t numHeaders = extent<decltype(includeNames)>::value;
        static const array<string, numHeaders> sourceStrings = {{
            string(""),  // DUMMY ENTRY TO SATISFY cuComplex_h inclusion
            string(""),  // DUMMY ENTRY TO SATISFY af/defines.h inclusion
            string(""),  // DUMMY ENTRY TO SATISFY af/defines.h inclusion
            string(""),  // DUMMY ENTRY TO SATISFY cuComplex_h inclusion
            string(""),  // DUMMY ENTRY TO SATISFY utility inclusion
            string(backend_hpp, backend_hpp_len),
            string(cuComplex_h, cuComplex_h_len),
            string(jit_cuh, jit_cuh_len),
            string(math_hpp, math_hpp_len),
            string(optypes_hpp, optypes_hpp_len),
            string(Param_hpp, Param_hpp_len),
            string(shared_hpp, shared_hpp_len),
            string(types_hpp, types_hpp_len),
            string(cuda_fp16_hpp, cuda_fp16_hpp_len),
            string(cuda_fp16_h, cuda_fp16_h_len),
            string(Binary_hpp, Binary_hpp_len),
            string(Transform_hpp, Transform_hpp_len),
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
            sourceStrings[26].c_str(), sourceStrings[27].c_str(),
            sourceStrings[28].c_str()};
        static_assert(extent<decltype(headers)>::value == numHeaders,
                      "headers array contains fewer sources than includeNames");
        NVRTC_CHECK(nvrtcCreateProgram(&prog, sources[0].c_str(),
                                       moduleKey.c_str(), numHeaders, headers,
                                       includeNames));
    }

    int device       = getActiveDeviceId();
    auto computeFlag = getComputeCapability(device);
    array<char, 32> arch;
    snprintf(arch.data(), arch.size(), "--gpu-architecture=compute_%d%d",
             computeFlag.first, computeFlag.second);
    vector<const char *> compiler_options = {
        arch.data(),
        "--std=c++14",
        "--device-as-default-execution-space",
#if !(defined(NDEBUG) || defined(__aarch64__) || defined(__LP64__))
        "--device-debug",
        "--generate-line-info"
#endif
    };
    if (!sourceIsJIT) {
        transform(begin(opts), end(opts),
                  back_insert_iterator<vector<const char *>>(compiler_options),
                  [](const string &s) { return s.data(); });

        for (auto &instantiation : kInstances) {
            NVRTC_CHECK(nvrtcAddNameExpression(prog, instantiation.c_str()));
        }
    }

    auto compile = high_resolution_clock::now();
    NVRTC_COMPILE_CHECK(nvrtcCompileProgram(prog, compiler_options.size(),
                                            compiler_options.data()));
    auto compile_end = high_resolution_clock::now();
    size_t ptx_size;
    vector<char> ptx;
    NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptx_size));
    ptx.resize(ptx_size);
    NVRTC_CHECK(nvrtcGetPTX(prog, ptx.data()));

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
                                ptx.size(), moduleKey.c_str(), 0, NULL, NULL));

    void *cubin = nullptr;
    size_t cubinSize;

    CUmodule modOut = nullptr;
    CU_LINK_CHECK(cuLinkComplete(linkState, &cubin, &cubinSize));
    CU_CHECK(cuModuleLoadData(&modOut, cubin));
    auto link_end = high_resolution_clock::now();

    Module retVal(modOut);
    if (!sourceIsJIT) {
        for (auto &instantiation : kInstances) {
            // memory allocated & destroyed by nvrtcProgram for below var
            const char *name = nullptr;
            NVRTC_CHECK(
                nvrtcGetLoweredName(prog, instantiation.c_str(), &name));
            retVal.add(instantiation, string(name, strlen(name)));
        }
    }

#ifdef AF_CACHE_KERNELS_TO_DISK
    // save kernel in cache
    const string &cacheDirectory = getCacheDirectory();
    if (!cacheDirectory.empty()) {
        const string cacheFile = cacheDirectory + AF_PATH_SEPARATOR +
                                 getKernelCacheFilename(device, moduleKey);
        const string tempFile =
            cacheDirectory + AF_PATH_SEPARATOR + makeTempFilename();
        try {
            // write module hash(everything: names, code & options) and CUBIN
            // data
            ofstream out(tempFile, std::ios::binary);
            if (!sourceIsJIT) {
                size_t mangledNamesListSize = retVal.map().size();
                out.write(reinterpret_cast<const char *>(&mangledNamesListSize),
                          sizeof(mangledNamesListSize));
                for (auto &iter : retVal.map()) {
                    size_t kySize   = iter.first.size();
                    size_t vlSize   = iter.second.size();
                    const char *key = iter.first.c_str();
                    const char *val = iter.second.c_str();
                    out.write(reinterpret_cast<const char *>(&kySize),
                              sizeof(kySize));
                    out.write(key, iter.first.size());
                    out.write(reinterpret_cast<const char *>(&vlSize),
                              sizeof(vlSize));
                    out.write(val, iter.second.size());
                }
            }

            // compute CUBIN hash
            const size_t cubinHash = deterministicHash(cubin, cubinSize);

            out.write(reinterpret_cast<const char *>(&cubinHash),
                      sizeof(cubinHash));
            out.write(reinterpret_cast<const char *>(&cubinSize),
                      sizeof(cubinSize));
            out.write(static_cast<const char *>(cubin), cubinSize);
            out.close();

            // try to rename temporary file into final cache file, if this fails
            // this means another thread has finished compiling this kernel
            // before the current thread.
            if (!renameFile(tempFile, cacheFile)) { removeFile(tempFile); }
        } catch (const std::ios_base::failure &e) {
            AF_TRACE("{{{:<30} : failed saving binary to {} for {}, {}}}",
                     moduleKey, cacheFile, getDeviceProp(device).name,
                     e.what());
        }
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
    AF_TRACE("{{ {:<20} : compile:{:>5} ms, link:{:>4} ms, {{ {} }}, {} }}",
             moduleKey,
             duration_cast<milliseconds>(compile_end - compile).count(),
             duration_cast<milliseconds>(link_end - link).count(),
             listOpts(compiler_options), getDeviceProp(device).name);
    return retVal;
}

Module loadModuleFromDisk(const int device, const string &moduleKey,
                          const bool isJIT) {
    const string &cacheDirectory = getCacheDirectory();
    if (cacheDirectory.empty()) return Module{nullptr};

    const string cacheFile = cacheDirectory + AF_PATH_SEPARATOR +
                             getKernelCacheFilename(device, moduleKey);

    CUmodule modOut = nullptr;
    Module retVal{nullptr};
    try {
        std::ifstream in(cacheFile, std::ios::binary);
        if (!in) {
            AF_TRACE("{{{:<20} : Unable to open {} for {}}}", moduleKey,
                     cacheFile, getDeviceProp(device).name);
            removeFile(cacheFile);  // Remove if exists
            return Module{nullptr};
        }
        in.exceptions(std::ios::failbit | std::ios::badbit);

        if (!isJIT) {
            size_t mangledListSize = 0;
            in.read(reinterpret_cast<char *>(&mangledListSize),
                    sizeof(mangledListSize));
            for (size_t i = 0; i < mangledListSize; ++i) {
                size_t keySize = 0;
                in.read(reinterpret_cast<char *>(&keySize), sizeof(keySize));
                vector<char> key;
                key.reserve(keySize);
                in.read(key.data(), keySize);

                size_t itemSize = 0;
                in.read(reinterpret_cast<char *>(&itemSize), sizeof(itemSize));
                vector<char> item;
                item.reserve(itemSize);
                in.read(item.data(), itemSize);

                retVal.add(string(key.data(), keySize),
                           string(item.data(), itemSize));
            }
        }

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
            AF_ERROR("Module on disk seems to be corrupted", AF_ERR_LOAD_SYM);
        }

        CU_CHECK(cuModuleLoadData(&modOut, cubin.data()));

        AF_TRACE("{{{:<20} : loaded from {} for {} }}", moduleKey, cacheFile,
                 getDeviceProp(device).name);

        retVal.set(modOut);
    } catch (const std::ios_base::failure &e) {
        AF_TRACE("{{{:<20} : Unable to read {} for {}}}", moduleKey, cacheFile,
                 getDeviceProp(device).name);
        removeFile(cacheFile);
    } catch (const AfError &e) {
        if (e.getError() == AF_ERR_LOAD_SYM) {
            AF_TRACE(
                "{{{:<20} : Corrupt binary({}) found on disk for {}, removed}}",
                moduleKey, cacheFile, getDeviceProp(device).name);
        } else {
            if (modOut != nullptr) { CU_CHECK(cuModuleUnload(modOut)); }
            AF_TRACE(
                "{{{:<20} : cuModuleLoadData failed with content from {} for "
                "{}, {}}}",
                moduleKey, cacheFile, getDeviceProp(device).name, e.what());
        }
        removeFile(cacheFile);
    }
    return retVal;
}

arrayfire::cuda::Kernel getKernel(const Module &mod, const string &nameExpr,
                                  const bool sourceWasJIT) {
    std::string name  = (sourceWasJIT ? nameExpr : mod.mangledName(nameExpr));
    CUfunction kernel = nullptr;
    CU_CHECK(cuModuleGetFunction(&kernel, mod.get(), name.c_str()));
    return {nameExpr, mod.get(), kernel};
}

}  // namespace common
}  // namespace arrayfire
