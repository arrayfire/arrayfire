/*******************************************************
 * Copyright (c) 2019, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <kernel_headers/jit.hpp>
#include <nvrtc/cache.hpp>
#include <nvrtc_kernel_headers/Param.hpp>
#include <nvrtc_kernel_headers/backend.hpp>
#include <nvrtc_kernel_headers/cuComplex.hpp>
#include <nvrtc_kernel_headers/math.hpp>
#include <nvrtc_kernel_headers/ops.hpp>
#include <nvrtc_kernel_headers/optypes.hpp>
#include <nvrtc_kernel_headers/shared.hpp>
#include <nvrtc_kernel_headers/types.hpp>
#include <optypes.hpp>
#include <platform.hpp>

#include <algorithm>
#include <array>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

using std::array;
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

namespace cuda {

using kc_t = map<string, Kernel>;

#ifndef NDEBUG
#define CU_LINK_CHECK(fn)                                                 \
    do {                                                                  \
        CUresult res = fn;                                                \
        if (res == CUDA_SUCCESS) break;                                   \
        char cu_err_msg[1024];                                            \
        const char* cu_err_name;                                          \
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
        char* logptr = log.get();                      \
        nvrtcGetProgramLog(prog, logptr);              \
        logptr[logSize] = '\x0';                       \
        AF_ERROR("NVRTC ERROR", AF_ERR_INTERNAL);      \
    } while (0)
#else
#define NVRTC_CHECK(fn)                                                   \
    do {                                                                  \
        nvrtcResult res = fn;                                             \
        if (res == NVRTC_SUCCESS) break;                                  \
        char nvrtc_err_msg[1024];                                         \
        snprintf(nvrtc_err_msg, sizeof(nvrtc_err_msg),                    \
                 "NVRTC Error(%d): %s\n", res, nvrtcGetErrorString(res)); \
        AF_ERROR(nvrtc_err_msg, AF_ERR_INTERNAL);                         \
    } while (0)
#endif

void Kernel::setConstant(const char* name, CUdeviceptr src, size_t bytes) {
    CUdeviceptr dst = 0;
    size_t size     = 0;
    CU_CHECK(cuModuleGetGlobal(&dst, &size, prog, name));
    CU_CHECK(cuMemcpyDtoDAsync(dst, src, bytes, getActiveStream()));
}

Kernel buildKernel(const string& nameExpr, const string& jit_ker,
                   const vector<string>& opts, const bool isJIT) {
    const char* ker_name = nameExpr.c_str();

    nvrtcProgram prog;
    if (isJIT) {
        NVRTC_CHECK(nvrtcCreateProgram(&prog, jit_ker.c_str(), ker_name, 0,
                                       NULL, NULL));
    } else {
        constexpr static const char* includeNames[] = {
            "math.h",          // DUMMY ENTRY TO SATISFY cuComplex_h inclusion
            "vector_types.h",  // DUMMY ENTRY TO SATISFY cuComplex_h inclusion
            "backend.hpp",    "complex.hpp", "jit.cuh",
            "math.hpp",       "ops.hpp",     "optypes.hpp",
            "Param.hpp",      "shared.hpp",  "types.hpp"};
        constexpr size_t NumHeaders = extent<decltype(includeNames)>::value;
        static const std::array<string, NumHeaders> sourceStrings = {{
            string(""),  // DUMMY ENTRY TO SATISFY cuComplex_h inclusion
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
        }};

        static const char* headers[] = {
            sourceStrings[0].c_str(), sourceStrings[1].c_str(),
            sourceStrings[2].c_str(), sourceStrings[3].c_str(),
            sourceStrings[4].c_str(), sourceStrings[5].c_str(),
            sourceStrings[6].c_str(), sourceStrings[7].c_str(),
            sourceStrings[8].c_str(), sourceStrings[9].c_str(),
            sourceStrings[10].c_str()};
        NVRTC_CHECK(nvrtcCreateProgram(&prog, jit_ker.c_str(), ker_name,
                                       NumHeaders, headers, includeNames));
    }

    auto computeFlag = getComputeFlag();
    array<char, 32> arch;
    snprintf(arch.data(), arch.size(), "--gpu-architecture=compute_%d%d",
             computeFlag.first, computeFlag.second);
    vector<const char*> compiler_options = {
        arch.data(),
        "--std=c++11",
#if !(defined(NDEBUG) || defined(__aarch64__) || defined(__LP64__))
        "--device-debug",
        "--generate-line-info"
#endif
    };
    if (!isJIT) {
        for (auto& s : opts) { compiler_options.push_back(&s[0]); }
        compiler_options.push_back("--device-as-default-execution-space");
        NVRTC_CHECK(nvrtcAddNameExpression(prog, ker_name));
    }

    NVRTC_CHECK(nvrtcCompileProgram(prog, compiler_options.size(),
                                    compiler_options.data()));
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

    void* linkOptionValues[] = {linkInfo, reinterpret_cast<void*>(linkLogSize),
                                linkError, reinterpret_cast<void*>(linkLogSize),
                                reinterpret_cast<void*>(1)};

    CU_LINK_CHECK(cuLinkCreate(5, linkOptions, linkOptionValues, &linkState));
    CU_LINK_CHECK(cuLinkAddData(linkState, CU_JIT_INPUT_PTX, (void*)ptx.data(),
                                ptx.size(), ker_name, 0, NULL, NULL));

    void* cubin = nullptr;
    size_t cubinSize;

    CUmodule module;
    CUfunction kernel;
    CU_LINK_CHECK(cuLinkComplete(linkState, &cubin, &cubinSize));
    CU_CHECK(cuModuleLoadDataEx(&module, cubin, 0, 0, 0));

    const char* name = ker_name;
    if (!isJIT) { NVRTC_CHECK(nvrtcGetLoweredName(prog, ker_name, &name)); }

    CU_CHECK(cuModuleGetFunction(&kernel, module, name));
    Kernel entry = {module, kernel};

    CU_LINK_CHECK(cuLinkDestroy(linkState));
    NVRTC_CHECK(nvrtcDestroyProgram(&prog));

    return entry;
}

kc_t& getCache(int device) {
    thread_local kc_t caches[DeviceManager::MAX_DEVICES];
    return caches[device];
}

Kernel findKernel(int device, const string nameExpr) {
    kc_t& cache = getCache(device);

    kc_t::iterator iter = cache.find(nameExpr);

    return (iter == cache.end() ? Kernel{0, 0} : iter->second);
}

void addKernelToCache(int device, const string nameExpr, Kernel entry) {
    getCache(device).emplace(nameExpr, entry);
}

string getOpEnumStr(af_op_t val) {
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
string toString(const char* str) {
    return string(str);
}

Kernel getKernel(const string& nameExpr, const string& source,
                 const vector<TemplateArg>& targs,
                 const vector<string>& compileOpts) {
    vector<string> args;
    args.reserve(targs.size());

    transform(targs.begin(), targs.end(), std::back_inserter(args),
              [](const TemplateArg& arg) -> string { return arg._tparam; });

    string tInstance = nameExpr + "<" + args[0];
    for (size_t i = 1; i < args.size(); ++i) { tInstance += ("," + args[i]); }
    tInstance += ">";

    int device    = getActiveDeviceId();
    Kernel kernel = findKernel(device, tInstance);

    if (kernel.prog == 0 || kernel.ker == 0) {
        kernel = buildKernel(tInstance, source, compileOpts);
        addKernelToCache(device, tInstance, kernel);
    }

    return kernel;
}

}  // namespace cuda
