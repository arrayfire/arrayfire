/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <Param.hpp>
#include <common/dispatch.hpp>
#include <common/kernel_cache.hpp>
#include <debug_opencl.hpp>
#include <kernel_headers/fftconvolve_multiply.hpp>
#include <kernel_headers/fftconvolve_pack.hpp>
#include <kernel_headers/fftconvolve_reorder.hpp>
#include <traits.hpp>
#include <af/defines.h>

#include <string>
#include <vector>

namespace arrayfire {
namespace opencl {
namespace kernel {

constexpr int THREADS = 256;

void calcParamSizes(Param& sig_tmp, Param& filter_tmp, Param& packed,
                    Param& sig, Param& filter, const int rank,
                    AF_BATCH_KIND kind) {
    sig_tmp.info.dims[0] = filter_tmp.info.dims[0] = packed.info.dims[0];
    sig_tmp.info.strides[0] = filter_tmp.info.strides[0] = 1;

    for (int k = 1; k < 4; k++) {
        if (k < rank) {
            sig_tmp.info.dims[k]    = packed.info.dims[k];
            filter_tmp.info.dims[k] = packed.info.dims[k];
        } else {
            sig_tmp.info.dims[k]    = sig.info.dims[k];
            filter_tmp.info.dims[k] = filter.info.dims[k];
        }

        sig_tmp.info.strides[k] =
            sig_tmp.info.strides[k - 1] * sig_tmp.info.dims[k - 1];
        filter_tmp.info.strides[k] =
            filter_tmp.info.strides[k - 1] * filter_tmp.info.dims[k - 1];
    }

    // Calculate memory offsets for packed signal and filter
    sig_tmp.data    = packed.data;
    filter_tmp.data = packed.data;

    if (kind == AF_BATCH_RHS) {
        filter_tmp.info.offset = 0;
        sig_tmp.info.offset =
            filter_tmp.info.strides[3] * filter_tmp.info.dims[3] * 2;
    } else {
        sig_tmp.info.offset = 0;
        filter_tmp.info.offset =
            sig_tmp.info.strides[3] * sig_tmp.info.dims[3] * 2;
    }
}

template<typename convT, typename T>
void packDataHelper(Param packed, Param sig, Param filter, const int rank,
                    AF_BATCH_KIND kind) {
    constexpr bool IsTypeDouble = std::is_same<T, double>::value;
    constexpr auto ctDType =
        static_cast<af_dtype>(dtype_traits<convT>::af_type);

    std::vector<TemplateArg> targs = {
        TemplateTypename<T>(),
        TemplateTypename<convT>(),
        TemplateArg(IsTypeDouble),
    };
    std::vector<std::string> options = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
    };
    if (ctDType == c32) {
        options.emplace_back(DefineKeyValue(CONVT, "float"));
    } else if (ctDType == c64 && IsTypeDouble) {
        options.emplace_back(DefineKeyValue(CONVT, "double"));
    }
    options.emplace_back(getTypeBuildDefinition<T, convT>());

    auto packData = common::getKernel("pack_data", {{fftconvolve_pack_cl_src}},
                                      targs, options);
    auto padArray = common::getKernel("pad_array", {{fftconvolve_pack_cl_src}},
                                      targs, options);

    Param sig_tmp, filter_tmp;
    calcParamSizes(sig_tmp, filter_tmp, packed, sig, filter, rank, kind);

    int sig_packed_elem = sig_tmp.info.strides[3] * sig_tmp.info.dims[3];
    int filter_packed_elem =
        filter_tmp.info.strides[3] * filter_tmp.info.dims[3];

    // Number of packed complex elements in dimension 0
    int sig_half_d0     = divup(sig.info.dims[0], 2);
    int sig_half_d0_odd = sig.info.dims[0] % 2;

    int blocks = divup(sig_packed_elem, THREADS);

    // Locate features kernel sizes
    cl::NDRange local(THREADS);
    cl::NDRange global(blocks * THREADS);

    // Pack signal in a complex matrix where first dimension is half the input
    // (allows faster FFT computation) and pad array to a power of 2 with 0s
    packData(cl::EnqueueArgs(getQueue(), global, local), *sig_tmp.data,
             sig_tmp.info, *sig.data, sig.info, sig_half_d0, sig_half_d0_odd);
    CL_DEBUG_FINISH(getQueue());

    blocks = divup(filter_packed_elem, THREADS);
    global = cl::NDRange(blocks * THREADS);

    // Pad filter array with 0s
    padArray(cl::EnqueueArgs(getQueue(), global, local), *filter_tmp.data,
             filter_tmp.info, *filter.data, filter.info);
    CL_DEBUG_FINISH(getQueue());
}

template<typename convT, typename T>
void complexMultiplyHelper(Param packed, Param sig, Param filter,
                           const int rank, AF_BATCH_KIND kind) {
    constexpr bool IsTypeDouble = std::is_same<T, double>::value;
    constexpr auto ctDType =
        static_cast<af_dtype>(dtype_traits<convT>::af_type);

    std::vector<TemplateArg> targs = {
        TemplateTypename<T>(),
        TemplateTypename<convT>(),
        TemplateArg(IsTypeDouble),
    };
    std::vector<std::string> options = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
        DefineKeyValue(AF_BATCH_NONE, static_cast<int>(AF_BATCH_NONE)),
        DefineKeyValue(AF_BATCH_LHS, static_cast<int>(AF_BATCH_LHS)),
        DefineKeyValue(AF_BATCH_RHS, static_cast<int>(AF_BATCH_RHS)),
        DefineKeyValue(AF_BATCH_SAME, static_cast<int>(AF_BATCH_SAME)),
    };
    if (ctDType == c32) {
        options.emplace_back(DefineKeyValue(CONVT, "float"));
    } else if (ctDType == c64 && IsTypeDouble) {
        options.emplace_back(DefineKeyValue(CONVT, "double"));
    }
    options.emplace_back(getTypeBuildDefinition<T, convT>());

    auto cplxMul = common::getKernel(
        "complex_multiply", {{fftconvolve_multiply_cl_src}}, targs, options);

    Param sig_tmp, filter_tmp;
    calcParamSizes(sig_tmp, filter_tmp, packed, sig, filter, rank, kind);

    int sig_packed_elem = sig_tmp.info.strides[3] * sig_tmp.info.dims[3];
    int filter_packed_elem =
        filter_tmp.info.strides[3] * filter_tmp.info.dims[3];
    int mul_elem = (sig_packed_elem < filter_packed_elem) ? filter_packed_elem
                                                          : sig_packed_elem;
    int blocks   = divup(mul_elem, THREADS);

    cl::NDRange local(THREADS);
    cl::NDRange global(blocks * THREADS);

    // Multiply filter and signal FFT arrays
    cplxMul(cl::EnqueueArgs(getQueue(), global, local), *packed.data,
            packed.info, *sig_tmp.data, sig_tmp.info, *filter_tmp.data,
            filter_tmp.info, mul_elem, (int)kind);
    CL_DEBUG_FINISH(getQueue());
}

template<typename T, typename convT>
void reorderOutputHelper(Param out, Param packed, Param sig, Param filter,
                         const int rank, AF_BATCH_KIND kind, bool expand) {
    constexpr bool IsTypeDouble = std::is_same<T, double>::value;
    constexpr auto ctDType =
        static_cast<af_dtype>(dtype_traits<convT>::af_type);
    constexpr bool RoundResult = std::is_integral<T>::value;

    std::vector<TemplateArg> targs = {
        TemplateTypename<T>(),     TemplateTypename<convT>(),
        TemplateArg(IsTypeDouble), TemplateArg(RoundResult),
        TemplateArg(expand),
    };
    std::vector<std::string> options = {
        DefineKeyValue(T, dtype_traits<T>::getName()),
        DefineKeyValue(ROUND_OUT, static_cast<int>(RoundResult)),
        DefineKeyValue(EXPAND, static_cast<int>(expand)),
    };
    if (ctDType == c32) {
        options.emplace_back(DefineKeyValue(CONVT, "float"));
    } else if (ctDType == c64 && IsTypeDouble) {
        options.emplace_back(DefineKeyValue(CONVT, "double"));
    }
    options.emplace_back(getTypeBuildDefinition<T, convT>());

    auto reorder = common::getKernel(
        "reorder_output", {{fftconvolve_reorder_cl_src}}, targs, options);

    int fftScale = 1;

    // Calculate the scale by which to divide clFFT results
    for (int k = 0; k < rank; k++) fftScale *= packed.info.dims[k];

    Param sig_tmp, filter_tmp;
    calcParamSizes(sig_tmp, filter_tmp, packed, sig, filter, rank, kind);

    // Number of packed complex elements in dimension 0
    int sig_half_d0 = divup(sig.info.dims[0], 2);

    int blocks = divup(out.info.strides[3] * out.info.dims[3], THREADS);

    cl::NDRange local(THREADS);
    cl::NDRange global(blocks * THREADS);

    if (kind == AF_BATCH_RHS) {
        reorder(cl::EnqueueArgs(getQueue(), global, local), *out.data, out.info,
                *filter_tmp.data, filter_tmp.info, filter.info, sig_half_d0,
                rank, fftScale);
    } else {
        reorder(cl::EnqueueArgs(getQueue(), global, local), *out.data, out.info,
                *sig_tmp.data, sig_tmp.info, filter.info, sig_half_d0, rank,
                fftScale);
    }
    CL_DEBUG_FINISH(getQueue());
}
}  // namespace kernel
}  // namespace opencl
}  // namespace arrayfire
