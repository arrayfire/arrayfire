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
#include <debug_cuda.hpp>
#include <nvrtc/cache.hpp>
#include <nvrtc_kernel_headers/fftconvolve_cuh.hpp>

#include <string>

namespace cuda {
namespace kernel {

static const int THREADS = 256;

static inline std::string fftConvSource() {
    static const std::string src(fftconvolve_cuh, fftconvolve_cuh_len);
    return src;
}

template<typename convT, typename T>
void packDataHelper(Param<convT> sig_packed, Param<convT> filter_packed,
                    CParam<T> sig, CParam<T> filter) {
    auto packData =
        getKernel("cuda::packData", fftConvSource(),
                  {TemplateTypename<convT>(), TemplateTypename<T>()});
    auto padArray =
        getKernel("cuda::padArray", fftConvSource(),
                  {TemplateTypename<convT>(), TemplateTypename<T>()});

    dim_t *sd = sig.dims;

    int sig_packed_elem    = 1;
    int filter_packed_elem = 1;

    for (int i = 0; i < 4; i++) {
        sig_packed_elem *= sig_packed.dims[i];
        filter_packed_elem *= filter_packed.dims[i];
    }

    // Number of packed complex elements in dimension 0
    int sig_half_d0      = divup(sd[0], 2);
    bool sig_half_d0_odd = (sd[0] % 2 == 1);

    dim3 threads(THREADS);
    dim3 blocks(divup(sig_packed_elem, threads.x));

    EnqueueArgs packQArgs(blocks, threads, getActiveStream());

    // Pack signal in a complex matrix where first dimension is half the input
    // (allows faster FFT computation) and pad array to a power of 2 with 0s
    packData(packQArgs, sig_packed, sig, sig_half_d0, sig_half_d0_odd);
    POST_LAUNCH_CHECK();

    blocks = dim3(divup(filter_packed_elem, threads.x));

    EnqueueArgs padQArgs(blocks, threads, getActiveStream());

    // Pad filter array with 0s
    padArray(padQArgs, filter_packed, filter);
    POST_LAUNCH_CHECK();
}

// TODO(umar): This needs a better name
template<typename T, typename convT>
void complexMultiplyHelper(Param<convT> sig_packed, Param<convT> filter_packed,
                           AF_BATCH_KIND kind) {
    auto cplxMul = getKernel("cuda::complexMultiply", fftConvSource(),
                             {TemplateTypename<convT>(), TemplateArg(kind)});

    int sig_packed_elem    = 1;
    int filter_packed_elem = 1;

    for (int i = 0; i < 4; i++) {
        sig_packed_elem *= sig_packed.dims[i];
        filter_packed_elem *= filter_packed.dims[i];
    }

    dim3 threads(THREADS);
    dim3 blocks(divup(sig_packed_elem / 2, threads.x));

    int mul_elem = (sig_packed_elem < filter_packed_elem) ? filter_packed_elem
                                                          : sig_packed_elem;
    blocks = dim3(divup(mul_elem, threads.x));

    EnqueueArgs qArgs(blocks, threads, getActiveStream());
    if (kind == AF_BATCH_RHS) {
        cplxMul(qArgs, filter_packed, sig_packed, filter_packed, mul_elem);
    } else {
        cplxMul(qArgs, sig_packed, sig_packed, filter_packed, mul_elem);
    }
    POST_LAUNCH_CHECK();
}

template<typename T, typename convT>
void reorderOutputHelper(Param<T> out, Param<convT> packed, CParam<T> sig,
                         CParam<T> filter, bool expand, int baseDim) {
    constexpr bool RoundResult = std::is_integral<T>::value;

    auto reorderOut =
        getKernel("cuda::reorderOutput", fftConvSource(),
                  {TemplateTypename<T>(), TemplateTypename<convT>(),
                   TemplateArg(expand), TemplateArg(RoundResult)});

    dim_t *sd    = sig.dims;
    int fftScale = 1;

    // Calculate the scale by which to divide cuFFT results
    for (int k = 0; k < baseDim; k++) fftScale *= packed.dims[k];

    // Number of packed complex elements in dimension 0
    int sig_half_d0 = divup(sd[0], 2);

    dim3 threads(THREADS);
    dim3 blocks(divup(out.strides[3] * out.dims[3], threads.x));

    EnqueueArgs qArgs(blocks, threads, getActiveStream());

    reorderOut(qArgs, out, packed, filter, sig_half_d0, baseDim, fftScale);
    POST_LAUNCH_CHECK();
}

}  // namespace kernel
}  // namespace cuda
