/*******************************************************
 * Copyright (c) 2023, ArrayFire
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
#include <debug_oneapi.hpp>
#include <kernel/accessors.hpp>
#include <af/defines.h>

#include <string>
#include <vector>

namespace arrayfire {
namespace oneapi {
namespace kernel {

constexpr int THREADS = 256;

template<typename T, typename convT>
void calcParamSizes(Param<T>& sig_tmp, Param<T>& filter_tmp,
                    Param<convT>& packed, Param<T>& sig, Param<T>& filter,
                    const int rank, AF_BATCH_KIND kind) {
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

    // NOTE: The OpenCL implementation on which this oneAPI port is
    // based treated the incoming `packed` buffer as a string of real
    // scalars instead of complex numbers. OpenCL accomplished this
    // with the hack depicted in the trailing two lines. This note
    // remains here in an explanation of SYCL buffer reinterpret's in
    // fftconvolve kernel invocations.

    // sig_tmp.data    = packed.data;
    // filter_tmp.data = packed.data;

    // Calculate memory offsets for packed signal and filter
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

}  // namespace kernel
}  // namespace oneapi
}  // namespace arrayfire
