/*******************************************************
 * Copyright (c) 2022, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <arith.hpp>
#include <convolve.hpp>
#include <err_oneapi.hpp>
#include <iir.hpp>
#include <kernel/iir.hpp>
#include <math.hpp>
#include <af/dim4.hpp>

using af::dim4;

namespace arrayfire {
namespace oneapi {
template<typename T>
Array<T> iir(const Array<T> &b, const Array<T> &a, const Array<T> &x) {
    AF_BATCH_KIND type = x.ndims() == 1 ? AF_BATCH_NONE : AF_BATCH_SAME;
    if (x.ndims() != b.ndims()) {
        type = (x.ndims() < b.ndims()) ? AF_BATCH_RHS : AF_BATCH_LHS;
    }

    // Extract the first N elements
    Array<T> c = convolve<T, T>(x, b, type, 1, true);
    dim4 cdims = c.dims();
    cdims[0]   = x.dims()[0];
    c.resetDims(cdims);

    int num_a = a.dims()[0];

    if (num_a == 1) { return c; }

    size_t local_bytes_req = (num_a * 2 + 1) * sizeof(T);
    if (local_bytes_req >
        getDevice().get_info<sycl::info::device::local_mem_size>()) {
        char errMessage[256];
        snprintf(errMessage, sizeof(errMessage),
                 "\ncurrent OneAPI device does not have sufficient local "
                 "memory,\n"
                 "for iir kernel, %zu(required) > %zu(available)\n",
                 local_bytes_req,
                 getDevice().get_info<sycl::info::device::local_mem_size>());
        AF_ERROR(errMessage, AF_ERR_RUNTIME);
    }

    dim4 ydims = c.dims();
    Array<T> y = createEmptyArray<T>(ydims);

    if (a.ndims() > 1) {
        kernel::iir<T, true>(y, c, a);
    } else {
        kernel::iir<T, false>(y, c, a);
    }
    return y;
}

#define INSTANTIATE(T)                                          \
    template Array<T> iir(const Array<T> &b, const Array<T> &a, \
                          const Array<T> &x);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(cfloat)
INSTANTIATE(cdouble)
}  // namespace oneapi
}  // namespace arrayfire
