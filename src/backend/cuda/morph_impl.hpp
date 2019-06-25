/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <err_cuda.hpp>
#include <kernel/morph.hpp>
#include <morph.hpp>
#include <af/dim4.hpp>

using af::dim4;

namespace cuda {
template<typename T, bool isDilation>
Array<T> morph(const Array<T> &in, const Array<T> &mask) {
    const dim4 mdims = mask.dims();
    if (mdims[0] != mdims[1]) {
        CUDA_NOT_SUPPORTED("Rectangular masks are not supported");
    }
    if (mdims[0] > 19) {
        CUDA_NOT_SUPPORTED("Kernels > 19x19 are not supported");
    }
    Array<T> out = createEmptyArray<T>(in.dims());
    kernel::morph<T>(out, in, mask, isDilation);
    return out;
}

#define INSTANTIATE(T, ISDILATE)                             \
    template Array<T> morph<T, ISDILATE>(const Array<T> &in, \
                                         const Array<T> &mask);
}  // namespace cuda
