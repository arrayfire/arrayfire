/*******************************************************
 * Copyright (c) 2019, ArrayFire
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

namespace arrayfire {
namespace cuda {

template<typename T>
Array<T> morph(const Array<T> &in, const Array<T> &mask, bool isDilation) {
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

template<typename T>
Array<T> morph3d(const Array<T> &in, const Array<T> &mask, bool isDilation) {
    const dim4 mdims = mask.dims();
    if (mdims[0] != mdims[1] || mdims[0] != mdims[2]) {
        CUDA_NOT_SUPPORTED("Only cubic masks are supported");
    }
    if (mdims[0] > 7) { CUDA_NOT_SUPPORTED("Kernels > 7x7x7 not supported"); }
    Array<T> out = createEmptyArray<T>(in.dims());
    kernel::morph3d<T>(out, in, mask, isDilation);
    return out;
}

#define INSTANTIATE(T)                                                    \
    template Array<T> morph<T>(const Array<T> &, const Array<T> &, bool); \
    template Array<T> morph3d<T>(const Array<T> &, const Array<T> &, bool);

INSTANTIATE(float)
INSTANTIATE(double)
INSTANTIATE(char)
INSTANTIATE(int)
INSTANTIATE(uint)
INSTANTIATE(uchar)
INSTANTIATE(short)
INSTANTIATE(ushort)

}  // namespace cuda
}  // namespace arrayfire
