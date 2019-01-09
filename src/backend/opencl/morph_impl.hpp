/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <Array.hpp>
#include <err_opencl.hpp>
#include <kernel/morph.hpp>
#include <math.hpp>
#include <morph.hpp>
#include <af/dim4.hpp>

using af::dim4;

namespace opencl {
template <typename T, bool isDilation>
Array<T> morph(const Array<T> &in, const Array<T> &mask) {
    const dim4 mdims = mask.dims();

    if (mdims[0] != mdims[1])
        OPENCL_NOT_SUPPORTED("Rectangular masks are not suported");

    if (mdims[0] > 19)
        OPENCL_NOT_SUPPORTED("Kernels > 19x19 are not supported");

    const dim4 dims = in.dims();
    Array<T> out    = createEmptyArray<T>(dims);

    switch (mdims[0]) {
        case 2: kernel::morph<T, isDilation, 2>(out, in, mask); break;
        case 3: kernel::morph<T, isDilation, 3>(out, in, mask); break;
        case 4: kernel::morph<T, isDilation, 4>(out, in, mask); break;
        case 5: kernel::morph<T, isDilation, 5>(out, in, mask); break;
        case 6: kernel::morph<T, isDilation, 6>(out, in, mask); break;
        case 7: kernel::morph<T, isDilation, 7>(out, in, mask); break;
        case 8: kernel::morph<T, isDilation, 8>(out, in, mask); break;
        case 9: kernel::morph<T, isDilation, 9>(out, in, mask); break;
        case 10: kernel::morph<T, isDilation, 10>(out, in, mask); break;
        default: kernel::morph<T, isDilation>(out, in, mask, mdims[0]); break;
    }

    return out;
}

#define INSTANTIATE(T, ISDILATE)                             \
    template Array<T> morph<T, ISDILATE>(const Array<T> &in, \
                                         const Array<T> &mask);
}  // namespace opencl
