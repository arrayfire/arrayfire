/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/dim4.hpp>
#include <Array.hpp>
#include <math.hpp>
#include <morph.hpp>
#include <kernel/morph.hpp>
#include <err_opencl.hpp>

using af::dim4;

namespace opencl
{
template<typename T, bool isDilation>
Array<T> morph(const Array<T> &in, const Array<T> &mask)
{
    const dim4 mdims    = mask.dims();

    if (mdims[0]!=mdims[1])
        OPENCL_NOT_SUPPORTED("Rectangular masks are not suported");

    if (mdims[0]>19)
        OPENCL_NOT_SUPPORTED("Kernels > 19x19 are not supported");

    const dim4 dims = in.dims();
    Array<T> out   = createEmptyArray<T>(dims);

    switch(mdims[0]) {
        case  2: kernel::morph<T, isDilation,  2>(out, in, mask); break;
        case  3: kernel::morph<T, isDilation,  3>(out, in, mask); break;
        case  4: kernel::morph<T, isDilation,  4>(out, in, mask); break;
        case  5: kernel::morph<T, isDilation,  5>(out, in, mask); break;
        case  6: kernel::morph<T, isDilation,  6>(out, in, mask); break;
        case  7: kernel::morph<T, isDilation,  7>(out, in, mask); break;
        case  8: kernel::morph<T, isDilation,  8>(out, in, mask); break;
        case  9: kernel::morph<T, isDilation,  9>(out, in, mask); break;
        case 10: kernel::morph<T, isDilation, 10>(out, in, mask); break;
        case 11: kernel::morph<T, isDilation, 11>(out, in, mask); break;
        case 12: kernel::morph<T, isDilation, 12>(out, in, mask); break;
        case 13: kernel::morph<T, isDilation, 13>(out, in, mask); break;
        case 14: kernel::morph<T, isDilation, 14>(out, in, mask); break;
        case 15: kernel::morph<T, isDilation, 15>(out, in, mask); break;
        case 16: kernel::morph<T, isDilation, 16>(out, in, mask); break;
        case 17: kernel::morph<T, isDilation, 17>(out, in, mask); break;
        case 18: kernel::morph<T, isDilation, 18>(out, in, mask); break;
        case 19: kernel::morph<T, isDilation, 19>(out, in, mask); break;
        default: OPENCL_NOT_SUPPORTED(); break;
    }

    return out;
}

#define INSTANTIATE(T, ISDILATE)                                                 \
    template Array<T> morph  <T, ISDILATE>(const Array<T> &in, const Array<T> &mask);
}
